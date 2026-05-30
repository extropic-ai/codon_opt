"""Potts model creation for codon optimization."""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PyTree

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models import DiscreteEBMInteraction
from thrml.models.discrete_ebm import CategoricalEBMFactor, CategoricalGibbsConditional
from thrml.pgm import CategoricalNode

from problem import (
    MAX_SYNONYMOUS_CODONS,
    CodonProblem,
    adapt_gc_coeff,
    compute_pairwise_penalties,
    compute_unary_biases,
    create_gc_counter,
)


def create_base_codon_model(problem: CodonProblem, linear_gc_coeff: Float[ArrayLike, ""] = jnp.array(0.0)):
    """Create a Potts model for the codon optimization problem.

    Uses a 2-coloring of the chain (even/odd positions) for block Gibbs sampling.

    Args:
        problem: The codon optimization problem.
        linear_gc_coeff: Coefficient for linear GC term. Positive encourages GC, negative penalizes.

    Returns:
        Tuple of (program, free_blocks, all_nodes, all_nodes_block) where:
        - program: FactorSamplingProgram ready for sampling
        - free_blocks: List of blocks for init state
        - all_nodes: List of all CategoricalNode objects
        - all_nodes_block: Block containing all nodes (for sampling output)
    """
    L = problem.seq_length
    n_cats = MAX_SYNONYMOUS_CODONS

    # Create nodes
    nodes = [CategoricalNode() for _ in range(L)]

    # Compute biases and penalties
    biases, _ = compute_unary_biases(problem, linear_gc_coeff)
    assert biases.shape == (L, n_cats), f"{biases.shape} != {(L, n_cats)}"

    pairwise = jnp.array(compute_pairwise_penalties(problem))
    assert pairwise.shape == (L - 1, n_cats, n_cats), f"{pairwise.shape} != {(L - 1, n_cats, n_cats)}"

    # Create factors
    bias_factor = CategoricalEBMFactor([Block(nodes)], biases)

    # Pairwise factor: connects node[i] to node[i+1]
    pairwise_factor = CategoricalEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], pairwise)

    factors = [bias_factor, pairwise_factor]

    # Block structure: 2-coloring for chain (even/odd positions)
    even_nodes = [nodes[i] for i in range(0, L, 2)]
    odd_nodes = [nodes[i] for i in range(1, L, 2)]

    free_blocks = []
    samplers = []
    sampler = CategoricalGibbsConditional(n_cats)

    if even_nodes:
        free_blocks.append(Block(even_nodes))
        samplers.append(sampler)
    if odd_nodes:
        free_blocks.append(Block(odd_nodes))
        samplers.append(sampler)

    spec = BlockGibbsSpec(free_blocks, [])
    program = FactorSamplingProgram(spec, samplers, factors, [])

    # Block with all nodes in order (for extracting samples)
    all_nodes_block = Block(nodes)

    return program, free_blocks, nodes, all_nodes_block


def create_init_state(problem: CodonProblem, key: Array) -> list[Array]:
    """Create random initial state for the free blocks.

    Samples uniformly from valid codons at each position.
    """
    L = problem.seq_length

    # Generate random valid indices for each position
    indices = []
    for i in range(L):
        key, subkey = jax.random.split(key)
        n_valid = problem.num_codons_at(i)
        idx = jax.random.randint(subkey, (), 0, n_valid, dtype=jnp.uint8)
        indices.append(idx)
    indices = jnp.stack(indices)

    # Split into even/odd blocks
    even_indices = indices[0::2]
    odd_indices = indices[1::2]

    init_state = []
    if len(even_indices) > 0:
        init_state.append(even_indices)
    if len(odd_indices) > 0:
        init_state.append(odd_indices)

    return init_state


def update_program_with_gc_coeff(
    program: FactorSamplingProgram,
    new_gc_coeff_times_beta: Float[Array, ""],
    editable_params: FactorSamplingProgram,
):
    """Update program weights by adding new_gc_coeff_times_beta * editable_params.

    Only float32 arrays are updated; integer arrays (e.g., indices) are preserved.
    """

    def update_fn(w, dw):
        if isinstance(w, jnp.ndarray) and w.dtype == jnp.float32:
            return w + new_gc_coeff_times_beta * dw
        return w

    updated_program = jax.tree.map(update_fn, program, editable_params, is_leaf=eqx.is_array)

    return updated_program


@eqx.filter_jit
def _run_sampling_jit(
    key: Array,
    n_chains: int,
    program,
    schedule: SamplingSchedule,
    all_nodes_block,
    n_codons_per_position: Optional[Array],
    init_states=None,
    gc_coeffs_times_beta=None,
    editable_params=None,
) -> Array:
    """JIT-compiled inner sampling loop.

    Separated from run_sampling to avoid hashing issues with CodonProblem.
    """

    def sample_single_chain(chain_key, init_state, new_gc_coeff_times_beta):
        init_key, sample_key = jax.random.split(chain_key)
        if init_state is None:
            assert n_codons_per_position is not None, "n_codons_per_position must be provided if init_states is None"
            init_state = _create_init_state_jit(init_key, n_codons_per_position)

        if new_gc_coeff_times_beta is not None:
            assert editable_params is not None, (
                "editable_params must be provided if" " new_gc_coeffs_times_beta is not None"
            )
            updated_program = update_program_with_gc_coeff(program, new_gc_coeff_times_beta, editable_params)
        else:
            updated_program = program

        samples = sample_states(
            key=sample_key,
            program=updated_program,
            schedule=schedule,
            init_state_free=init_state,
            state_clamp=[],
            nodes_to_sample=[all_nodes_block],
        )
        return samples[0]

    keys = jax.random.split(key, n_chains)

    out = jax.vmap(sample_single_chain)(keys, init_states, gc_coeffs_times_beta)

    correct_shape = (n_chains, schedule.n_samples, len(all_nodes_block))
    assert out.shape == correct_shape, f"{out.shape} != {correct_shape}"
    return out


def _create_init_state_jit(key: Array, n_codons_per_position: Array) -> list[Array]:
    """Create random initial state (helper if we don't have access to `problem`).

    Args:
        key: JAX random key.
        n_codons_per_position: Array of shape (L,) with number of valid codons at each position.

    Returns:
        List of arrays for even/odd blocks.
    """
    L = n_codons_per_position.shape[0]

    # Generate random indices for each position
    keys = jax.random.split(key, L)
    indices = jax.vmap(lambda k, n: jax.random.randint(k, (), 0, n, dtype=jnp.uint8))(keys, n_codons_per_position)

    # Split into even/odd blocks
    even_indices = indices[0::2]
    odd_indices = indices[1::2]

    init_state = [even_indices]
    if odd_indices.shape[0] > 0:
        init_state.append(odd_indices)

    return init_state


def scale_model_by_beta(
    program: FactorSamplingProgram,
    beta: Float[ArrayLike, ""],
) -> FactorSamplingProgram:
    """Scale the model's biases and pairwise terms by beta.

    Args:
        program: The original FactorSamplingProgram.
        beta: Inverse temperature.
    Returns:
        New FactorSamplingProgram with scaled parameters.
    """

    def is_interaction(x):
        return isinstance(x, DiscreteEBMInteraction)

    get_weights = lambda prog: [x.weights for x in jax.tree.leaves(prog, is_interaction) if is_interaction(x)]

    replace_fn = lambda w: w * beta

    return eqx.tree_at(get_weights, program, replace_fn=replace_fn)


def run_sampling(
    problem: CodonProblem,
    key: Array,
    n_chains: int,
    schedule: SamplingSchedule,
    beta: Float[ArrayLike, ""] = jnp.array(1.0),
) -> Array:
    """Run Gibbs sampling on the codon Potts model.

    Creates the model, then runs JIT-compiled vmapped sampling.

    Args:
        problem: The codon optimization problem.
        key: JAX random key.
        n_chains: Number of parallel chains (vmapped).
        schedule: SamplingSchedule specifying burnin, n_samples, thinning.
        beta: Inverse temperature.

    Returns:
        Array of shape (n_chains, n_samples, L) with sampled codon indices.
    """
    # Create model (not jitted - runs once)
    base_program, free_blocks, nodes, all_nodes_block = create_base_codon_model(problem)

    # Scale biases and pairwise terms by beta
    program = scale_model_by_beta(base_program, beta)

    # Precompute number of valid codons per position for init state
    n_codons_per_position = jnp.array([problem.num_codons_at(i) for i in range(problem.seq_length)])

    # Run jitted sampling
    return _run_sampling_jit(key, n_chains, program, schedule, all_nodes_block, n_codons_per_position)


def array_subtract(a, b):
    """Subtract two arrays if both are float32, otherwise return zeros or the original value.

    Used to compute the difference between two programs to isolate the effect of linear_gc_coeff.
    """
    if isinstance(a, jnp.ndarray) and isinstance(b, jnp.ndarray):
        if a.dtype == b.dtype == jnp.float32:
            return a - b
        return jnp.zeros_like(a)
    return a


@eqx.filter_jit
def run_annealing(
    problem: CodonProblem,
    key: Array,
    n_chains: int,
    steps_per_beta: int,
    betas: Float[Array, "n_betas"],
    gc_coeff_adapt_mult: Optional[Float[Array, ""]],
    only_return_final: bool,
    metrics: Optional[Callable[[Array], PyTree[Array]]] = None,
) -> tuple[Array, dict[str, Array], Optional[PyTree[Array]]]:
    """Run simulated annealing on the codon Potts model.

    Performs Gibbs sampling while gradually increasing beta (inverse temperature)
    according to the provided schedule.

    Args:
        problem: The codon optimization problem.
        key: JAX random key.
        n_chains: Number of parallel chains (vmapped).
        steps_per_beta: Number of Gibbs sweeps at each beta value.
        betas: Array of inverse temperatures, typically increasing.
        gc_coeff_adapt_mult: Multiplier for GC coefficient adaptation. If None, no adaptation.
        only_return_final: If True, return only the final states.
        metrics: Optional function to compute additional metrics from states.

    Returns:
        Tuple of (states, stats, metric_outs).
    """
    # Create model
    base_program, free_blocks, nodes, all_nodes_block = create_base_codon_model(problem)

    # Compute which weight values in the program depend on linear_gc_coeff
    prog1, _, _, _ = create_base_codon_model(problem, linear_gc_coeff=jnp.array(1.0))
    prog_diff = jax.tree.map(array_subtract, prog1, base_program, is_leaf=eqx.is_array)

    vec_gc_counter = jax.jit(jax.vmap(create_gc_counter(problem)))

    def run_single_beta_section(carry, inputs):
        """Run Gibbs sampling at a single beta value."""
        key, beta = inputs
        prev_states, gc_coeffs = carry
        schedule = SamplingSchedule(n_warmup=steps_per_beta, n_samples=1, steps_per_sample=1)

        gc_fractions = vec_gc_counter(prev_states) / (3 * problem.seq_length)
        if gc_coeff_adapt_mult is not None:
            gc_coeffs = adapt_gc_coeff(
                gc_fractions,
                gc_coeffs,
                problem.target_gc_fraction,
                problem.weight_gc,
                problem.nucleotide_length,
                gc_coeff_adapt_mult,
            )
            assert gc_coeffs.shape == (n_chains,)
            gc_coeffs_times_beta = gc_coeffs * beta
        else:
            gc_coeffs_times_beta = None

        program = scale_model_by_beta(base_program, beta)

        # Split flat state (n_chains, L) into free blocks: even positions [0,2,4,...] and odd [1,3,5,...]
        prev_states_blocked = [prev_states[:, i::2] for i in range(len(free_blocks))]

        new_states = _run_sampling_jit(
            key,
            n_chains,
            program,
            schedule,
            all_nodes_block,
            None,
            init_states=prev_states_blocked,
            gc_coeffs_times_beta=gc_coeffs_times_beta,
            editable_params=prog_diff,
        )

        # new_states shape: (n_chains, 1, L), squeeze to (n_chains, L)
        squeezed = new_states[:, 0, :]
        if only_return_final:
            out_states = None
        else:
            out_states = squeezed

        # Compute metrics if requested
        if metrics is None:
            metric_outs = None
        else:
            metric_outs = jax.vmap(metrics)(squeezed)

        stats = {"gc_fractions": gc_fractions}
        if gc_coeffs is not None:
            stats["gc_coeffs"] = gc_coeffs

        return (squeezed, gc_coeffs), (out_states, stats, metric_outs)

    init_key, scan_key = jax.random.split(key)

    # Create random initial states for each chain
    n_codons_per_position = jnp.array([problem.num_codons_at(i) for i in range(problem.seq_length)])
    init_keys = jax.random.split(init_key, len(n_codons_per_position))
    init_states = jax.vmap(lambda k, n: jax.random.randint(k, (n_chains,), 0, n, dtype=jnp.uint8))(
        init_keys, n_codons_per_position
    )  # shape (L, n_chains)
    init_states = jnp.transpose(init_states, (1, 0))  # shape (n_chains, L)
    init_gc_coeff = None if gc_coeff_adapt_mult is None else jnp.zeros((n_chains,), dtype=float)
    init_carry = (init_states, init_gc_coeff)

    scan_keys = jax.random.split(scan_key, len(betas))

    (final_state, _), (trajectory, all_stats, metric_outs) = jax.lax.scan(
        run_single_beta_section, init_carry, (scan_keys, betas)
    )

    if metrics is not None:
        # Transpose metric_outs from (n_betas, n_chains, ...) to (n_chains, n_betas, ...)
        metric_outs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), metric_outs)

    if only_return_final:
        return final_state, all_stats, metric_outs

    # trajectory shape: (n_betas, n_chains, L) -> transpose to (n_chains, n_betas, L)
    trajectory = jnp.transpose(trajectory, (1, 0, 2))

    return trajectory, all_stats, metric_outs
