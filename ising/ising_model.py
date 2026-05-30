"""Ising model creation and sampling for codon optimization using Domain Wall Encoding."""

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from thrml.block_management import Block
from thrml.block_sampling import BlockGibbsSpec, SamplingSchedule, sample_states
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.pgm import SpinNode

from .dwc import (
    compute_ising_couplings,
    compute_spin_layout,
    potts_to_spin,
    spin_to_potts,
)
from problem import (
    CodonProblem,
    adapt_gc_coeff,
    compute_pairwise_penalties,
    compute_unary_biases,
    create_gc_counter,
)


# ---------------------------------------------------------------------------
# Static preparation (called once, outside JIT)
# ---------------------------------------------------------------------------

def prepare_ising_statics(problem: CodonProblem) -> dict:
    """One-time preparation of all static structures from a CodonProblem.

    Extracts Potts weights, spin layout, node objects, edge topology,
    4-color block decomposition, and precomputed index/mask arrays for
    vectorized weight computation inside JIT.

    Args:
        problem: The codon optimization problem.

    Returns:
        Dict containing all static data needed by build_ising_program.
    """
    # Potts weights: base (gc_coeff=0) and gc component
    base_potts_biases, Ks = compute_unary_biases(problem, linear_gc_coeff=0.0)
    gc_potts_biases, _ = compute_unary_biases(problem, linear_gc_coeff=1.0)
    gc_biases = gc_potts_biases - base_potts_biases

    potts_pairwise = compute_pairwise_penalties(problem) if problem.seq_length > 1 else None

    # Spin layout
    pos_of_spin, spin_pos_index, pos_matrix, spin_offsets = compute_spin_layout(Ks)
    n_total = len(pos_of_spin)
    L = len(Ks)

    if n_total == 0:
        raise ValueError("No spins to create (all positions have K=1)")

    # SpinNode objects (identity-based, created once and reused)
    all_spin_nodes = [SpinNode() for _ in range(n_total)]

    # Edge topology and inter-position weights (depend only on Ks and potts_pairwise,
    # not on P/beta/gc_coeff). We pass P=1.0 since we only need constraint edge topology;
    # constraint weights are recomputed dynamically in build_ising_program.
    constraint_edges, _, inter_edges, inter_weights_jax = compute_ising_couplings(potts_pairwise, Ks, 1.0)

    # Pre-build node lists for factor Blocks
    constraint_left = [all_spin_nodes[a] for a, b in constraint_edges] if constraint_edges else []
    constraint_right = [all_spin_nodes[b] for a, b in constraint_edges] if constraint_edges else []
    inter_left = [all_spin_nodes[a] for a, b in inter_edges] if inter_edges else []
    inter_right = [all_spin_nodes[b] for a, b in inter_edges] if inter_edges else []

    # --- Precompute vectorized arrays for fast weight computation inside JIT ---

    # Boundary masks for DWC constraint penalty terms:
    #   is_first[s]=1 if spin s is the first in its position's chain (j==0)
    #   is_last[s]=1  if spin s is the last in its position's chain (j==K_p-2)
    is_first = (spin_pos_index == 0).astype(np.float32)
    K_of_spin = np.array([Ks[p] for p in pos_of_spin])
    is_last = (spin_pos_index == K_of_spin - 2).astype(np.float32)

    # Static pairwise bias contributions to each spin's Ising bias.
    pairwise_bias_static = np.zeros(n_total, dtype=np.float32)
    idx = 0
    for p in range(L):
        K_p = Ks[p]
        for j in range(K_p - 1):
            pb = 0.0
            if potts_pairwise is not None and p > 0:
                Wp_left = potts_pairwise[p - 1]
                K_left = Ks[p - 1]
                pb += float(
                    Wp_left[0, j + 1] - Wp_left[0, j] + Wp_left[K_left - 1, j + 1] - Wp_left[K_left - 1, j]
                ) / 4
            if potts_pairwise is not None and p < L - 1:
                Wp_right = potts_pairwise[p]
                K_right = Ks[p + 1]
                pb += float(
                    Wp_right[j + 1, 0] - Wp_right[j, 0] + Wp_right[j + 1, K_right - 1] - Wp_right[j, K_right - 1]
                ) / 4
            pairwise_bias_static[idx] = pb
            idx += 1

    # Static inter-position coupling weights (already computed by compute_ising_couplings)
    inter_weights_static = jnp.array(inter_weights_jax)

    # 4-color block decomposition: (position_parity, spin_index_parity)
    color_keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    color_node_lists = {c: [] for c in color_keys}
    color_index_lists = {c: [] for c in color_keys}

    for s_idx in range(n_total):
        pp = int(pos_of_spin[s_idx]) % 2
        si = int(spin_pos_index[s_idx]) % 2
        color_node_lists[(pp, si)].append(all_spin_nodes[s_idx])
        color_index_lists[(pp, si)].append(s_idx)

    free_blocks = []
    block_indices = []
    for c in color_keys:
        if color_node_lists[c]:
            free_blocks.append(Block(color_node_lists[c]))
            block_indices.append(np.array(color_index_lists[c], dtype=np.int32))

    all_spins_block = Block(all_spin_nodes)

    return {
        "Ks": Ks,
        "base_potts_biases": jnp.array(base_potts_biases),
        "gc_biases": jnp.array(gc_biases),
        "all_spin_nodes": all_spin_nodes,
        "all_spins_block": all_spins_block,
        "free_blocks": free_blocks,
        "block_indices": block_indices,
        "constraint_left": constraint_left,
        "constraint_right": constraint_right,
        "inter_left": inter_left,
        "inter_right": inter_right,
        "n_total_spins": n_total,
        "n_constraint_edges": len(constraint_edges),
        # Index arrays for vectorized Ising bias computation
        "pos_of_spin": jnp.array(pos_of_spin),
        "spin_pos_index": jnp.array(spin_pos_index),
        "pos_matrix": jnp.array(pos_matrix),
        "spin_offsets": spin_offsets,
        "is_first": jnp.array(is_first),
        "is_last": jnp.array(is_last),
        "pairwise_bias_static": jnp.array(pairwise_bias_static),
        "inter_weights_static": jnp.array(inter_weights_static),
    }


# ---------------------------------------------------------------------------
# Program construction (JIT-compatible, vectorized)
# ---------------------------------------------------------------------------

def build_ising_program(statics: dict, beta, P, gc_coeff) -> FactorSamplingProgram:
    """Build a FactorSamplingProgram from dynamic parameters.

    JIT-compatible: uses vectorized array operations instead of Python loops.
    beta, P, and gc_coeff may be traced JAX scalars.

    Args:
        statics: Output of prepare_ising_statics.
        beta: Inverse temperature scalar.
        P: Constraint penalty strength scalar.
        gc_coeff: Linear GC coefficient scalar.

    Returns:
        A FactorSamplingProgram ready for sampling.
    """
    # Combine base potts biases with gc term
    potts_biases = statics["base_potts_biases"] + gc_coeff * statics["gc_biases"]

    # Vectorized Ising bias computation
    p_idx = statics["pos_of_spin"]
    j_idx = statics["spin_pos_index"]
    unary_diff = (potts_biases[p_idx, j_idx + 1] - potts_biases[p_idx, j_idx]) / 2

    ising_biases = (
        unary_diff
        + statics["is_first"] * (P / 4)
        - statics["is_last"] * (P / 4)
        + statics["pairwise_bias_static"]
    ) * beta

    # Constraint weights: all P*beta/4
    n_constraint = statics["n_constraint_edges"]
    constraint_weights = jnp.full(n_constraint, P * beta / 4) if n_constraint > 0 else jnp.zeros(0)

    # Inter-position weights: static, scaled by beta
    inter_weights = statics["inter_weights_static"] * beta

    # Build factors using pre-computed node structure
    factors = [SpinEBMFactor([Block(statics["all_spin_nodes"])], ising_biases)]

    if statics["constraint_left"]:
        factors.append(
            SpinEBMFactor(
                [Block(statics["constraint_left"]), Block(statics["constraint_right"])],
                constraint_weights,
            )
        )

    if statics["inter_left"]:
        factors.append(
            SpinEBMFactor(
                [Block(statics["inter_left"]), Block(statics["inter_right"])],
                inter_weights,
            )
        )

    spec = BlockGibbsSpec(statics["free_blocks"], [])
    samplers = [SpinGibbsConditional() for _ in spec.free_blocks]
    return FactorSamplingProgram(spec, samplers, factors, [])


# ---------------------------------------------------------------------------
# Backward-compatible wrapper
# ---------------------------------------------------------------------------

def create_ising_codon_model(
    problem: CodonProblem,
    P: float = 10.0,
    linear_gc_coeff: float = 0.0,
) -> tuple:
    """Create an Ising model for codon optimization using DWC encoding.

    Convenience wrapper around prepare_ising_statics + build_ising_program.

    Args:
        problem: The codon optimization problem.
        P: Constraint penalty strength.
        linear_gc_coeff: Coefficient for linear GC term.

    Returns:
        (program, free_blocks, all_spin_nodes, all_spins_block, metadata)
    """
    statics = prepare_ising_statics(problem)
    program = build_ising_program(statics, beta=1.0, P=P, gc_coeff=linear_gc_coeff)

    metadata = {
        "Ks": statics["Ks"],
        "pos_of_spin": np.array(statics["pos_of_spin"]),
        "spin_pos_index": np.array(statics["spin_pos_index"]),
        "pos_matrix": np.array(statics["pos_matrix"]),
        "spin_offsets": statics["spin_offsets"],
        "block_indices": statics["block_indices"],
        "n_total_spins": statics["n_total_spins"],
    }

    return program, statics["free_blocks"], statics["all_spin_nodes"], statics["all_spins_block"], metadata


# ---------------------------------------------------------------------------
# Annealing
# ---------------------------------------------------------------------------

@eqx.filter_jit
def run_ising_annealing(
    problem: CodonProblem,
    key: Array,
    n_chains: int,
    steps_per_beta: int,
    betas: Float[Array, "n_betas"],
    Ps: Float[Array, "n_betas"],
    gc_coeff_adapt_mult: Optional[Float[Array, ""]],
    only_return_final: bool,
    metrics: Optional[Callable[[Array], PyTree[Array]]] = None,
) -> tuple[Array, dict[str, Array], Optional[PyTree[Array]]]:
    """Run simulated annealing on the Ising codon model.

    Args:
        problem: The codon optimization problem.
        key: JAX random key.
        n_chains: Number of parallel chains.
        steps_per_beta: Gibbs sweeps per annealing step.
        betas: Array of inverse temperatures.
        Ps: Array of constraint penalties (one per beta step).
        gc_coeff_adapt_mult: Multiplier for GC adaptation, or None.
        only_return_final: If True, only return final states.
        metrics: Optional function for energy metrics (takes GC-sorted Potts indices).

    Returns:
        (states, stats, metric_outs)
    """
    statics = prepare_ising_statics(problem)

    pos_matrix_jax = statics["pos_matrix"]
    pos_of_spin_jax = statics["pos_of_spin"]
    spin_pos_index_jax = statics["spin_pos_index"]
    block_indices = statics["block_indices"]
    all_spins_block = statics["all_spins_block"]

    vec_gc_counter = jax.jit(jax.vmap(create_gc_counter(problem)))
    L = problem.seq_length

    def run_single_step(carry, inputs):
        key, beta, P = inputs
        prev_spin_state, gc_coeffs = carry

        # Convert spin -> Potts for GC computation
        potts_indices = spin_to_potts(prev_spin_state, pos_matrix_jax)
        gc_fractions = vec_gc_counter(potts_indices) / (3 * L)

        # Adapt GC coefficient per chain
        if gc_coeff_adapt_mult is not None:
            gc_coeffs = adapt_gc_coeff(
                gc_fractions,
                gc_coeffs,
                problem.target_gc_fraction,
                problem.weight_gc,
                problem.nucleotide_length,
                gc_coeff_adapt_mult,
            )

        schedule = SamplingSchedule(n_warmup=steps_per_beta, n_samples=1, steps_per_sample=1)

        # Split spin state into color blocks for init
        init_states = [prev_spin_state[:, idx] for idx in block_indices]

        # Per-chain gc_coeffs (0 when no adaptation)
        gc_coeffs_for_vmap = gc_coeffs if gc_coeffs is not None else jnp.zeros(n_chains)

        def sample_single_chain(chain_key, chain_init, gc_coeff):
            program = build_ising_program(statics, beta, P, gc_coeff)
            samples = sample_states(
                key=chain_key,
                program=program,
                schedule=schedule,
                init_state_free=chain_init,
                state_clamp=[],
                nodes_to_sample=[all_spins_block],
            )
            return samples[0]

        keys = jax.random.split(key, n_chains)
        new_spin_state = jax.vmap(sample_single_chain)(keys, init_states, gc_coeffs_for_vmap)

        # Shape: (n_chains, 1, n_total_spins) -> (n_chains, n_total_spins)
        squeezed = new_spin_state[:, 0, :]

        if only_return_final:
            out_states = None
        else:
            out_states = squeezed

        # Compute metrics from Potts indices
        if metrics is None:
            metric_outs = None
        else:
            new_potts = spin_to_potts(squeezed, pos_matrix_jax)
            metric_outs = jax.vmap(metrics)(new_potts)

        stats = {"gc_fractions": gc_fractions}
        if gc_coeffs is not None:
            stats["gc_coeffs"] = gc_coeffs

        return (squeezed, gc_coeffs), (out_states, stats, metric_outs)

    # Create random initial spin states
    init_key, scan_key = jax.random.split(key)

    n_codons_per_position = jnp.array([problem.num_codons_at(i) for i in range(L)])
    init_keys = jax.random.split(init_key, L)
    init_potts = jax.vmap(lambda k, n: jax.random.randint(k, (n_chains,), 0, n, dtype=jnp.uint8))(
        init_keys, n_codons_per_position
    )  # (L, n_chains)
    init_potts = jnp.transpose(init_potts, (1, 0))  # (n_chains, L)
    init_spin_state = potts_to_spin(init_potts, pos_of_spin_jax, spin_pos_index_jax)

    init_gc_coeff = None if gc_coeff_adapt_mult is None else jnp.zeros((n_chains,), dtype=float)
    init_carry = (init_spin_state, init_gc_coeff)

    scan_keys = jax.random.split(scan_key, len(betas))

    (final_state, _), (trajectory, all_stats, metric_outs) = jax.lax.scan(
        run_single_step, init_carry, (scan_keys, betas, Ps)
    )

    if metrics is not None:
        metric_outs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), metric_outs)

    if only_return_final:
        return final_state, all_stats, metric_outs

    trajectory = jnp.transpose(trajectory, (1, 0, 2))
    return trajectory, all_stats, metric_outs
