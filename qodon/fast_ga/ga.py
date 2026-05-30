"""Fast JAX-based genetic algorithm for codon optimization.

Implements the same algorithm as qodon's classical_ga.py but fully JIT-compiled:
1. Initialize population with frequency-weighted random codons
2. For each generation (via jax.lax.scan):
   - Score all individuals
   - Select top n_elite + random n_lucky survivors
   - Generate n_offspring via crossover + mutation
   - New population = elite + lucky + offspring (fixed size)
3. Return best individual

Note: The original GA grows the population each generation (appending offspring
to the full history). This version uses a fixed-size population with elitism,
which is algorithmically equivalent for optimization quality - the elite
individuals are always preserved, and the fixed size enables full JIT compilation
of the generation loop via jax.lax.scan.

Multi-chain support uses jax.vmap for true parallel execution.
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path

# Ensure we can import from this directory
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

try:
    from .scoring import ScoringConfig, build_lookup_tables, create_scoring_fn
except ImportError:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fast_ga_scoring", _this_dir / "scoring.py")
    _scoring_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_scoring_module)
    ScoringConfig = _scoring_module.ScoringConfig
    build_lookup_tables = _scoring_module.build_lookup_tables
    create_scoring_fn = _scoring_module.create_scoring_fn


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""

    # Population parameters
    pop_size: int = 62  # Fixed population size = n_elite + n_lucky + n_offspring
    n_elite: int = 10  # Number of elite individuals to keep (elitelist)
    n_lucky: int = 2  # Number of random survivors (randomlist)
    # n_offspring is derived: pop_size - n_elite - n_lucky = 50
    n_generations: int = 100  # Number of generations (numgens)

    # Mutation parameters
    mutation_rate: float = 0.05  # Probability of mutation per position

    # Multi-chain support
    n_chains: int = 1  # Number of independent GA chains

    # Random seed
    seed: int = 42

    @property
    def n_offspring(self) -> int:
        return self.pop_size - self.n_elite - self.n_lucky


class FastGA:
    """Fast JAX-based genetic algorithm for codon optimization.

    The entire GA loop is JIT-compiled via jax.lax.scan, and multi-chain
    runs use jax.vmap for true parallelism.
    """

    def __init__(
        self,
        amino_acid_seq: str,
        codon_table: dict[str, list[str]],
        codon_scores: dict[str, float],
        code_map: dict[str, dict],
        ga_config: GAConfig | None = None,
        scoring_config: ScoringConfig | None = None,
    ):
        self.amino_acid_seq = amino_acid_seq
        self.codon_table = codon_table
        self.codon_scores = codon_scores
        self.code_map = code_map
        self.ga_config = ga_config or GAConfig()
        self.scoring_config = scoring_config or ScoringConfig()

        self.L = len(amino_acid_seq)

        # Build lookup tables
        gc_lookup, rarity_lookup, repeat_lookup, n_codons, max_codons = build_lookup_tables(
            amino_acid_seq, codon_table, codon_scores
        )
        self.gc_lookup = jnp.array(gc_lookup)
        self.rarity_lookup = jnp.array(rarity_lookup)
        self.repeat_lookup = jnp.array(repeat_lookup)
        self.n_codons = jnp.array(n_codons)
        self.max_codons = max_codons

        # Build cumulative probability lookup for frequency-weighted sampling
        self.cum_probs = self._build_cum_probs()

        # Precompute constants
        self.N = 3 * self.L  # nucleotide length

        # Create the detailed scoring function (for final reporting)
        self.score_fn = create_scoring_fn(amino_acid_seq, codon_table, codon_scores, self.scoring_config)

        # Build the JIT-compiled GA runner
        self._jit_run_chain = self._build_jit_runner()

    def _build_cum_probs(self) -> jnp.ndarray:
        """Build cumulative probability lookup for sampling."""
        cum_probs = np.zeros((self.L, self.max_codons), dtype=np.float32)
        for i, aa in enumerate(self.amino_acid_seq):
            probs = self.code_map[aa]["probs"]
            max_prob = probs[-1]
            normalized = [p / max_prob for p in probs]
            for j, p in enumerate(normalized):
                cum_probs[i, j] = p
            for j in range(len(probs), self.max_codons):
                cum_probs[i, j] = 1.0
        return jnp.array(cum_probs)

    def _build_jit_runner(self):
        """Build a fully JIT-compiled single-chain GA runner."""
        # Capture all config as Python values (static for JIT)
        cfg = self.ga_config
        scfg = self.scoring_config
        pop_size = cfg.pop_size
        n_elite = cfg.n_elite
        n_lucky = cfg.n_lucky
        n_offspring = cfg.n_offspring
        n_generations = cfg.n_generations
        mutation_rate = cfg.mutation_rate
        L = self.L
        N = self.N

        # Capture lookup tables
        gc_lookup = self.gc_lookup
        rarity_lookup = self.rarity_lookup
        repeat_lookup = self.repeat_lookup
        n_codons = self.n_codons
        cum_probs = self.cum_probs

        # Scoring weights
        gc_weight = scfg.gc_weight
        repeat_weight = scfg.repeat_weight
        rarity_weight = scfg.rarity_weight
        target_gc = scfg.target_gc

        n_parents = n_elite + n_lucky
        pos_indices = jnp.arange(L)
        pair_indices = jnp.arange(L - 1)

        def _score_total(population):
            """Compute total score for each individual. population: (pop_size, L)."""
            # GC
            gc_counts = gc_lookup[pos_indices, population]  # (pop_size, L)
            gc_frac = gc_counts.sum(axis=-1) / N
            gc_term = gc_weight * (gc_frac - target_gc) ** 2
            # Rarity
            rarity = rarity_lookup[pos_indices, population]
            rarity_term = rarity_weight * rarity.sum(axis=-1)
            # Repeat
            repeat = repeat_lookup[pair_indices, population[:, :-1], population[:, 1:]]
            repeat_term = repeat_weight * repeat.sum(axis=-1)
            return gc_term + rarity_term + repeat_term

        def _sample_codons(key, shape):
            """Sample codon indices weighted by frequency. shape = (batch, L)."""
            rand_vals = jax.random.uniform(key, shape)
            # cum_probs: (L, max_codons), rand_vals: (batch, L)
            # For each (i, j), count how many cum_probs[j, :] < rand_vals[i, j]
            indices = jnp.sum(
                cum_probs[None, :, :] < rand_vals[:, :, None], axis=-1
            ).astype(jnp.int32)
            indices = jnp.minimum(indices, n_codons[None, :] - 1)
            return indices

        def _generation_step(carry, _):
            """One generation of the GA. Fully JAX-traceable."""
            population, key = carry

            # Score population
            scores = _score_total(population)  # (pop_size,)

            # Track stats for this generation
            best_score = jnp.min(scores)
            mean_score = jnp.mean(scores)

            # Sort by score (ascending = best first)
            sorted_idx = jnp.argsort(scores)

            # Elite: top n_elite
            elite_idx = sorted_idx[:n_elite]

            # Lucky: random from the rest
            key, k_lucky = jax.random.split(key)
            rest_idx = sorted_idx[n_elite:]
            lucky_perm = jax.random.permutation(k_lucky, rest_idx.shape[0])
            lucky_idx = rest_idx[lucky_perm[:n_lucky]]

            # Parents = elite + lucky
            parent_idx = jnp.concatenate([elite_idx, lucky_idx])
            parents = population[parent_idx]  # (n_parents, L)

            # Generate offspring via crossover + mutation
            key, k_p1, k_p2, k_cross, k_mut_mask, k_mut_sample = jax.random.split(key, 6)

            # Random parent pairs
            idx1 = jax.random.randint(k_p1, (n_offspring,), 0, n_parents)
            idx2 = jax.random.randint(k_p2, (n_offspring,), 0, n_parents)
            p1 = parents[idx1]
            p2 = parents[idx2]

            # Uniform crossover
            cross_mask = jax.random.uniform(k_cross, (n_offspring, L)) < 0.5
            children = jnp.where(cross_mask, p1, p2)

            # Mutation
            mut_mask = jax.random.uniform(k_mut_mask, (n_offspring, L)) < mutation_rate
            new_codons = _sample_codons(k_mut_sample, (n_offspring, L))
            children = jnp.where(mut_mask, new_codons, children)

            # New population = parents + offspring
            new_population = jnp.concatenate([parents, children], axis=0)

            return (new_population, key), (best_score, mean_score)

        @jax.jit
        def run_chain(key):
            """Run a single GA chain. Returns (final_pop, best_history, mean_history)."""
            # Initialize population
            key, k_init = jax.random.split(key)
            population = _sample_codons(k_init, (pop_size, L))

            # Run all generations via scan
            (final_pop, _), (best_history, mean_history) = jax.lax.scan(
                _generation_step, (population, key), None, length=n_generations
            )

            return final_pop, best_history, mean_history

        return run_chain

    def run(self, key: jax.Array | None = None) -> dict:
        """Run the genetic algorithm.

        Args:
            key: JAX random key. If None, uses seed from config.

        Returns:
            Dict with results including best sequence, scores, and history.
        """
        if key is None:
            key = jax.random.key(self.ga_config.seed)

        n_chains = self.ga_config.n_chains

        if n_chains == 1:
            return self._run_single(key)
        else:
            return self._run_multi(key)

    def _run_single(self, key: jax.Array) -> dict:
        """Run a single GA chain."""
        final_pop, best_history, mean_history = self._jit_run_chain(key)

        # Detailed scoring of final population
        final_scores = self.score_fn(final_pop)
        best_idx = jnp.argmin(final_scores["total"])
        best_indices = final_pop[best_idx]
        best_score_details = {k: float(v[best_idx]) for k, v in final_scores.items()}

        best_nuc_seq = self._indices_to_nucleotides(best_indices)

        return {
            "best_indices": np.array(best_indices),
            "best_nucleotides": best_nuc_seq,
            "best_score": best_score_details,
            "history": {
                "best_scores": np.array(best_history),
                "mean_scores": np.array(mean_history),
            },
            "final_population": np.array(final_pop),
            "final_scores": {k: np.array(v) for k, v in final_scores.items()},
        }

    def _run_multi(self, key: jax.Array) -> dict:
        """Run multiple independent GA chains in parallel via vmap."""
        n_chains = self.ga_config.n_chains
        keys = jax.random.split(key, n_chains)

        # vmap over keys - runs all chains in parallel
        vmap_run = jax.vmap(self._jit_run_chain)
        all_final_pops, all_best_history, all_mean_history = vmap_run(keys)
        # all_final_pops: (n_chains, pop_size, L)
        # all_best_history: (n_chains, n_generations)
        # all_mean_history: (n_chains, n_generations)

        # Detailed scoring of all final populations
        final_scores_all = jax.vmap(self.score_fn)(all_final_pops)
        # final_scores_all["total"]: (n_chains, pop_size)

        final_totals = final_scores_all["total"]

        # Find global best
        flat_idx = jnp.argmin(final_totals)
        best_chain = flat_idx // all_final_pops.shape[1]
        best_in_chain = flat_idx % all_final_pops.shape[1]

        best_indices = all_final_pops[best_chain, best_in_chain]
        best_score_details = {k: float(v[best_chain, best_in_chain]) for k, v in final_scores_all.items()}

        # Best per chain
        best_per_chain_idx = jnp.argmin(final_totals, axis=1)
        best_per_chain_scores = []
        for i in range(n_chains):
            chain_best = {k: float(v[i, best_per_chain_idx[i]]) for k, v in final_scores_all.items()}
            best_per_chain_scores.append(chain_best)

        best_nuc_seq = self._indices_to_nucleotides(best_indices)

        return {
            "best_indices": np.array(best_indices),
            "best_nucleotides": best_nuc_seq,
            "best_score": best_score_details,
            "best_chain": int(best_chain),
            "history": {
                "best_scores": np.array(all_best_history),  # (n_chains, n_generations)
                "mean_scores": np.array(all_mean_history),
            },
            "best_per_chain": best_per_chain_scores,
            "final_populations": np.array(all_final_pops),
            "final_scores": {k: np.array(v) for k, v in final_scores_all.items()},
        }

    def _indices_to_nucleotides(self, indices: jax.Array) -> str:
        """Convert codon indices to nucleotide sequence."""
        indices = np.array(indices)
        codons = []
        for i, aa in enumerate(self.amino_acid_seq):
            codon_list = self.codon_table[aa]
            codons.append(codon_list[indices[i]])
        return "".join(codons)


def run_fast_ga(
    amino_acid_seq: str,
    ga_config: GAConfig | None = None,
    scoring_config: ScoringConfig | None = None,
    seed: int | None = None,
) -> dict:
    """Convenience function to run the fast GA."""
    qodon_path = Path(__file__).parent.parent
    if str(qodon_path) not in sys.path:
        sys.path.insert(0, str(qodon_path))
    from constants import codon_table, codon_scores, code_map

    if ga_config is None:
        ga_config = GAConfig()
    if seed is not None:
        ga_config.seed = seed

    ga = FastGA(
        amino_acid_seq=amino_acid_seq,
        codon_table=codon_table,
        codon_scores=codon_scores,
        code_map=code_map,
        ga_config=ga_config,
        scoring_config=scoring_config,
    )

    return ga.run()
