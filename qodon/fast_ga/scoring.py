"""JAX-compatible scoring functions for codon optimization.

Implements the same scoring as qodon's SeqScorer:
- GC score: (GC_fraction - 0.5)^2
- Repeat score: sum over adjacent pairs of (longest_run^2 - 1)
- Rarity score: sum of |log(freq/max_freq)| for each codon
"""

from dataclasses import dataclass
from functools import partial
from itertools import groupby
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ScoringConfig:
    """Configuration for scoring function weights."""

    gc_weight: float = 2e4
    repeat_weight: float = 0.2
    rarity_weight: float = 0.1
    target_gc: float = 0.5


def _compute_repeat_penalty(codon1: str, codon2: str) -> int:
    """Compute repeat penalty for two adjacent codons.

    Same as qodon's _repeat_score logic: finds longest run of identical
    nucleotides in the 6-mer formed by concatenating the two codons.

    Returns:
        m^2 - 1 where m is the longest run length.
    """
    s = codon1 + codon2
    runs = [(letter, len(list(group))) for letter, group in groupby(s)]
    max_run = max(run[1] for run in runs)
    return max_run**2 - 1


def _gc_count(codon: str) -> int:
    """Count G and C nucleotides in a codon."""
    return sum(1 for n in codon if n in "GC")


def build_lookup_tables(
    amino_acid_seq: str,
    codon_table: dict[str, list[str]],
    codon_scores: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build precomputed lookup tables for JAX scoring.

    Args:
        amino_acid_seq: The amino acid sequence.
        codon_table: Dict mapping amino acid -> list of codons.
        codon_scores: Dict mapping codon -> rarity score (|log(freq/max)|).

    Returns:
        Tuple of:
        - gc_lookup: (L, max_codons) GC counts per codon
        - rarity_lookup: (L, max_codons) rarity scores per codon
        - repeat_lookup: (L-1, max_codons, max_codons) repeat penalties
        - n_codons: (L,) number of valid codons at each position
        - max_codons: maximum number of codons for any amino acid
    """
    L = len(amino_acid_seq)
    max_codons = max(len(codon_table[aa]) for aa in amino_acid_seq)

    # GC counts and rarity scores per (position, codon_idx)
    gc_lookup = np.zeros((L, max_codons), dtype=np.float32)
    rarity_lookup = np.zeros((L, max_codons), dtype=np.float32)
    n_codons = np.zeros(L, dtype=np.int32)

    for i, aa in enumerate(amino_acid_seq):
        codons = codon_table[aa]
        n_codons[i] = len(codons)
        for j, codon in enumerate(codons):
            gc_lookup[i, j] = _gc_count(codon)
            rarity_lookup[i, j] = codon_scores.get(codon, 0.0)

    # Repeat penalties for adjacent codon pairs
    repeat_lookup = np.zeros((L - 1, max_codons, max_codons), dtype=np.float32)
    for i in range(L - 1):
        aa1 = amino_acid_seq[i]
        aa2 = amino_acid_seq[i + 1]
        codons1 = codon_table[aa1]
        codons2 = codon_table[aa2]
        for j, c1 in enumerate(codons1):
            for k, c2 in enumerate(codons2):
                repeat_lookup[i, j, k] = _compute_repeat_penalty(c1, c2)

    return gc_lookup, rarity_lookup, repeat_lookup, n_codons, max_codons


def create_scoring_fn(
    amino_acid_seq: str,
    codon_table: dict[str, list[str]],
    codon_scores: dict[str, float],
    config: ScoringConfig | None = None,
) -> Callable:
    """Create a JIT-compiled scoring function for the given sequence.

    Args:
        amino_acid_seq: The amino acid sequence.
        codon_table: Dict mapping amino acid -> list of codons.
        codon_scores: Dict mapping codon -> rarity score.
        config: Scoring weights configuration.

    Returns:
        A function that takes codon indices of shape (..., L) and returns
        a dict with 'total', 'gc', 'repeat', 'rarity' scores of shape (...).
    """
    if config is None:
        config = ScoringConfig()

    gc_lookup, rarity_lookup, repeat_lookup, n_codons, max_codons = build_lookup_tables(
        amino_acid_seq, codon_table, codon_scores
    )

    # Convert to JAX arrays
    gc_lookup_jax = jnp.array(gc_lookup)
    rarity_lookup_jax = jnp.array(rarity_lookup)
    repeat_lookup_jax = jnp.array(repeat_lookup)

    L = len(amino_acid_seq)
    N = 3 * L  # nucleotide length

    gc_weight = config.gc_weight
    repeat_weight = config.repeat_weight
    rarity_weight = config.rarity_weight
    target_gc = config.target_gc

    @jax.jit
    def score_fn(indices):
        """Score codon indices.

        Args:
            indices: Array of shape (..., L) with codon indices.

        Returns:
            Dict with scores of shape (...).
        """
        # Handle batched inputs
        batch_shape = indices.shape[:-1]
        L_check = indices.shape[-1]

        # Flatten batch dimensions for computation
        flat_indices = indices.reshape(-1, L_check)
        n_batch = flat_indices.shape[0]

        # GC term: sum GC counts, compute (fraction - target)^2
        pos_indices = jnp.arange(L_check)
        gc_counts = gc_lookup_jax[pos_indices, flat_indices]  # (n_batch, L)
        total_gc = gc_counts.sum(axis=-1)  # (n_batch,)
        gc_frac = total_gc / N
        gc_score_raw = (gc_frac - target_gc) ** 2
        gc_term = gc_weight * gc_score_raw

        # Rarity term: sum rarity scores
        rarity_scores = rarity_lookup_jax[pos_indices, flat_indices]  # (n_batch, L)
        rarity_score_raw = rarity_scores.sum(axis=-1)  # (n_batch,)
        rarity_term = rarity_weight * rarity_score_raw

        # Repeat term: sum repeat penalties for adjacent pairs
        pair_indices = jnp.arange(L_check - 1)
        repeat_penalties = repeat_lookup_jax[
            pair_indices, flat_indices[:, :-1], flat_indices[:, 1:]
        ]  # (n_batch, L-1)
        repeat_score_raw = repeat_penalties.sum(axis=-1)  # (n_batch,)
        repeat_term = repeat_weight * repeat_score_raw

        total = gc_term + rarity_term + repeat_term

        # Reshape back to batch shape
        return {
            "total": total.reshape(batch_shape),
            "gc": gc_term.reshape(batch_shape),
            "gc_raw": gc_score_raw.reshape(batch_shape),
            "gc_frac": gc_frac.reshape(batch_shape),
            "repeat": repeat_term.reshape(batch_shape),
            "repeat_raw": repeat_score_raw.reshape(batch_shape),
            "rarity": rarity_term.reshape(batch_shape),
            "rarity_raw": rarity_score_raw.reshape(batch_shape),
        }

    return score_fn


def verify_scoring_matches_qodon(
    nucleotide_seq: str,
    codon_table: dict[str, list[str]],
    codon_scores: dict[str, float],
    amino_acid_seq: str,
    config: ScoringConfig | None = None,
) -> dict:
    """Verify JAX scoring matches qodon's SeqScorer.

    Args:
        nucleotide_seq: The nucleotide sequence to score.
        codon_table: Dict mapping amino acid -> list of codons.
        codon_scores: Dict mapping codon -> rarity score.
        amino_acid_seq: The amino acid sequence.
        config: Scoring weights.

    Returns:
        Dict with both scores for comparison.
    """
    import sys
    from pathlib import Path

    # Import qodon's SeqScorer
    qodon_path = Path(__file__).parent.parent
    if str(qodon_path) not in sys.path:
        sys.path.insert(0, str(qodon_path))
    from scoring import SeqScorer

    if config is None:
        config = ScoringConfig()

    # Get qodon score
    qodon_scorer = SeqScorer(nucleotide_seq)
    qodon_total = qodon_scorer.score
    qodon_gc = qodon_scorer.gc_score
    qodon_repeat = qodon_scorer.rep_score
    qodon_rarity = qodon_scorer.rarity_score

    # Convert nucleotide seq to codon indices
    L = len(amino_acid_seq)
    indices = []
    for i in range(L):
        codon = nucleotide_seq[3 * i : 3 * i + 3]
        aa = amino_acid_seq[i]
        codons = codon_table[aa]
        idx = codons.index(codon)
        indices.append(idx)
    indices = jnp.array(indices)

    # Get JAX score
    score_fn = create_scoring_fn(amino_acid_seq, codon_table, codon_scores, config)
    jax_scores = score_fn(indices)

    return {
        "qodon": {
            "total": qodon_total,
            "gc_raw": qodon_gc,
            "repeat_raw": qodon_repeat,
            "rarity_raw": qodon_rarity,
        },
        "jax": {
            "total": float(jax_scores["total"]),
            "gc_raw": float(jax_scores["gc_raw"]),
            "repeat_raw": float(jax_scores["repeat_raw"]),
            "rarity_raw": float(jax_scores["rarity_raw"]),
        },
    }
