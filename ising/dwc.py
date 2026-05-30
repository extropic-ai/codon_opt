"""Domain Wall Encoding (DWC) for codon optimization.

Provides Potts→Ising weight compilation and state conversion between
spin (Ising) and categorical (Potts) representations.

Codon ordering, Potts weight computation, GC counting, and energy metrics
are in problem.py (shared by both Potts and Ising models).
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


# ---------------------------------------------------------------------------
# 1. Spin index arrays
# ---------------------------------------------------------------------------

def compute_spin_layout(Ks: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute index arrays for the flat spin representation.

    Args:
        Ks: Number of Potts states at each position.

    Returns:
        (pos_of_spin, spin_pos_index, pos_matrix, spin_offsets) where:
        - pos_of_spin: [n_total] position index for each spin
        - spin_pos_index: [n_total] 0-based index within position's DWC chain
        - pos_matrix: [n_total, L] one-hot encoding of pos_of_spin
        - spin_offsets: [L] cumulative spin count before each position
    """
    L = len(Ks)
    pos_of_spin = []
    spin_pos_index = []
    spin_offsets = np.zeros(L, dtype=np.int32)

    offset = 0
    for p in range(L):
        spin_offsets[p] = offset
        n_spins = max(Ks[p] - 1, 0)
        for j in range(n_spins):
            pos_of_spin.append(p)
            spin_pos_index.append(j)
        offset += n_spins

    pos_of_spin = np.array(pos_of_spin, dtype=np.int32)
    spin_pos_index = np.array(spin_pos_index, dtype=np.int32)
    pos_matrix = np.eye(L, dtype=np.int32)[pos_of_spin] if len(pos_of_spin) > 0 else np.zeros((0, L), dtype=np.int32)

    return pos_of_spin, spin_pos_index, pos_matrix, spin_offsets


# ---------------------------------------------------------------------------
# 2. Ising weight compilation
# ---------------------------------------------------------------------------

def compute_ising_biases(
    potts_biases,
    potts_pairwise,
    Ks: list[int],
    P,
):
    """Compile Potts weights to Ising bias weights.

    JIT-compatible: all Python control flow depends on Ks (static).
    potts_biases, potts_pairwise, and P may be traced JAX arrays.

    Args:
        potts_biases: [L, K_max] THRML unary weights.
        potts_pairwise: [L-1, K_max, K_max] THRML pairwise weights, or None.
        Ks: Number of Potts states per position.
        P: Constraint penalty strength.

    Returns:
        [n_total_spins] array of Ising bias weights.
    """
    L = len(Ks)
    bias_list = []

    for p in range(L):
        K_p = Ks[p]
        W = potts_biases[p]
        for j in range(K_p - 1):
            b = (W[j + 1] - W[j]) / 2
            if j == 0:
                b = b + P / 4
            if j == K_p - 2:
                b = b - P / 4

            # Pairwise bias from left neighbor (edge p-1 -> p)
            if potts_pairwise is not None and p > 0:
                Wp_left = potts_pairwise[p - 1]
                K_left = Ks[p - 1]
                pb = (Wp_left[0, j + 1] - Wp_left[0, j] + Wp_left[K_left - 1, j + 1] - Wp_left[K_left - 1, j]) / 4
                b = b + pb

            # Pairwise bias from right neighbor (edge p -> p+1)
            if potts_pairwise is not None and p < L - 1:
                Wp_right = potts_pairwise[p]
                K_right = Ks[p + 1]
                pb = (Wp_right[j + 1, 0] - Wp_right[j, 0] + Wp_right[j + 1, K_right - 1] - Wp_right[j, K_right - 1]) / 4
                b = b + pb

            bias_list.append(b)

    if bias_list:
        return jnp.stack(bias_list)
    return jnp.zeros(0)


def compute_ising_couplings(
    potts_pairwise,
    Ks: list[int],
    P,
):
    """Compute Ising coupling edges and weights.

    JIT-compatible: edges are static (depend on Ks), weights may be traced.

    Args:
        potts_pairwise: [L-1, K_max, K_max] THRML pairwise weights, or None.
        Ks: Number of Potts states per position.
        P: Constraint penalty strength.

    Returns:
        (constraint_edges, constraint_weights, inter_edges, inter_weights) where
        edges are lists of (flat_spin_i, flat_spin_j) tuples and weights are JAX arrays.
    """
    L = len(Ks)

    # Constraint edges (within positions) — static
    constraint_edges = []
    for p in range(L):
        K = Ks[p]
        offset = sum(max(Ks[pp] - 1, 0) for pp in range(p))
        for j in range(K - 2):
            constraint_edges.append((offset + j, offset + j + 1))

    n_constraint = len(constraint_edges)
    constraint_weights = jnp.full(n_constraint, P / 4) if n_constraint > 0 else jnp.zeros(0)

    # Inter-position edges — edges are static, weights may be traced
    inter_edges = []
    inter_weights_list = []
    if potts_pairwise is not None:
        for e in range(L - 1):
            p, q = e, e + 1
            K_p, K_q = Ks[p], Ks[q]
            if K_p <= 1 or K_q <= 1:
                continue
            Wp = potts_pairwise[e]
            offset_p = sum(max(Ks[pp] - 1, 0) for pp in range(p))
            offset_q = sum(max(Ks[pp] - 1, 0) for pp in range(q))
            for i in range(K_p - 1):
                for j in range(K_q - 1):
                    w = (Wp[i + 1, j + 1] - Wp[i, j + 1] - Wp[i + 1, j] + Wp[i, j]) / 4
                    inter_edges.append((offset_p + i, offset_q + j))
                    inter_weights_list.append(w)

    inter_weights = jnp.stack(inter_weights_list) if inter_weights_list else jnp.zeros(0)

    return constraint_edges, constraint_weights, inter_edges, inter_weights


# ---------------------------------------------------------------------------
# 3. State conversion
# ---------------------------------------------------------------------------

def potts_to_spin(potts_indices: Array, pos_of_spin: Array, spin_pos_index: Array) -> Array:
    """Convert Potts indices to spin states (thermometer encoding).

    Args:
        potts_indices: [..., L] uint8 Potts state indices.
        pos_of_spin: [n_total_spins] int32 position for each spin.
        spin_pos_index: [n_total_spins] int32 index within position.

    Returns:
        [..., n_total_spins] bool spin states (True = +1).
    """
    k = potts_indices[..., pos_of_spin]
    return spin_pos_index < k


def spin_to_potts(spin_bool: Array, pos_matrix: Array) -> Array:
    """Convert spin states to Potts indices via matrix multiply.

    Args:
        spin_bool: [..., n_total_spins] bool spin states.
        pos_matrix: [n_total_spins, L] int32 one-hot of pos_of_spin.

    Returns:
        [..., L] uint8 Potts indices (in GC-sorted ordering).
    """
    return (spin_bool.astype(jnp.int32) @ pos_matrix).astype(jnp.uint8)
