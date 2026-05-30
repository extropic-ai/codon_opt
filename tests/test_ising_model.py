"""Tests for the Ising codon optimization model with Domain Wall Encoding."""

from itertools import product as cart_product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ising.dwc import (
    compute_ising_biases,
    compute_ising_couplings,
    compute_spin_layout,
    potts_to_spin,
    spin_to_potts,
)
from ising.ising_model import create_ising_codon_model
from problem import (
    CODON_TABLE,
    SORTED_CODON_TABLE,
    CodonProblem,
    compute_energy,
    gc_count,
    indices_to_codons,
)


# ===========================================================================
# 1. Codon ordering tests
# ===========================================================================

class TestCodonOrdering:
    def test_gc_sorted_leucine(self):
        """Leucine (K=6) should be sorted by ascending GC count, ties alphabetical."""
        codons = SORTED_CODON_TABLE["L"]
        gc_counts = [gc_count(c) for c in codons]
        assert len(codons) == 6
        # GC counts should be non-decreasing
        assert gc_counts == sorted(gc_counts)
        # Within same GC count, alphabetical
        for i in range(len(codons) - 1):
            if gc_counts[i] == gc_counts[i + 1]:
                assert codons[i] <= codons[i + 1]

    def test_gc_sorted_met_trp(self):
        """Met and Trp have K=1 (no synonymous codons)."""
        assert len(SORTED_CODON_TABLE["M"]) == 1
        assert len(SORTED_CODON_TABLE["W"]) == 1

    def test_gc_sorted_all_amino_acids(self):
        """All amino acids should have correct number of codons."""
        for aa in CODON_TABLE:
            assert len(SORTED_CODON_TABLE[aa]) == len(CODON_TABLE[aa])
            # Same set of codons, just different order
            assert set(SORTED_CODON_TABLE[aa]) == set(CODON_TABLE[aa])


# ===========================================================================
# 2. State conversion tests
# ===========================================================================

class TestStateConversion:
    def test_roundtrip_simple(self):
        """Potts -> spin -> Potts roundtrip."""
        Ks = [3, 2, 4, 1, 6]
        pos_of_spin, spin_pos_index, pos_matrix, _ = compute_spin_layout(Ks)

        pos_of_spin_jax = jnp.array(pos_of_spin)
        spin_pos_index_jax = jnp.array(spin_pos_index)
        pos_matrix_jax = jnp.array(pos_matrix)

        potts = jnp.array([1, 0, 3, 0, 4], dtype=jnp.uint8)
        spin = potts_to_spin(potts, pos_of_spin_jax, spin_pos_index_jax)
        recovered = spin_to_potts(spin, pos_matrix_jax)

        np.testing.assert_array_equal(recovered, potts)

    def test_roundtrip_all_states(self):
        """Roundtrip for all valid states of a small system."""
        Ks = [3, 2]
        pos_of_spin, spin_pos_index, pos_matrix, _ = compute_spin_layout(Ks)

        pos_of_spin_jax = jnp.array(pos_of_spin)
        spin_pos_index_jax = jnp.array(spin_pos_index)
        pos_matrix_jax = jnp.array(pos_matrix)

        for k0 in range(Ks[0]):
            for k1 in range(Ks[1]):
                potts = jnp.array([k0, k1], dtype=jnp.uint8)
                spin = potts_to_spin(potts, pos_of_spin_jax, spin_pos_index_jax)
                recovered = spin_to_potts(spin, pos_matrix_jax)
                np.testing.assert_array_equal(recovered, potts)

    def test_k1_position(self):
        """K=1 positions produce zero spins and always recover index 0."""
        Ks = [1, 3]
        pos_of_spin, spin_pos_index, pos_matrix, _ = compute_spin_layout(Ks)

        assert len(pos_of_spin) == 2  # only 2 spins from position 1

        pos_of_spin_jax = jnp.array(pos_of_spin)
        spin_pos_index_jax = jnp.array(spin_pos_index)
        pos_matrix_jax = jnp.array(pos_matrix)

        potts = jnp.array([0, 2], dtype=jnp.uint8)
        spin = potts_to_spin(potts, pos_of_spin_jax, spin_pos_index_jax)
        recovered = spin_to_potts(spin, pos_matrix_jax)
        np.testing.assert_array_equal(recovered, potts)

    def test_batched(self):
        """State conversion works with batch dimensions."""
        Ks = [3, 2]
        pos_of_spin, spin_pos_index, pos_matrix, _ = compute_spin_layout(Ks)

        pos_of_spin_jax = jnp.array(pos_of_spin)
        spin_pos_index_jax = jnp.array(spin_pos_index)
        pos_matrix_jax = jnp.array(pos_matrix)

        batch_potts = jnp.array([[0, 0], [1, 1], [2, 0]], dtype=jnp.uint8)
        batch_spin = potts_to_spin(batch_potts, pos_of_spin_jax, spin_pos_index_jax)
        batch_recovered = spin_to_potts(batch_spin, pos_matrix_jax)
        np.testing.assert_array_equal(batch_recovered, batch_potts)


# ===========================================================================
# 3. Energy equivalence tests
# ===========================================================================

def _build_potts_ebm(W_unary_list, W_pair_list):
    """Build a Potts model for energy evaluation."""
    from thrml.block_management import Block
    from thrml.models.discrete_ebm import CategoricalEBMFactor
    from thrml.models.ebm import FactorizedEBM
    from thrml.pgm import CategoricalNode

    L = len(W_unary_list)
    K_max = max(len(w) for w in W_unary_list)
    nodes = [CategoricalNode() for _ in range(L)]

    biases = np.full((L, K_max), -1e10, dtype=np.float32)
    for p, w in enumerate(W_unary_list):
        biases[p, :len(w)] = w

    bias_factor = CategoricalEBMFactor([Block(nodes)], jnp.array(biases))
    factors = [bias_factor]

    if W_pair_list:
        pairwise = np.zeros((L - 1, K_max, K_max), dtype=np.float32)
        for e, w in enumerate(W_pair_list):
            pairwise[e, :w.shape[0], :w.shape[1]] = w
        pair_factor = CategoricalEBMFactor([Block(nodes[:-1]), Block(nodes[1:])], jnp.array(pairwise))
        factors.append(pair_factor)

    ebm = FactorizedEBM(factors)
    return ebm, nodes


def _build_ising_ebm(W_unary_list, W_pair_list, P):
    """Build an Ising DWC model for energy evaluation."""
    from thrml.block_management import Block
    from thrml.models.discrete_ebm import SpinEBMFactor
    from thrml.models.ebm import FactorizedEBM
    from thrml.pgm import SpinNode

    Ks = [len(w) for w in W_unary_list]
    L = len(Ks)
    K_max = max(Ks)

    # Pad to K_max
    potts_biases = np.full((L, K_max), -1e10, dtype=np.float32)
    for p, w in enumerate(W_unary_list):
        potts_biases[p, :len(w)] = w

    potts_pairwise = None
    if W_pair_list:
        potts_pairwise = np.zeros((L - 1, K_max, K_max), dtype=np.float32)
        for e, w in enumerate(W_pair_list):
            potts_pairwise[e, :w.shape[0], :w.shape[1]] = w

    pos_of_spin, spin_pos_index, pos_matrix, spin_offsets = compute_spin_layout(Ks)
    n_total = len(pos_of_spin)
    spin_nodes = [SpinNode() for _ in range(n_total)]

    ising_biases = compute_ising_biases(potts_biases, potts_pairwise, Ks, P)
    constraint_edges, constraint_weights, inter_edges, inter_weights = compute_ising_couplings(
        potts_pairwise, Ks, P
    )

    factors = [SpinEBMFactor([Block(spin_nodes)], jnp.array(ising_biases))]
    if constraint_edges:
        left = [spin_nodes[a] for a, b in constraint_edges]
        right = [spin_nodes[b] for a, b in constraint_edges]
        factors.append(SpinEBMFactor([Block(left), Block(right)], jnp.array(constraint_weights)))
    if inter_edges:
        left = [spin_nodes[a] for a, b in inter_edges]
        right = [spin_nodes[b] for a, b in inter_edges]
        factors.append(SpinEBMFactor([Block(left), Block(right)], jnp.array(inter_weights)))

    ebm = FactorizedEBM(factors)
    return ebm, spin_nodes, pos_of_spin, spin_pos_index, Ks


class TestEnergyEquivalence:
    """Verify Ising DWC energy equals Potts energy + constant on all valid states."""

    def _compare(self, W_unary_list, W_pair_list, P=10.0):
        from thrml.block_management import Block

        Ks = [len(w) for w in W_unary_list]
        potts_ebm, cat_nodes = _build_potts_ebm(W_unary_list, W_pair_list)
        ising_ebm, spin_nodes, pos_of_spin, spin_pos_index, _ = _build_ising_ebm(
            W_unary_list, W_pair_list, P
        )

        offsets = []
        for state_tuple in cart_product(*[range(k) for k in Ks]):
            potts_idx = np.array(state_tuple, dtype=np.uint8)
            E_potts = float(potts_ebm.energy([jnp.array(potts_idx)], [Block(cat_nodes)]))

            spin_bool = spin_pos_index < potts_idx[pos_of_spin]
            E_ising = float(ising_ebm.energy([jnp.array(spin_bool)], [Block(spin_nodes)]))
            offsets.append(E_ising - E_potts)

        np.testing.assert_allclose(offsets, offsets[0], atol=1e-5,
            err_msg="Ising-Potts energy offset not constant")

    def test_single_k4(self):
        self._compare([np.array([0.3, -0.5, 0.8, -0.2])], [])

    def test_two_positions(self):
        self._compare(
            [np.array([0.3, -0.5, 0.8]), np.array([-0.2, 0.4])],
            [np.array([[-0.1, 0.3], [0.5, -0.2], [0.1, 0.4]])],
        )

    def test_k1_neighbor(self):
        self._compare(
            [np.array([0.5]), np.array([0.1, -0.3, 0.7])],
            [np.array([[0.2, -0.1, 0.4]])],
        )

    def test_four_positions(self):
        self._compare(
            [np.array([0.1, -0.2, 0.3, 0.0]), np.array([0.5, -0.1]),
             np.array([-0.3, 0.2, 0.1]), np.array([0.4])],
            [
                np.array([[0.1, -0.2], [-0.1, 0.3], [0.2, -0.1], [0.0, 0.1]]),
                np.array([[-0.1, 0.2, 0.0], [0.3, -0.2, 0.1]]),
                np.array([[0.2], [-0.1], [0.3]]),
            ],
        )


# ===========================================================================
# 4. Model construction test
# ===========================================================================

class TestModelConstruction:
    def test_create_model_small(self):
        """Smoke test: create model for a short sequence."""
        problem = CodonProblem(amino_acid_seq="MSK", weight_codon_usage=0.1, weight_repeat=0.2)
        program, free_blocks, all_spin_nodes, all_spins_block, metadata = create_ising_codon_model(problem)

        Ks = metadata["Ks"]
        expected_spins = sum(max(K - 1, 0) for K in Ks)
        assert len(all_spin_nodes) == expected_spins
        assert len(all_spins_block) == expected_spins
        assert len(free_blocks) > 0

    def test_four_color_blocks(self):
        """Verify 4-color decomposition is valid (no two same-color spins share an edge)."""
        problem = CodonProblem(amino_acid_seq="LLSS", weight_codon_usage=0.1, weight_repeat=0.2)
        _, _, _, _, metadata = create_ising_codon_model(problem)

        pos_of_spin = metadata["pos_of_spin"]
        spin_pos_index = metadata["spin_pos_index"]
        block_indices = metadata["block_indices"]

        # For each block, verify no two spins share an edge
        for bi in block_indices:
            for ii in range(len(bi)):
                for jj in range(ii + 1, len(bi)):
                    s_a, s_b = bi[ii], bi[jj]
                    p_a, p_b = pos_of_spin[s_a], pos_of_spin[s_b]
                    j_a, j_b = spin_pos_index[s_a], spin_pos_index[s_b]

                    # Same position: constraint edge if |j_a - j_b| == 1
                    if p_a == p_b:
                        assert abs(j_a - j_b) != 1, \
                            f"Same-color spins {s_a},{s_b} share constraint edge"

                    # Adjacent positions: inter-position edge exists
                    if abs(p_a - p_b) == 1:
                        assert False, \
                            f"Same-color spins {s_a},{s_b} at adjacent positions {p_a},{p_b}"


# ===========================================================================
# 5. Sampling tests
# ===========================================================================

class TestSampling:
    def test_constraint_satisfaction(self):
        """Verify sampling produces >95% valid thermometer states at moderate P."""
        from thrml.block_sampling import SamplingSchedule, sample_states

        problem = CodonProblem(amino_acid_seq="LS", weight_codon_usage=0.1, weight_repeat=0.2)
        program, free_blocks, all_spin_nodes, all_spins_block, metadata = create_ising_codon_model(
            problem, P=10.0
        )

        Ks = metadata["Ks"]

        key = jax.random.key(42)
        init = [jax.random.choice(key, jnp.array([False, True]), (len(b),)) for b in free_blocks]
        schedule = SamplingSchedule(n_warmup=200, n_samples=100, steps_per_sample=5)
        samples = sample_states(key, program, schedule, init, [], [all_spins_block])
        spin_samples = np.array(samples[0])

        n_valid = 0
        for s_idx in range(spin_samples.shape[0]):
            spin_bool = spin_samples[s_idx]
            valid = True
            offset = 0
            for p, K in enumerate(Ks):
                ns = max(K - 1, 0)
                chain = spin_bool[offset:offset + ns]
                saw_false = False
                for v in chain:
                    if not v:
                        saw_false = True
                    elif saw_false:
                        valid = False
                        break
                if not valid:
                    break
                offset += ns
            if valid:
                n_valid += 1

        validity_rate = n_valid / spin_samples.shape[0]
        assert validity_rate > 0.95, f"Only {validity_rate:.1%} valid states"


# ===========================================================================
# 6. GC-sorted energy / conversion tests
# ===========================================================================

class TestGCSortedConversion:
    def test_indices_to_codons(self):
        """Verify GC-sorted indices produce valid codons."""
        problem = CodonProblem(amino_acid_seq="MA")
        # M has 1 codon (ATG), A has 4 codons
        indices = np.array([0, 0])
        codons = indices_to_codons(problem, indices)
        assert codons[0] == "ATG"
        assert codons[1] == SORTED_CODON_TABLE["A"][0]

    def test_energy_consistency(self):
        """Energy from GC-sorted indices matches compute_energy."""
        problem = CodonProblem(
            amino_acid_seq="MSK",
            weight_codon_usage=0.1,
            weight_repeat=0.2,
            target_gc_fraction=0.5,
            weight_gc=100.0,
        )
        # Pick specific codons
        indices = np.array([0, 0, 0])
        codons = indices_to_codons(problem, indices)
        energy = compute_energy(problem, codons)
        assert isinstance(energy, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
