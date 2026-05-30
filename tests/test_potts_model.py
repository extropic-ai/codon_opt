"""Tests for the codon optimization Potts model."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from potts.model import (
    create_base_codon_model,
    create_init_state,
    run_annealing,
)
from problem import (
    CODON_TABLE,
    INVALID_BIAS,
    MAX_SYNONYMOUS_CODONS,
    SORTED_CODON_TABLE,
    CodonProblem,
    compute_energy,
    compute_pairwise_penalties,
    compute_unary_biases,
    create_gc_counter,
    gc_count,
    get_default_rarity_scores,
    indices_to_codons,
    longest_run,
    repeat_penalty,
)


class TestProblemUtilities(unittest.TestCase):
    def test_gc_count(self):
        self.assertEqual(gc_count("GGG"), 3)
        self.assertEqual(gc_count("AAA"), 0)
        self.assertEqual(gc_count("GCA"), 2)
        self.assertEqual(gc_count("ATT"), 0)

    def test_longest_run(self):
        self.assertEqual(longest_run("ABCDEF"), 1)
        self.assertEqual(longest_run("AABCDE"), 2)
        self.assertEqual(longest_run("AAABCD"), 3)
        self.assertEqual(longest_run("ABCCCC"), 4)
        self.assertEqual(longest_run("AABBCC"), 2)

    def test_repeat_penalty(self):
        # No repeat (longest run = 1): penalty = 0
        self.assertEqual(repeat_penalty("ATG", "CGA"), 0)
        # Run of 2: penalty = 3
        self.assertEqual(repeat_penalty("ATG", "GCA"), 3)  # GG in the middle
        # Run of 3: penalty = 8
        self.assertEqual(repeat_penalty("AGG", "GCA"), 8)  # GGG
        # Run of 4: penalty = 15
        self.assertEqual(repeat_penalty("AGG", "GGC"), 15)  # GGGG

    def test_indices_to_codons(self):
        problem = CodonProblem("MW")  # Met has 1 codon, Trp has 1 codon
        codons = indices_to_codons(problem, np.array([0, 0]))
        self.assertEqual(codons, ["ATG", "TGG"])

    def test_invalid_index_raises(self):
        problem = CodonProblem("M")  # Met has only 1 codon
        with self.assertRaises(ValueError):
            indices_to_codons(problem, np.array([1]))  # Index 1 is invalid

    def test_sorted_codon_table_gc_order(self):
        """Verify SORTED_CODON_TABLE is sorted by (gc_count, alphabetical)."""
        for aa, codons in SORTED_CODON_TABLE.items():
            gc_counts = [gc_count(c) for c in codons]
            for i in range(len(codons) - 1):
                self.assertLessEqual(gc_counts[i], gc_counts[i + 1])
                if gc_counts[i] == gc_counts[i + 1]:
                    self.assertLessEqual(codons[i], codons[i + 1])
            # Same set of codons as CODON_TABLE
            self.assertEqual(set(codons), set(CODON_TABLE[aa]))


class TestUnaryBiases(unittest.TestCase):
    def test_shape(self):
        problem = CodonProblem("MSKL")
        biases, Ks = compute_unary_biases(problem)
        self.assertEqual(biases.shape, (4, MAX_SYNONYMOUS_CODONS))
        self.assertEqual(Ks, [1, 6, 2, 6])

    def test_invalid_slots_masked(self):
        problem = CodonProblem("M")  # Met has only 1 codon
        biases, _ = compute_unary_biases(problem)
        # First slot should be valid (not INVALID_BIAS)
        self.assertGreater(biases[0, 0], INVALID_BIAS + 1)
        # Other slots should be invalid
        for j in range(1, MAX_SYNONYMOUS_CODONS):
            self.assertAlmostEqual(float(biases[0, j]), INVALID_BIAS, places=5)

    def test_lower_rarity_higher_bias(self):
        # Create a problem with non-uniform rarity scores
        rarity_scores = get_default_rarity_scores()
        rarity_scores["GCT"] = 0.0  # Make one alanine codon very common (lowest rarity)
        rarity_scores["GCC"] = 2.0  # Make another rare (high rarity)
        problem = CodonProblem("A", codon_rarity_scores=rarity_scores)
        biases, _ = compute_unary_biases(problem)
        # In GC-sorted order for A: GCA(2), GCT(2), GCC(3), GCG(3)
        # GCT is at index 1 (rarity=0), GCC is at index 2 (rarity=2)
        self.assertGreater(biases[0, 1], biases[0, 2])


class TestPairwisePenalties(unittest.TestCase):
    def test_shape(self):
        problem = CodonProblem("MSKL")
        pairwise = compute_pairwise_penalties(problem)
        # L-1 pairs, each is n_cats x n_cats
        self.assertEqual(pairwise.shape, (3, MAX_SYNONYMOUS_CODONS, MAX_SYNONYMOUS_CODONS))

    def test_penalties_are_negative(self):
        # Penalties should be converted to negative weights
        problem = CodonProblem("LL")  # Leucine has many codons
        pairwise = compute_pairwise_penalties(problem)
        # All valid entries should be <= 0 (penalties converted to negative weights)
        # Invalid entries are 0
        self.assertTrue(np.all(pairwise <= 0))

    def test_specific_penalty(self):
        problem = CodonProblem("LL", weight_repeat=1.0)
        pairwise = compute_pairwise_penalties(problem)
        # In GC-sorted order for L: TTA(0), CTT(1), CTA(1), TTG(1), CTC(2), CTG(2)
        # Index 0 = TTA, so pair (TTA, TTA) = TTATTA, longest run = 2, penalty = 3
        self.assertAlmostEqual(float(pairwise[0, 0, 0]), -3.0, places=5)


class TestModelCreation(unittest.TestCase):
    def test_creates_program(self):
        problem = CodonProblem("MSK")
        program, free_blocks, nodes, all_nodes_block = create_base_codon_model(problem)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(all_nodes_block.nodes), 3)
        # 2 free blocks (even/odd)
        self.assertEqual(len(free_blocks), 2)

    def test_init_state_valid(self):
        problem = CodonProblem("MSK")
        key = jax.random.key(42)
        init_state = create_init_state(problem, key)
        # Should have 2 blocks
        self.assertEqual(len(init_state), 2)
        # Even block (M, K): 2 positions
        self.assertEqual(init_state[0].shape, (2,))
        # Odd block (S): 1 position
        self.assertEqual(init_state[1].shape, (1,))
        # All indices should be valid
        self.assertTrue(init_state[0][0] < problem.num_codons_at(0))  # M: 1 codon
        self.assertTrue(init_state[1][0] < problem.num_codons_at(1))  # S: 6 codons
        self.assertTrue(init_state[0][1] < problem.num_codons_at(2))  # K: 2 codons


class TestEnergy(unittest.TestCase):
    def test_energy_computation(self):
        problem = CodonProblem("MW", weight_codon_usage=1.0, weight_repeat=1.0)
        # ATG + TGG: ATGTGG has longest run GG = 2, penalty = 3
        codons = ["ATG", "TGG"]
        energy, terms = compute_energy(problem, codons, return_terms=True)
        self.assertEqual(terms["repeat"], 3.0)
        # M and W have only 1 codon each, so rarity score = 0 (they are the most common)
        self.assertAlmostEqual(terms["usage"], 0.0, places=5)

    def test_usage_term_with_rarity_scores(self):
        # Test usage term with custom rarity scores
        rarity_scores = get_default_rarity_scores()
        rarity_scores["TTA"] = 1.5  # Set a specific rarity score
        rarity_scores["TTG"] = 0.5
        problem = CodonProblem("LL", codon_rarity_scores=rarity_scores, weight_codon_usage=1.0, weight_repeat=0.0)
        # Two positions using TTA with rarity 1.5 each
        codons = ["TTA", "TTA"]
        _, terms = compute_energy(problem, codons, return_terms=True)
        expected_usage = 2 * 1.5  # Two positions, each with rarity 1.5
        self.assertAlmostEqual(terms["usage"], expected_usage, places=5)

        # Test with different codons
        codons2 = ["TTA", "TTG"]
        _, terms2 = compute_energy(problem, codons2, return_terms=True)
        expected_usage2 = 1.5 + 0.5  # TTA + TTG
        self.assertAlmostEqual(terms2["usage"], expected_usage2, places=5)

    def test_gc_term(self):
        problem = CodonProblem("MW", target_gc_fraction=0.5, weight_gc=10.0)
        codons = ["ATG", "TGG"]  # GC count: 1 + 2 = 3 out of 6, fraction = 0.5
        _, terms = compute_energy(problem, codons, return_terms=True)
        self.assertAlmostEqual(terms["gc"], 0.0, places=5)

        # Different target
        problem2 = CodonProblem("MW", target_gc_fraction=0.0, weight_gc=10.0)
        _, terms2 = compute_energy(problem2, codons, return_terms=True)
        self.assertGreater(terms2["gc"], 0)


class TestAnnealing(unittest.TestCase):
    def test_output_shape_final_only(self):
        problem = CodonProblem("MSK")
        key = jax.random.key(42)
        betas = jnp.array([0.1, 1.0, 5.0])
        final_states, stats, _ = run_annealing(
            problem, key, n_chains=4, steps_per_beta=3, betas=betas,
            gc_coeff_adapt_mult=None, only_return_final=True
        )
        self.assertEqual(final_states.shape, (4, 3))  # (n_chains, L)
        self.assertEqual(stats["gc_fractions"].shape, (3, 4))  # (n_betas, n_chains)

    def test_output_shape_trajectory(self):
        problem = CodonProblem("MSK")
        key = jax.random.key(42)
        betas = jnp.array([0.1, 1.0, 5.0])
        trajectory, stats, _ = run_annealing(
            problem, key, n_chains=4, steps_per_beta=3, betas=betas,
            gc_coeff_adapt_mult=None, only_return_final=False
        )
        self.assertEqual(trajectory.shape, (4, 3, 3))  # (n_chains, n_betas, L)
        self.assertEqual(stats["gc_fractions"].shape, (3, 4))  # (n_betas, n_chains)

    def test_valid_indices(self):
        problem = CodonProblem("MWL")  # M:1 codon, W:1 codon, L:6 codons
        key = jax.random.key(123)
        betas = jnp.array([1.0, 2.0])
        final_states, _, _ = run_annealing(
            problem, key, n_chains=8, steps_per_beta=5, betas=betas,
            gc_coeff_adapt_mult=None, only_return_final=True
        )
        # Check all indices are valid for each position
        self.assertTrue(np.all(final_states[:, 0] < 1))  # M has 1 codon
        self.assertTrue(np.all(final_states[:, 1] < 1))  # W has 1 codon
        self.assertTrue(np.all(final_states[:, 2] < 6))  # L has 6 codons

    def test_high_beta_finds_good_solutions(self):
        # Annealing to high beta should find lower energy than random
        problem = CodonProblem("LLLL", weight_repeat=1.0, weight_codon_usage=0.0)
        key = jax.random.key(0)
        betas = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        final_states, _, _ = run_annealing(
            problem, key, n_chains=16, steps_per_beta=20, betas=betas,
            gc_coeff_adapt_mult=None, only_return_final=True
        )

        # Compute energies of annealed solutions
        annealed_energies = []
        for i in range(final_states.shape[0]):
            codon_seq = indices_to_codons(problem, np.array(final_states[i]))
            e = compute_energy(problem, codon_seq)
            annealed_energies.append(e)

        # Random baseline: sample uniformly
        random_key = jax.random.key(999)
        random_indices = jax.random.randint(random_key, (16, 4), 0, 6, dtype=jnp.uint8)
        random_energies = []
        for i in range(16):
            codon_seq = indices_to_codons(problem, np.array(random_indices[i]))
            e = compute_energy(problem, codon_seq)
            random_energies.append(e)

        # Annealed mean should be lower than random mean
        self.assertLess(np.mean(annealed_energies), np.mean(random_energies))


class TestGCAdaptation(unittest.TestCase):
    def test_gc_adaptation_returns_gc_coeff_stats(self):
        problem = CodonProblem("MSKL", target_gc_fraction=0.5)
        key = jax.random.key(42)
        betas = jnp.array([0.1, 1.0, 5.0])
        _, stats, _ = run_annealing(
            problem, key, n_chains=4, steps_per_beta=3, betas=betas,
            gc_coeff_adapt_mult=jnp.array(1.0), only_return_final=True
        )
        self.assertIn("gc_coeffs", stats)
        self.assertEqual(stats["gc_coeffs"].shape, (3, 4))  # (n_betas, n_chains)

    def test_gc_adaptation_adjusts_coefficient(self):
        # With a high target GC, the coefficient should become positive to encourage GC
        problem_high_gc = CodonProblem("LLLL", target_gc_fraction=0.9)
        key = jax.random.key(0)
        betas = jnp.array([1.0] * 10)  # Multiple steps at same beta to see adaptation
        _, stats_high, _ = run_annealing(
            problem_high_gc, key, n_chains=8, steps_per_beta=5, betas=betas,
            gc_coeff_adapt_mult=jnp.array(1.0), only_return_final=True
        )
        # With a low target GC, the coefficient should become negative to penalize GC
        problem_low_gc = CodonProblem("LLLL", target_gc_fraction=0.1)
        _, stats_low, _ = run_annealing(
            problem_low_gc, key, n_chains=8, steps_per_beta=5, betas=betas,
            gc_coeff_adapt_mult=jnp.array(1.0), only_return_final=True
        )
        # Final GC coefficients (mean across chains) should differ in the expected direction
        self.assertGreater(
            float(stats_high["gc_coeffs"][-1].mean()),
            float(stats_low["gc_coeffs"][-1].mean())
        )

    def test_gc_adaptation_moves_gc_fraction_toward_target(self):
        # Test that GC adaptation actually helps move GC fraction toward target
        target_gc = 0.6
        problem = CodonProblem("LLLLLLLL", target_gc_fraction=target_gc)
        key = jax.random.key(123)
        betas = jnp.logspace(-1, 1, 20)

        # With adaptation
        _, stats_adapted, _ = run_annealing(
            problem, key, n_chains=16, steps_per_beta=10, betas=betas,
            gc_coeff_adapt_mult=jnp.array(2.0), only_return_final=True
        )

        # Without adaptation
        _, stats_unadapted, _ = run_annealing(
            problem, key, n_chains=16, steps_per_beta=10, betas=betas,
            gc_coeff_adapt_mult=None, only_return_final=True
        )

        # Final GC fraction (mean across chains) with adaptation should be closer to target
        final_gc_adapted = float(stats_adapted["gc_fractions"][-1].mean())
        final_gc_unadapted = float(stats_unadapted["gc_fractions"][-1].mean())
        error_adapted = abs(final_gc_adapted - target_gc)
        error_unadapted = abs(final_gc_unadapted - target_gc)
        self.assertLess(error_adapted, error_unadapted)


class TestGCCounter(unittest.TestCase):
    def test_basic_gc_count(self):
        problem = CodonProblem("MSK")
        gc_counter = create_gc_counter(problem)
        # GC-sorted: M=ATG(1gc), S=AGT(1),TCA(1),TCT(1),AGC(2),TCC(2),TCG(2), K=AAA(0),AAG(1)
        # [0, 0, 0] -> ATG(1gc), AGT(1gc), AAA(0gc) -> 1 + 1 + 0 = 2
        indices = jnp.array([0, 0, 0], dtype=jnp.uint8)
        self.assertEqual(int(gc_counter(indices)), 2)

    def test_gc_count_varies_with_codon_choice(self):
        problem = CodonProblem("MSK")
        gc_counter = create_gc_counter(problem)
        # GC-sorted: M=ATG(1gc), S=AGT(1),TCA(1),TCT(1),AGC(2),TCC(2),TCG(2), K=AAA(0),AAG(1)
        # [0, 3, 1] -> ATG(1gc), AGC(2gc), AAG(1gc) -> 1 + 2 + 1 = 4
        indices = jnp.array([0, 3, 1], dtype=jnp.uint8)
        self.assertEqual(int(gc_counter(indices)), 4)

    def test_gc_count_all_gc(self):
        # Glycine: GC-sorted = GGA(2), GGT(2), GGC(3), GGG(3)
        problem = CodonProblem("GG")
        gc_counter = create_gc_counter(problem)
        # [2, 2] -> GGC(3gc), GGC(3gc) -> 3 + 3 = 6
        indices = jnp.array([2, 2], dtype=jnp.uint8)
        self.assertEqual(int(gc_counter(indices)), 6)

    def test_gc_count_no_gc(self):
        # F: GC-sorted = TTT(0), TTC(1); K: AAA(0), AAG(1)
        problem = CodonProblem("FK")
        gc_counter = create_gc_counter(problem)
        # [0, 0] -> TTT(0gc), AAA(0gc) -> 0
        indices = jnp.array([0, 0], dtype=jnp.uint8)
        self.assertEqual(int(gc_counter(indices)), 0)


class TestLinearGCCoeff(unittest.TestCase):
    def test_zero_coeff_matches_original(self):
        problem = CodonProblem("MSK")
        biases_default, _ = compute_unary_biases(problem)
        biases_zero, _ = compute_unary_biases(problem, linear_gc_coeff=jnp.array(0.0))
        np.testing.assert_array_almost_equal(biases_default, biases_zero)

    def test_positive_coeff_increases_high_gc_bias(self):
        problem = CodonProblem("S")
        # S GC-sorted: AGT(1gc), TCA(1gc), TCT(1gc), AGC(2gc), TCC(2gc), TCG(2gc)
        biases_no_gc, _ = compute_unary_biases(problem, linear_gc_coeff=jnp.array(0.0))
        biases_pos_gc, _ = compute_unary_biases(problem, linear_gc_coeff=jnp.array(1.0))
        diff = biases_pos_gc - biases_no_gc
        # Index 0 = AGT (1gc), Index 3 = AGC (2gc)
        self.assertAlmostEqual(float(diff[0, 0]), 1.0, places=5)  # AGT: 1 GC
        self.assertAlmostEqual(float(diff[0, 3]), 2.0, places=5)  # AGC: 2 GC

    def test_negative_coeff_decreases_high_gc_bias(self):
        problem = CodonProblem("S")
        biases_no_gc, _ = compute_unary_biases(problem, linear_gc_coeff=jnp.array(0.0))
        biases_neg_gc, _ = compute_unary_biases(problem, linear_gc_coeff=jnp.array(-1.0))
        diff = biases_neg_gc - biases_no_gc
        self.assertAlmostEqual(float(diff[0, 0]), -1.0, places=5)  # AGT: 1 GC
        self.assertAlmostEqual(float(diff[0, 3]), -2.0, places=5)  # AGC: 2 GC

    def test_create_base_codon_model_accepts_linear_gc_coeff(self):
        problem = CodonProblem("MSK")
        # Should not raise
        program, free_blocks, nodes, all_nodes_block = create_base_codon_model(
            problem, linear_gc_coeff=jnp.array(0.5)
        )
        self.assertEqual(len(nodes), 3)

    def test_linear_gc_coeff_is_differentiable(self):
        """Test that we can take gradients w.r.t. linear_gc_coeff through compute_unary_biases."""
        problem = CodonProblem("MSK")

        def sum_biases(linear_gc_coeff):
            biases, _ = compute_unary_biases(problem, linear_gc_coeff)
            return jnp.sum(biases)

        grad_fn = jax.grad(sum_biases)
        grad = grad_fn(jnp.array(0.0))
        # Gradient should be the sum of GC counts for all valid codon slots
        # M=ATG(1), S=AGT(1),TCA(1),TCT(1),AGC(2),TCC(2),TCG(2), K=AAA(0),AAG(1)
        # Total = 1 + (1+1+1+2+2+2) + (0+1) = 1 + 9 + 1 = 11
        expected_grad = 11.0
        self.assertAlmostEqual(float(grad), expected_grad, places=5)


if __name__ == "__main__":
    unittest.main()
