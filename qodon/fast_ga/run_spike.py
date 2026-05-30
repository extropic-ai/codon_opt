#!/usr/bin/env python3
"""Run the fast GA on the spike protein sequence and compare with original."""

import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory for qodon imports
qodon_path = Path(__file__).parent.parent
if str(qodon_path) not in sys.path:
    sys.path.insert(0, str(qodon_path))

from constants import gc_constant, repeat_constant, rarity_constant, codon_table, codon_scores, code_map
from scoring import SeqScorer

# Import from this directory explicitly
import importlib.util
_this_dir = Path(__file__).parent

spec = importlib.util.spec_from_file_location("fast_ga_scoring", _this_dir / "scoring.py")
fast_ga_scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast_ga_scoring)
ScoringConfig = fast_ga_scoring.ScoringConfig
verify_scoring_matches_qodon = fast_ga_scoring.verify_scoring_matches_qodon

spec = importlib.util.spec_from_file_location("fast_ga_ga", _this_dir / "ga.py")
fast_ga_ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast_ga_ga)
FastGA = fast_ga_ga.FastGA
GAConfig = fast_ga_ga.GAConfig

# Spike protein sequence
SPIKE_PROTEIN_SEQ = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS"
    "NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLI"
    "VNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMD"
    "LEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITR"
    "FQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPL"
    "SETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRK"
    "RISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQT"
    "GKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQA"
    "GSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLV"
    "KNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFG"
    "GVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIG"
    "AEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNS"
    "IAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGI"
    "AVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTL"
    "ADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWT"
    "FGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALG"
    "KLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQ"
    "TYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLH"
    "VTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVS"
    "GNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKE"
    "IDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCS"
    "CLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
).replace(" ", "")


def verify_scoring():
    """Verify that JAX scoring matches qodon's SeqScorer."""
    print("=" * 70)
    print("Verifying JAX scoring matches qodon SeqScorer")
    print("=" * 70)

    test_aa = "MFVFLVLLPL"
    scoring_config = ScoringConfig(
        gc_weight=gc_constant,
        repeat_weight=repeat_constant,
        rarity_weight=rarity_constant,
    )

    import random
    random.seed(42)
    test_nuc = ""
    for aa in test_aa:
        codons = codon_table[aa]
        test_nuc += random.choice(codons)

    result = verify_scoring_matches_qodon(
        test_nuc, codon_table, codon_scores, test_aa, scoring_config
    )

    print(f"\nTest sequence: {test_aa}")
    print(f"Nucleotides:   {test_nuc}")
    for name in ["qodon", "jax"]:
        r = result[name]
        print(f"\n{name} scores:")
        print(f"  total:      {r['total']:.6f}")
        print(f"  gc_raw:     {r['gc_raw']:.6f}")
        print(f"  repeat_raw: {r['repeat_raw']:.6f}")
        print(f"  rarity_raw: {r['rarity_raw']:.6f}")

    tol = 1e-5
    rel_tol = 1e-4
    match = all(
        abs(result['qodon'][k] - result['jax'][k]) < max(tol, rel_tol * abs(result['qodon'][k]))
        for k in ['total', 'gc_raw', 'repeat_raw', 'rarity_raw']
    )
    print(f"\nScoring match: {'PASS' if match else 'FAIL'}")
    return match


def run_fast_ga_benchmark():
    """Run the fast GA and report results."""
    print("\n" + "=" * 70)
    print("Running Fast GA on Spike Protein (single chain)")
    print("=" * 70)

    print(f"\nSequence length: {len(SPIKE_PROTEIN_SEQ)} amino acids")

    scoring_config = ScoringConfig(
        gc_weight=gc_constant,
        repeat_weight=repeat_constant,
        rarity_weight=rarity_constant,
    )

    ga_config = GAConfig(
        pop_size=200,
        n_elite=10,
        n_lucky=2,
        n_generations=1000,
        mutation_rate=0.003,
        n_chains=1,
        seed=42,
    )

    print(f"\nGA parameters: pop_size={ga_config.pop_size}, n_elite={ga_config.n_elite}, "
          f"n_lucky={ga_config.n_lucky}, n_offspring={ga_config.n_offspring}, "
          f"n_generations={ga_config.n_generations}, mutation_rate={ga_config.mutation_rate}")
    print(f"Weights: gc={scoring_config.gc_weight}, repeat={scoring_config.repeat_weight}, "
          f"rarity={scoring_config.rarity_weight}")

    ga = FastGA(
        amino_acid_seq=SPIKE_PROTEIN_SEQ,
        codon_table=codon_table,
        codon_scores=codon_scores,
        code_map=code_map,
        ga_config=ga_config,
        scoring_config=scoring_config,
    )

    # JIT warmup
    print("\nJIT compiling...")
    import jax
    t0 = time.time()
    result = ga.run(jax.random.key(0))
    jax.block_until_ready(result["best_indices"])
    t_compile = time.time() - t0
    print(f"First run (includes JIT): {t_compile:.2f}s")

    # Timed run
    t0 = time.time()
    result = ga.run(jax.random.key(42))
    jax.block_until_ready(result["best_indices"])
    t_run = time.time() - t0
    print(f"Second run (JIT cached):  {t_run:.2f}s")

    best = result["best_score"]
    print(f"\nBest score: {best['total']:.4f}")
    print(f"  GC:     {best['gc']:.4f} (frac: {best['gc_frac']:.4f})")
    print(f"  Repeat: {best['repeat']:.4f} (raw: {best['repeat_raw']:.1f})")
    print(f"  Rarity: {best['rarity']:.4f} (raw: {best['rarity_raw']:.4f})")

    # Verify with qodon
    qodon_scorer = SeqScorer(result["best_nucleotides"])
    print(f"\nVerification with qodon SeqScorer: {qodon_scorer.score:.4f} "
          f"({'PASS' if abs(qodon_scorer.score - best['total']) < 0.1 else 'FAIL'})")

    history = result["history"]
    print(f"\nConvergence: {history['best_scores'][0]:.1f} -> {history['best_scores'][-1]:.1f}")

    return result, t_run


def run_multi_chain_benchmark(gc_weight, repeat_weight, rarity_weight, label):
    """Run the fast GA with 512 chains and report mean ± std."""
    print("\n" + "=" * 70)
    print(f"Running Fast GA with 512 chains — {label}")
    print("=" * 70)

    scoring_config = ScoringConfig(
        gc_weight=gc_weight,
        repeat_weight=repeat_weight,
        rarity_weight=rarity_weight,
    )

    ga_config = GAConfig(
        pop_size=200,
        n_elite=10,
        n_lucky=2,
        n_generations=1000,
        mutation_rate=0.003,
        n_chains=512,
        seed=42,
    )

    print(f"\nGA parameters: pop_size={ga_config.pop_size}, n_elite={ga_config.n_elite}, "
          f"n_lucky={ga_config.n_lucky}, n_offspring={ga_config.n_offspring}, "
          f"n_generations={ga_config.n_generations}, mutation_rate={ga_config.mutation_rate}")
    print(f"Weights: gc={scoring_config.gc_weight}, repeat={scoring_config.repeat_weight}, "
          f"rarity={scoring_config.rarity_weight}")

    ga = FastGA(
        amino_acid_seq=SPIKE_PROTEIN_SEQ,
        codon_table=codon_table,
        codon_scores=codon_scores,
        code_map=code_map,
        ga_config=ga_config,
        scoring_config=scoring_config,
    )

    import jax

    # JIT warmup
    print("\nJIT compiling...")
    t0 = time.time()
    result = ga.run(jax.random.key(0))
    jax.block_until_ready(result["best_indices"])
    t_compile = time.time() - t0
    print(f"First run (includes JIT): {t_compile:.2f}s")

    # Timed run
    t0 = time.time()
    result = ga.run(jax.random.key(42))
    jax.block_until_ready(result["best_indices"])
    t_run = time.time() - t0
    print(f"Second run (JIT cached):  {t_run:.2f}s")

    # Report mean ± std over all chains
    chain_scores = np.array([cs["total"] for cs in result["best_per_chain"]])
    mean_score = np.mean(chain_scores)
    std_score = np.std(chain_scores)
    best_score = np.min(chain_scores)
    print(f"\nResults over {len(chain_scores)} chains:")
    print(f"  Mean score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"  Best score: {best_score:.4f}")
    print(f"  Overall best (chain {result['best_chain']}): {result['best_score']['total']:.4f}")

    return result, t_run


def compare_with_original():
    """Compare speed with original qodon GA."""
    print("\n" + "=" * 70)
    print("Comparison with Original qodon GA")
    print("=" * 70)

    from classical_ga import CodonOptimization

    print("\nRunning original GA...")
    t0 = time.time()
    original = CodonOptimization(SPIKE_PROTEIN_SEQ)
    t_original = time.time() - t0
    original_score = SeqScorer(original.n_seq).score
    print(f"  Time:  {t_original:.2f}s")
    print(f"  Score: {original_score:.4f}")

    return t_original, original_score


def main():
    if not verify_scoring():
        print("\nScoring verification failed!")
        return

    # Easy (Fox) parameters: gc=1.0, repeat=0.1, rarity=0.1
    run_multi_chain_benchmark(
        gc_weight=gc_constant, repeat_weight=repeat_constant, rarity_weight=rarity_constant,
        label="Easy (Fox) params",
    )

    # Hard parameters: gc=2e4, repeat=0.2, rarity=0.1
    run_multi_chain_benchmark(
        gc_weight=2e4, repeat_weight=0.2, rarity_weight=0.1,
        label="Hard params",
    )


if __name__ == "__main__":
    main()
