#!/usr/bin/env python3
"""Parameter sweep for fast GA to find configs that match Potts model score (~445).

Systematically explores GA parameters and reports:
- Best score achieved
- Total fitness evaluations (pop_size * n_generations * n_chains)
- Wall-clock time

The goal is to find GA parameters that achieve score ~445 with minimal compute.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory for qodon imports
qodon_path = Path(__file__).parent.parent
if str(qodon_path) not in sys.path:
    sys.path.insert(0, str(qodon_path))

from constants import gc_constant, repeat_constant, rarity_constant, codon_table, codon_scores, code_map

# Import from this directory
import importlib.util
_this_dir = Path(__file__).parent

spec = importlib.util.spec_from_file_location("fast_ga_scoring", _this_dir / "scoring.py")
fast_ga_scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast_ga_scoring)
ScoringConfig = fast_ga_scoring.ScoringConfig

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


@dataclass
class SweepResult:
    pop_size: int
    n_elite: int
    n_lucky: int
    n_generations: int
    mutation_rate: float
    n_chains: int
    total_evals: int  # pop_size * n_generations * n_chains
    best_score: float
    mean_best_per_chain: float
    compile_time: float
    run_time: float


def run_config(pop_size, n_elite, n_lucky, n_generations, mutation_rate, n_chains, n_seeds=3):
    """Run a single GA configuration with multiple seeds and return results."""
    scoring_config = ScoringConfig(
        gc_weight=gc_constant,
        repeat_weight=repeat_constant,
        rarity_weight=rarity_constant,
    )

    ga_config = GAConfig(
        pop_size=pop_size,
        n_elite=n_elite,
        n_lucky=n_lucky,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        n_chains=n_chains,
        seed=42,
    )

    ga = FastGA(
        amino_acid_seq=SPIKE_PROTEIN_SEQ,
        codon_table=codon_table,
        codon_scores=codon_scores,
        code_map=code_map,
        ga_config=ga_config,
        scoring_config=scoring_config,
    )

    # JIT warmup
    t0 = time.time()
    result = ga.run(jax.random.key(0))
    jax.block_until_ready(result["best_indices"])
    t_compile = time.time() - t0

    # Run with multiple seeds and collect best scores
    best_scores = []
    mean_bests = []
    t_total = 0
    for seed in range(n_seeds):
        t0 = time.time()
        result = ga.run(jax.random.key(seed + 1))
        jax.block_until_ready(result["best_indices"])
        t_total += time.time() - t0
        best_scores.append(result["best_score"]["total"])
        if n_chains > 1:
            mean_bests.append(np.mean([c["total"] for c in result["best_per_chain"]]))
        else:
            mean_bests.append(result["best_score"]["total"])

    total_evals = pop_size * n_generations * n_chains

    return SweepResult(
        pop_size=pop_size,
        n_elite=n_elite,
        n_lucky=n_lucky,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        n_chains=n_chains,
        total_evals=total_evals,
        best_score=min(best_scores),
        mean_best_per_chain=np.mean(mean_bests),
        compile_time=t_compile,
        run_time=t_total / n_seeds,
    )


def main():
    print("=" * 90)
    print("GA Parameter Sweep for Codon Optimization")
    print(f"Target score: ~445 (Potts model reference)")
    print(f"Sequence length: {len(SPIKE_PROTEIN_SEQ)} amino acids")
    print(f"Weights: gc={gc_constant}, repeat={repeat_constant}, rarity={rarity_constant}")
    print("=" * 90)

    results = []

    # Phase 1: Understand scaling with generations and population
    # Fix n_chains=1, mutation_rate=0.05, vary pop_size and n_generations
    print("\n--- Phase 1: Population size & generations scaling (1 chain) ---")
    print(f"{'pop':>5} {'elite':>5} {'lucky':>5} {'gens':>6} {'mut':>5} {'chains':>6} "
          f"{'evals':>10} {'best':>8} {'mean_best':>10} {'jit':>6} {'run':>6}")
    print("-" * 90)

    phase1_configs = [
        # (pop_size, n_elite, n_lucky, n_generations, mutation_rate)
        # Baseline
        (62, 10, 2, 100, 0.05),
        # More generations
        (62, 10, 2, 500, 0.05),
        (62, 10, 2, 1000, 0.05),
        (62, 10, 2, 2000, 0.05),
        # Larger population
        (200, 30, 10, 100, 0.05),
        (200, 30, 10, 500, 0.05),
        (200, 30, 10, 1000, 0.05),
        # Even larger population
        (500, 50, 10, 100, 0.05),
        (500, 50, 10, 500, 0.05),
        (500, 50, 10, 1000, 0.05),
        # Very large population, fewer generations
        (1000, 100, 20, 100, 0.05),
        (1000, 100, 20, 500, 0.05),
    ]

    for pop_size, n_elite, n_lucky, n_gens, mut_rate in phase1_configs:
        r = run_config(pop_size, n_elite, n_lucky, n_gens, mut_rate, n_chains=1, n_seeds=3)
        results.append(r)
        print(f"{r.pop_size:>5} {r.n_elite:>5} {r.n_lucky:>5} {r.n_generations:>6} "
              f"{r.mutation_rate:>5.2f} {r.n_chains:>6} {r.total_evals:>10,} "
              f"{r.best_score:>8.1f} {r.mean_best_per_chain:>10.1f} "
              f"{r.compile_time:>5.1f}s {r.run_time:>5.2f}s")

    # Phase 2: Multi-chain scaling (take best across chains)
    print("\n--- Phase 2: Multi-chain scaling ---")
    print(f"{'pop':>5} {'elite':>5} {'lucky':>5} {'gens':>6} {'mut':>5} {'chains':>6} "
          f"{'evals':>10} {'best':>8} {'mean_best':>10} {'jit':>6} {'run':>6}")
    print("-" * 90)

    # Use a moderate single-chain config and scale chains
    phase2_configs = [
        # (pop_size, n_elite, n_lucky, n_generations, mutation_rate, n_chains)
        (62, 10, 2, 500, 0.05, 8),
        (62, 10, 2, 500, 0.05, 32),
        (62, 10, 2, 500, 0.05, 128),
        (62, 10, 2, 500, 0.05, 256),
        (62, 10, 2, 1000, 0.05, 32),
        (62, 10, 2, 1000, 0.05, 128),
        (62, 10, 2, 1000, 0.05, 256),
        (200, 30, 10, 500, 0.05, 32),
        (200, 30, 10, 500, 0.05, 128),
        (200, 30, 10, 500, 0.05, 256),
        (200, 30, 10, 1000, 0.05, 32),
        (200, 30, 10, 1000, 0.05, 128),
    ]

    for pop_size, n_elite, n_lucky, n_gens, mut_rate, n_chains in phase2_configs:
        r = run_config(pop_size, n_elite, n_lucky, n_gens, mut_rate, n_chains, n_seeds=3)
        results.append(r)
        print(f"{r.pop_size:>5} {r.n_elite:>5} {r.n_lucky:>5} {r.n_generations:>6} "
              f"{r.mutation_rate:>5.2f} {r.n_chains:>6} {r.total_evals:>10,} "
              f"{r.best_score:>8.1f} {r.mean_best_per_chain:>10.1f} "
              f"{r.compile_time:>5.1f}s {r.run_time:>5.2f}s")

    # Phase 3: Mutation rate exploration with best configs from above
    print("\n--- Phase 3: Mutation rate exploration ---")
    print(f"{'pop':>5} {'elite':>5} {'lucky':>5} {'gens':>6} {'mut':>5} {'chains':>6} "
          f"{'evals':>10} {'best':>8} {'mean_best':>10} {'jit':>6} {'run':>6}")
    print("-" * 90)

    phase3_configs = [
        (200, 30, 10, 1000, 0.01, 32),
        (200, 30, 10, 1000, 0.02, 32),
        (200, 30, 10, 1000, 0.10, 32),
        (200, 30, 10, 1000, 0.15, 32),
        (200, 30, 10, 1000, 0.20, 32),
    ]

    for pop_size, n_elite, n_lucky, n_gens, mut_rate, n_chains in phase3_configs:
        r = run_config(pop_size, n_elite, n_lucky, n_gens, mut_rate, n_chains, n_seeds=3)
        results.append(r)
        print(f"{r.pop_size:>5} {r.n_elite:>5} {r.n_lucky:>5} {r.n_generations:>6} "
              f"{r.mutation_rate:>5.2f} {r.n_chains:>6} {r.total_evals:>10,} "
              f"{r.best_score:>8.1f} {r.mean_best_per_chain:>10.1f} "
              f"{r.compile_time:>5.1f}s {r.run_time:>5.2f}s")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY: Top 10 configs by best score")
    print("=" * 90)
    sorted_results = sorted(results, key=lambda r: r.best_score)
    print(f"{'pop':>5} {'elite':>5} {'lucky':>5} {'gens':>6} {'mut':>5} {'chains':>6} "
          f"{'evals':>10} {'best':>8} {'mean_best':>10} {'run':>6}")
    print("-" * 90)
    for r in sorted_results[:10]:
        print(f"{r.pop_size:>5} {r.n_elite:>5} {r.n_lucky:>5} {r.n_generations:>6} "
              f"{r.mutation_rate:>5.2f} {r.n_chains:>6} {r.total_evals:>10,} "
              f"{r.best_score:>8.1f} {r.mean_best_per_chain:>10.1f} {r.run_time:>5.2f}s")

    print("\nSUMMARY: Top 10 configs by efficiency (best score / total_evals)")
    print("=" * 90)
    # Only consider configs that got reasonably close to target
    close_results = [r for r in results if r.best_score < 600]
    if close_results:
        sorted_eff = sorted(close_results, key=lambda r: r.total_evals)
        print(f"{'pop':>5} {'elite':>5} {'lucky':>5} {'gens':>6} {'mut':>5} {'chains':>6} "
              f"{'evals':>10} {'best':>8} {'mean_best':>10}")
        print("-" * 90)
        for r in sorted_eff[:10]:
            print(f"{r.pop_size:>5} {r.n_elite:>5} {r.n_lucky:>5} {r.n_generations:>6} "
                  f"{r.mutation_rate:>5.2f} {r.n_chains:>6} {r.total_evals:>10,} "
                  f"{r.best_score:>8.1f} {r.mean_best_per_chain:>10.1f}")
    else:
        print("(No config achieved score < 600)")

    # Potts model comparison
    print("\n" + "=" * 90)
    print("POTTS MODEL REFERENCE")
    print("=" * 90)
    potts_steps = 10
    potts_chains = 256
    L = len(SPIKE_PROTEIN_SEQ)
    potts_node_updates = potts_steps * potts_chains * L
    print(f"Potts model: {potts_steps} Gibbs steps x {potts_chains} chains x {L} positions = "
          f"{potts_node_updates:,} node updates")
    print(f"Potts model best score: ~444")
    print(f"Potts model mean score: ~445 +/- 1")


if __name__ == "__main__":
    main()
