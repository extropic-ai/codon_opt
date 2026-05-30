#!/usr/bin/env python3
"""Refined parameter sweep focusing on low mutation rates and efficiency."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

qodon_path = Path(__file__).parent.parent
if str(qodon_path) not in sys.path:
    sys.path.insert(0, str(qodon_path))

from constants import gc_constant, repeat_constant, rarity_constant, codon_table, codon_scores, code_map

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


def run_config(pop_size, n_elite, n_lucky, n_generations, mutation_rate, n_chains, n_seeds=5):
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
    result = ga.run(jax.random.key(0))
    jax.block_until_ready(result["best_indices"])

    # Run with multiple seeds
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
            chain_scores = [c["total"] for c in result["best_per_chain"]]
            mean_bests.append(np.mean(chain_scores))
        else:
            mean_bests.append(result["best_score"]["total"])

    total_evals = pop_size * n_generations * n_chains
    return {
        "pop_size": pop_size, "n_elite": n_elite, "n_lucky": n_lucky,
        "n_generations": n_generations, "mutation_rate": mutation_rate,
        "n_chains": n_chains, "total_evals": total_evals,
        "best_score": min(best_scores), "mean_best": np.mean(best_scores),
        "std_best": np.std(best_scores),
        "mean_chain_avg": np.mean(mean_bests),
        "run_time": t_total / n_seeds,
    }


def print_row(r):
    print(f"{r['pop_size']:>5} {r['n_elite']:>5} {r['n_lucky']:>5} {r['n_generations']:>6} "
          f"{r['mutation_rate']:>6.3f} {r['n_chains']:>6} {r['total_evals']:>12,} "
          f"{r['best_score']:>7.1f} {r['mean_best']:>7.1f}±{r['std_best']:>4.1f} "
          f"{r['mean_chain_avg']:>7.1f} {r['run_time']:>6.2f}s")


def print_header():
    print(f"{'pop':>5} {'elite':>5} {'lucky':>5} {'gens':>6} {'mut':>6} {'chains':>6} "
          f"{'evals':>12} {'best':>7} {'mean±std':>13} {'ch_avg':>7} {'time':>7}")
    print("-" * 100)


def main():
    print("=" * 100)
    print("Refined GA Parameter Sweep")
    print(f"Target: ~445 (Potts model), Sequence: {len(SPIKE_PROTEIN_SEQ)} aa")
    print(f"Weights: gc={gc_constant}, repeat={repeat_constant}, rarity={rarity_constant}")
    print("=" * 100)

    results = []

    # Phase 1: Fine-tune mutation rate (key finding from coarse sweep)
    print("\n--- Phase 1: Mutation rate fine-tuning (pop=200, gens=1000, chains=32) ---")
    print_header()
    for mut in [0.003, 0.005, 0.007, 0.01, 0.012, 0.015]:
        r = run_config(200, 30, 10, 1000, mut, 32, n_seeds=5)
        results.append(r)
        print_row(r)

    # Phase 2: With best mutation rate, explore pop/gens trade-off
    print("\n--- Phase 2: Population/generations trade-off (mut=0.01, chains=32) ---")
    print_header()
    configs = [
        (100, 15, 5, 500),
        (100, 15, 5, 1000),
        (100, 15, 5, 2000),
        (200, 30, 10, 500),
        (200, 30, 10, 1000),
        (200, 30, 10, 2000),
        (500, 50, 10, 500),
        (500, 50, 10, 1000),
        (500, 50, 10, 2000),
        (1000, 100, 20, 500),
        (1000, 100, 20, 1000),
    ]
    for pop, elite, lucky, gens in configs:
        r = run_config(pop, elite, lucky, gens, 0.01, 32, n_seeds=5)
        results.append(r)
        print_row(r)

    # Phase 3: Chain scaling with best single-chain configs
    print("\n--- Phase 3: Chain scaling (mut=0.01) ---")
    print_header()
    for n_chains in [1, 8, 32, 64, 128, 256]:
        r = run_config(500, 50, 10, 1000, 0.01, n_chains, n_seeds=5)
        results.append(r)
        print_row(r)

    # Phase 4: Push for lowest score - big configs
    print("\n--- Phase 4: Pushing for best score ---")
    print_header()
    configs = [
        (500, 50, 10, 2000, 0.01, 64),
        (500, 50, 10, 3000, 0.01, 32),
        (500, 50, 10, 5000, 0.01, 32),
        (1000, 100, 20, 2000, 0.01, 32),
        (1000, 100, 20, 3000, 0.01, 16),
        (200, 30, 10, 5000, 0.01, 32),
        # Try even lower mutation rates
        (500, 50, 10, 2000, 0.005, 32),
        (500, 50, 10, 2000, 0.007, 32),
    ]
    for pop, elite, lucky, gens, mut, chains in configs:
        r = run_config(pop, elite, lucky, gens, mut, chains, n_seeds=5)
        results.append(r)
        print_row(r)

    # Summary
    print("\n" + "=" * 100)
    print("TOP 15 CONFIGS BY BEST SCORE")
    print("=" * 100)
    sorted_results = sorted(results, key=lambda r: r["best_score"])
    print_header()
    for r in sorted_results[:15]:
        print_row(r)

    # Find configs closest to Potts score
    print("\n" + "=" * 100)
    print("CONFIGS WITH BEST SCORE < 450 (sorted by total_evals)")
    print("=" * 100)
    close = sorted([r for r in results if r["best_score"] < 450], key=lambda r: r["total_evals"])
    if close:
        print_header()
        for r in close:
            print_row(r)
    else:
        print("(No config achieved best_score < 450)")

    print("\nCONFIGS WITH MEAN SCORE < 460 (sorted by total_evals)")
    close_mean = sorted([r for r in results if r["mean_best"] < 460], key=lambda r: r["total_evals"])
    if close_mean:
        print_header()
        for r in close_mean:
            print_row(r)
    else:
        print("(No config achieved mean_best < 460)")


if __name__ == "__main__":
    main()
