#!/usr/bin/env python3
"""Energy consumption comparison: GA (CPU/GPU) vs Potts model (thermodynamic chip).

Estimates energy via:
1. FLOP counting + idealized J/FLOP for modern hardware
2. Measured wall-clock time + hardware TDP (actual benchmark)
"""

import sys
import time
import os
from pathlib import Path

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


def estimate_ga_flops(pop_size, n_elite, n_lucky, n_generations, mutation_rate, n_chains, L, max_codons=6):
    """Estimate total FLOPs for one GA run.

    Counts arithmetic operations per generation step:
    1. Scoring (all individuals):
       - GC: L lookups + L-1 additions + 1 division + 1 subtraction + 1 square + 1 multiply = L + L + 4 ≈ 2L + 4
       - Rarity: L lookups + L-1 additions + 1 multiply = 2L + 1
       - Repeat: (L-1) lookups + (L-2) additions + 1 multiply = 2(L-1) + 1
       - Total per individual: ~6L + 6 ≈ 6L
       - Total scoring: pop_size × 6L

    2. Sorting: pop_size × log2(pop_size) comparisons ≈ pop_size × log2(pop_size) × 2

    3. Selection (gather operations): (n_elite + n_lucky) × L

    4. Crossover per offspring:
       - 2 random ints + 2 gathers (2×L) + L random floats + L comparisons + L selects = 5L + 2
       Total: n_offspring × 5L

    5. Mutation per offspring:
       - L random floats + L comparisons (mutation mask)
       - L × max_codons random floats + L × max_codons comparisons + L sum (for sampling)
       - L selects (where)
       Total: n_offspring × (3L + 2L × max_codons) ≈ n_offspring × (3 + 2×max_codons)L
       But most mutations don't fire (rate is low), so the random generation dominates.
       Total: n_offspring × (3 + 2×max_codons)L

    6. Concatenation: pop_size × L (memory operations, ~1 FLOP each)

    Total per generation per chain ≈ pop_size × 6L + n_offspring × (8 + 2×max_codons)L + overhead
    """
    n_offspring = pop_size - n_elite - n_lucky
    log2_pop = np.log2(pop_size)

    # FLOPs per generation per chain
    scoring = pop_size * 6 * L
    sorting = pop_size * log2_pop * 2
    selection = (n_elite + n_lucky) * L
    crossover = n_offspring * 5 * L
    mutation = n_offspring * (3 + 2 * max_codons) * L
    concat = pop_size * L

    flops_per_gen = scoring + sorting + selection + crossover + mutation + concat
    total = flops_per_gen * n_generations * n_chains

    return {
        "per_gen_per_chain": flops_per_gen,
        "total": total,
        "breakdown": {
            "scoring": scoring * n_generations * n_chains,
            "sorting": sorting * n_generations * n_chains,
            "selection": selection * n_generations * n_chains,
            "crossover": crossover * n_generations * n_chains,
            "mutation": mutation * n_generations * n_chains,
            "concat": concat * n_generations * n_chains,
        }
    }


def estimate_potts_flops(n_steps, n_chains, L, max_codons=6):
    """Estimate total FLOPs for Potts model Gibbs sampling.

    Per position update (computing conditional distribution):
    - For each of max_codons options:
      - GC term: delta_gc computation, O(1) if tracked incrementally → ~3 FLOPs
      - Repeat term: check left+right neighbors → ~4 FLOPs (2 lookups + 2 additions)
      - Rarity term: 1 lookup → ~1 FLOP
      Total per option: ~8 FLOPs
    - Softmax over max_codons options: max_codons exponentials + sum + division → ~3×max_codons FLOPs
    - Sampling: 1 random float + max_codons comparisons → ~max_codons + 1 FLOPs

    Per position: max_codons × 8 + 3 × max_codons + max_codons + 1 ≈ 12 × max_codons + 1

    Per full sweep: L × (12 × max_codons + 1)
    Total: n_steps × n_chains × L × (12 × max_codons + 1)
    """
    flops_per_position = 12 * max_codons + 1
    flops_per_sweep = L * flops_per_position
    total = n_steps * n_chains * flops_per_sweep

    return {
        "per_sweep": flops_per_sweep,
        "total": total,
        "per_position": flops_per_position,
    }


def run_benchmark(platform="gpu"):
    """Run the GA benchmark and return timing results."""
    import jax
    import jax.numpy as jnp

    scoring_config = ScoringConfig(
        gc_weight=gc_constant,
        repeat_weight=repeat_constant,
        rarity_weight=rarity_constant,
    )

    # Best efficient config matching Potts score
    ga_config = GAConfig(
        pop_size=200,
        n_elite=30,
        n_lucky=10,
        n_generations=1000,
        mutation_rate=0.003,
        n_chains=32,
    )

    print(f"\nBenchmarking on {platform.upper()}...")
    print(f"Config: pop={ga_config.pop_size}, gens={ga_config.n_generations}, "
          f"mut={ga_config.mutation_rate}, chains={ga_config.n_chains}")

    ga = FastGA(
        amino_acid_seq=SPIKE_PROTEIN_SEQ,
        codon_table=codon_table,
        codon_scores=codon_scores,
        code_map=code_map,
        ga_config=ga_config,
        scoring_config=scoring_config,
    )

    # JIT warmup
    print("JIT compiling...")
    t0 = time.time()
    result = ga.run(jax.random.key(0))
    jax.block_until_ready(result["best_indices"])
    t_compile = time.time() - t0
    print(f"JIT compile: {t_compile:.2f}s")

    # Timed runs
    times = []
    scores = []
    for seed in range(5):
        t0 = time.time()
        result = ga.run(jax.random.key(seed + 1))
        jax.block_until_ready(result["best_indices"])
        dt = time.time() - t0
        times.append(dt)
        scores.append(result["best_score"]["total"])

    mean_time = np.mean(times)
    best_score = min(scores)
    mean_score = np.mean(scores)

    print(f"Mean run time: {mean_time:.3f}s (over 5 seeds)")
    print(f"Best score: {best_score:.1f}, Mean: {mean_score:.1f}")

    return mean_time, best_score, mean_score


def main():
    import jax

    L = len(SPIKE_PROTEIN_SEQ)
    max_codons = max(len(codon_table[aa]) for aa in SPIKE_PROTEIN_SEQ)

    print("=" * 80)
    print("ENERGY CONSUMPTION COMPARISON")
    print("GA (CPU/GPU) vs Potts Model (Thermodynamic Chip)")
    print("=" * 80)
    print(f"\nSequence: spike protein, L={L} amino acids, N={3*L} nucleotides")
    print(f"Max codons per position: {max_codons}")
    print(f"Target score: ~444-445")

    # ==========================================
    # FLOP Estimates
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 1: FLOP ESTIMATES")
    print("=" * 80)

    # GA config that matches Potts score
    ga_flops = estimate_ga_flops(
        pop_size=200, n_elite=30, n_lucky=10,
        n_generations=1000, mutation_rate=0.003,
        n_chains=32, L=L, max_codons=max_codons,
    )

    print(f"\n--- GA (pop=200, gens=1000, mut=0.003, chains=32) ---")
    print(f"Score achieved: ~444.5 (best), ~444.8 (mean)")
    print(f"Total evals: {200 * 1000 * 32:,}")
    print(f"FLOPs per generation per chain: {ga_flops['per_gen_per_chain']:,.0f}")
    print(f"Total FLOPs: {ga_flops['total']:,.0f} ({ga_flops['total']:.2e})")
    print(f"Breakdown:")
    for name, val in ga_flops["breakdown"].items():
        pct = 100 * val / ga_flops["total"]
        print(f"  {name:>12}: {val:>15,.0f} ({pct:5.1f}%)")

    # Potts model
    potts_flops = estimate_potts_flops(
        n_steps=10, n_chains=256, L=L, max_codons=max_codons,
    )

    print(f"\n--- Potts Model (steps=10, chains=256) ---")
    print(f"Score achieved: ~444 (best), ~445 (mean)")
    print(f"FLOPs per position: {potts_flops['per_position']}")
    print(f"FLOPs per sweep: {potts_flops['per_sweep']:,}")
    print(f"Total FLOPs: {potts_flops['total']:,.0f} ({potts_flops['total']:.2e})")

    print(f"\n--- FLOP Ratio ---")
    flop_ratio = ga_flops["total"] / potts_flops["total"]
    print(f"GA / Potts = {flop_ratio:.1f}x")
    print(f"(The GA requires {flop_ratio:.0f}x more FLOPs to achieve the same score)")

    # ==========================================
    # Idealized Energy Estimates
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 2: IDEALIZED ENERGY ESTIMATES (FLOP-based)")
    print("=" * 80)

    # Reference J/FLOP values for modern hardware
    # These are peak theoretical throughput values (generous to the CPU/GPU)
    hw_specs = {
        "CPU (Intel i9-13900K)":   {"tflops": 2.0,  "tdp_w": 253, "j_per_flop": None},
        "CPU (Apple M2 Max)":      {"tflops": 3.6,  "tdp_w": 22,  "j_per_flop": None},
        "CPU (AMD EPYC 9654)":     {"tflops": 5.0,  "tdp_w": 360, "j_per_flop": None},
        "GPU (NVIDIA RTX 4090)":   {"tflops": 82.6, "tdp_w": 450, "j_per_flop": None},
        "GPU (NVIDIA A100)":       {"tflops": 19.5, "tdp_w": 250, "j_per_flop": None},
        "GPU (NVIDIA H100 SXM)":   {"tflops": 67.0, "tdp_w": 700, "j_per_flop": None},
    }

    # Compute J/FLOP from TDP and peak TFLOPS
    for name, spec in hw_specs.items():
        spec["j_per_flop"] = spec["tdp_w"] / (spec["tflops"] * 1e12)

    print(f"\nHardware energy efficiency (at peak FP32 throughput):")
    print(f"{'Hardware':>30} {'TFLOPS':>8} {'TDP (W)':>8} {'pJ/FLOP':>10}")
    print("-" * 60)
    for name, spec in hw_specs.items():
        pj = spec["j_per_flop"] * 1e12
        print(f"{name:>30} {spec['tflops']:>8.1f} {spec['tdp_w']:>8} {pj:>10.1f}")

    print(f"\n{'':>30} {'GA Energy':>15} {'Potts Energy':>15} {'Ratio':>10}")
    print("-" * 75)

    # Potts on thermodynamic chip
    potts_chip_energy = 10 * 256 * 1e-10  # 10 steps × 256 chains × 1e-10 J/sweep
    print(f"{'Thermo chip (1e-10 J/sweep)':>30} {'N/A':>15} {potts_chip_energy:.2e} J {'':>10}")

    for name, spec in hw_specs.items():
        ga_energy = ga_flops["total"] * spec["j_per_flop"]
        potts_energy = potts_flops["total"] * spec["j_per_flop"]
        ratio_vs_chip = ga_energy / potts_chip_energy
        print(f"{name:>30} {ga_energy:>12.4f} J {potts_energy:>12.6f} J {ratio_vs_chip:>10.1e}")

    print(f"\n--- Key comparisons (GA on best hardware vs Potts on chip) ---")
    # Most favorable CPU for GA
    best_cpu = min(
        [(n, s) for n, s in hw_specs.items() if "CPU" in n],
        key=lambda x: ga_flops["total"] * x[1]["j_per_flop"]
    )
    best_gpu = min(
        [(n, s) for n, s in hw_specs.items() if "GPU" in n],
        key=lambda x: ga_flops["total"] * x[1]["j_per_flop"]
    )

    ga_cpu_energy = ga_flops["total"] * best_cpu[1]["j_per_flop"]
    ga_gpu_energy = ga_flops["total"] * best_gpu[1]["j_per_flop"]

    print(f"Best CPU for GA: {best_cpu[0]}")
    print(f"  GA energy: {ga_cpu_energy:.4f} J")
    print(f"  Potts chip energy: {potts_chip_energy:.2e} J")
    print(f"  Ratio: {ga_cpu_energy / potts_chip_energy:.2e}x")

    print(f"\nBest GPU for GA: {best_gpu[0]}")
    print(f"  GA energy: {ga_gpu_energy:.4f} J")
    print(f"  Potts chip energy: {potts_chip_energy:.2e} J")
    print(f"  Ratio: {ga_gpu_energy / potts_chip_energy:.2e}x")

    # Potts model on GPU (for fun)
    potts_gpu_energy = potts_flops["total"] * best_gpu[1]["j_per_flop"]
    print(f"\nPotts model on {best_gpu[0]}:")
    print(f"  Energy: {potts_gpu_energy:.6f} J ({potts_gpu_energy*1e3:.3f} mJ)")
    print(f"  vs chip: {potts_gpu_energy / potts_chip_energy:.2e}x")

    # ==========================================
    # Actual Benchmark
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 3: MEASURED BENCHMARK")
    print("=" * 80)

    platform = jax.default_backend()
    print(f"\nJAX backend: {platform}")
    print(f"Devices: {jax.devices()}")

    mean_time, best_score, mean_score = run_benchmark(platform)

    # Estimate actual energy from wall time
    if platform == "gpu":
        # Assume GPU + CPU system draws ~300W during compute
        system_power_w = 300
        label = "GPU system (~300W)"
    else:
        # Single CPU core: ~15W, multi-core: ~100W
        system_power_w = 100
        label = "CPU (~100W)"

    actual_energy = mean_time * system_power_w
    print(f"\nActual energy estimate ({label}):")
    print(f"  Wall time: {mean_time:.3f}s")
    print(f"  Energy: {actual_energy:.2f} J")
    print(f"  vs Potts chip: {actual_energy / potts_chip_energy:.2e}x")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
Problem: Codon optimization for spike protein (L={L})
Both methods achieve score ~444-445

Potts model (thermodynamic chip):
  - 10 Gibbs sweeps × 256 chains
  - Energy: {potts_chip_energy:.2e} J = {potts_chip_energy*1e9:.0f} nJ
  - Total FLOPs: {potts_flops['total']:.2e}

GA (optimized, pop=200, gens=1000, mut=0.003, 32 chains):
  - 6.4M fitness evaluations
  - Total FLOPs: {ga_flops['total']:.2e} ({flop_ratio:.0f}x more than Potts)

Energy comparison (to reach same score ~445):
  Method                        Energy          vs Chip
  ────────────────────────────  ──────────────  ────────
  Potts chip                    {potts_chip_energy*1e9:>8.0f} nJ       1x
  GA on best CPU (idealized)    {ga_cpu_energy:>8.4f} J   {ga_cpu_energy/potts_chip_energy:>10.1e}x
  GA on best GPU (idealized)    {ga_gpu_energy:>8.4f} J   {ga_gpu_energy/potts_chip_energy:>10.1e}x
  GA on this machine (measured) {actual_energy:>8.2f} J   {actual_energy/potts_chip_energy:>10.1e}x

The thermodynamic chip is ~{ga_cpu_energy/potts_chip_energy:.0e}x more energy efficient
than the best idealized CPU for this problem.
""")


if __name__ == "__main__":
    main()
