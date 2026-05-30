#!/usr/bin/env python3
"""Main script for codon optimization using Ising model with Domain Wall Encoding."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ising.dwc import compute_spin_layout, spin_to_potts
from ising.ising_model import run_ising_annealing
from problem import (
    SPIKE_PROTEIN_SEQ,
    compute_energy,
    compute_unary_biases,
    codons_to_nucleotides,
    create_metrics_fn,
    indices_to_codons,
)
from script_utils import (
    compute_qodon_score,
    copy_job_file,
    create_problem_from_args,
    plot_best_solution_stats,
    plot_energy_convergence,
    plot_gc_adaptation_stats,
    print_results_summary,
    save_results_json,
    setup_run_directory,
    write_log,
)


def parse_ising_args() -> argparse.Namespace:
    """Parse command line arguments for the Ising codon optimizer."""
    parser = argparse.ArgumentParser(
        description="Codon optimization using Ising model with Domain Wall Encoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem specification
    parser.add_argument("--seq", type=str,
        default="MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI",
        help="Amino acid sequence (single-letter codes)")
    parser.add_argument("--weight_usage", type=float, default=0.1, help="Weight for codon usage term")
    parser.add_argument("--weight_repeat", type=float, default=0.1, help="Weight for repeat penalty term")
    parser.add_argument("--target_gc", type=float, default=0.5, help="Target GC fraction (0-1)")
    parser.add_argument("--weight_gc", type=float, default=1.0, help="Weight for GC content term")
    parser.add_argument("--codon_freqs", type=str, choices=["uniform", "ecoli"], default="ecoli",
        help="Codon frequency table")

    # Sampling parameters
    parser.add_argument("--n_chains", type=int, default=512, help="Number of parallel Gibbs chains")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of annealing steps")
    parser.add_argument("--steps_per_sample", type=int, default=2000, help="Gibbs sweeps per annealing step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Annealing parameters
    parser.add_argument("--anneal", action="store_true",
        help="Accepted for compatibility; the Ising sampler always uses simulated annealing")
    parser.add_argument("--beta_min", type=float, default=2.0, help="Starting beta")
    parser.add_argument("--beta_max", type=float, default=200.0, help="Ending beta")
    parser.add_argument("--P_min", type=float, default=2.0, help="Starting constraint penalty")
    parser.add_argument("--P_max", type=float, default=200.0, help="Ending constraint penalty")
    parser.add_argument("--n_const_P_steps", type=int, default=5,
        help="Number of initial annealing steps where P stays at P_min before ramping")

    # GC adaptation
    parser.add_argument("--use_adaptive_gc_coeff", action="store_true",
        help="Use adaptive GC coefficient during annealing")
    parser.add_argument("--gc_coeff_adapt_mult", type=float, default=0.9,
        help="Multiplier for GC coefficient adaptation rate")

    # Output
    parser.add_argument("--run_name", type=str, default="ising_codon_opt", help="Name for this run")
    parser.add_argument("--output_dir", type=str,
        default=str(Path(__file__).parent.parent / "output"), help="Base directory for output")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting")

    return parser.parse_args()


def main():
    args = parse_ising_args()
    # The Ising sampler always uses simulated annealing (with simultaneous P-ramping),
    # so record that in the saved results regardless of the --anneal flag.
    args.anneal = True

    # Setup run directory
    run_dir, log_file, timestamp = setup_run_directory(args)
    write_log(log_file, f"Run directory: {run_dir}\n\n")
    copy_job_file(run_dir)

    # Log hyperparameters
    write_log(log_file, "=" * 60 + "\n")
    write_log(log_file, "HYPERPARAMETERS (Ising DWC)\n")
    write_log(log_file, "=" * 60 + "\n")
    write_log(log_file, f"Run name: {args.run_name}\n")
    write_log(log_file, f"Amino acid sequence: {args.seq}\n")
    write_log(log_file, f"Codon frequencies: {args.codon_freqs}\n")
    write_log(log_file, f"Weights: usage={args.weight_usage}, repeat={args.weight_repeat}, gc={args.weight_gc}\n")
    write_log(log_file, f"Target GC fraction: {args.target_gc}\n" if args.target_gc is not None else "GC term disabled\n")
    write_log(log_file, f"Sampling: n_chains={args.n_chains}, n_samples={args.n_samples}, steps_per_sample={args.steps_per_sample}\n")
    write_log(log_file, f"Seed: {args.seed}\n")
    write_log(log_file, f"Annealing: beta=[{args.beta_min}, {args.beta_max}], P=[{args.P_min}, {args.P_max}], n_const_P_steps={args.n_const_P_steps}\n")
    if args.use_adaptive_gc_coeff:
        write_log(log_file, f"Adaptive GC: gc_coeff_adapt_mult={args.gc_coeff_adapt_mult}\n")
    write_log(log_file, "=" * 60 + "\n\n")

    # Create problem
    if args.seq == "spike_protein":
        args.seq = SPIKE_PROTEIN_SEQ.replace(" ", "").upper()

    problem = create_problem_from_args(args)

    # Report spin count
    _, Ks = compute_unary_biases(problem)
    n_total_spins = sum(max(K - 1, 0) for K in Ks)
    write_log(log_file, f"Problem: {problem.seq_length} amino acids, {n_total_spins} Ising spins\n")
    write_log(log_file, f"K distribution: {dict(zip(*np.unique(Ks, return_counts=True)))}\n")

    key = jax.random.key(args.seed)

    # Schedules
    betas = jnp.logspace(jnp.log10(args.beta_min), jnp.log10(args.beta_max), args.n_samples)
    n_const = args.n_const_P_steps
    n_ramp = args.n_samples - n_const
    Ps_const = jnp.full(n_const, args.P_min)
    Ps_ramp = jnp.logspace(jnp.log10(args.P_min), jnp.log10(args.P_max), n_ramp)
    Ps = jnp.concatenate([Ps_const, Ps_ramp])

    write_log(log_file, f"\nAnnealing: {args.n_chains} chains, {args.n_samples} steps\n")
    write_log(log_file, f"  beta: {args.beta_min:.3f} -> {args.beta_max:.3f}\n")
    write_log(log_file, f"  P: {args.P_min:.3f} -> {args.P_max:.3f} (const for first {n_const} steps)\n")
    write_log(log_file, f"  {args.steps_per_sample} Gibbs sweeps per step\n")

    # GC adaptation
    if args.use_adaptive_gc_coeff:
        if args.target_gc is None:
            raise ValueError("--use_adaptive_gc_coeff requires --target_gc")
        gc_coeff_adapt_mult = jnp.array(args.gc_coeff_adapt_mult)
        write_log(log_file, f"  GC adaptation: mult={args.gc_coeff_adapt_mult}, target={args.target_gc}\n")
    else:
        gc_coeff_adapt_mult = None

    # Metrics function
    metrics_fn = create_metrics_fn(problem)

    # Run annealing
    final_states, stats, metrics_results = run_ising_annealing(
        problem,
        key,
        n_chains=args.n_chains,
        steps_per_beta=args.steps_per_sample,
        betas=betas,
        Ps=Ps,
        gc_coeff_adapt_mult=gc_coeff_adapt_mult,
        only_return_final=True,
        metrics=metrics_fn,
    )

    # final_states is spin state (n_chains, n_total_spins) bool
    # Convert to Potts indices for results
    pos_of_spin, spin_pos_index, pos_matrix, _ = compute_spin_layout(Ks)
    pos_matrix_jax = jnp.array(pos_matrix)

    total = np.array(metrics_results["total"])
    usage = np.array(metrics_results["usage"])
    gc = np.array(metrics_results["gc"])
    repeat = np.array(metrics_results["repeat"])

    # Find best solution from final states
    final_energies = total[:, -1]
    best_chain = int(np.argmin(final_energies))

    # Convert best chain's final spin state to codons
    final_spin = np.array(final_states)
    best_spin = jnp.array(final_spin[best_chain])
    best_potts = spin_to_potts(best_spin, pos_matrix_jax)
    best_indices = np.array(best_potts)
    best_codon_seq = indices_to_codons(problem, best_indices)
    best_energy, terms = compute_energy(problem, best_codon_seq, return_terms=True)

    write_log(log_file, f"Final states shape: {final_spin.shape}\n")

    # Print results
    print_results_summary(problem, best_codon_seq, best_energy, terms, log_file)

    # qodon score
    best_nuc_seq = codons_to_nucleotides(best_codon_seq)
    try:
        qodon_score = compute_qodon_score(best_nuc_seq)
        write_log(log_file, f"\nqodon SeqScorer score: {qodon_score:.4f}\n")
    except Exception:
        pass

    # Convergence stats
    write_log(log_file, f"\nConvergence:\n")
    write_log(log_file, f"  Initial energy (mean): {total[:, 0].mean():.4f} +/- {total[:, 0].std():.4f}\n")
    write_log(log_file, f"  Final energy (mean):   {total[:, -1].mean():.4f} +/- {total[:, -1].std():.4f}\n")
    write_log(log_file, f"  Best energy: {best_energy:.4f}\n")

    # Save results
    save_results_json(
        Path(run_dir) / "results.json",
        problem,
        best_codon_seq,
        best_energy,
        terms,
        total,
        args,
    )

    # Plot
    if not args.no_plot:
        plot_energy_convergence(total, usage, gc, repeat, Path(run_dir) / "energy_convergence.png")
        plot_best_solution_stats(problem, best_codon_seq, Path(run_dir) / "solution_stats.png")

    if stats is not None:
        stats_np = {k: np.array(v) for k, v in stats.items()}
        if "gc_coeffs" in stats_np:
            final_gc_coeffs = stats_np["gc_coeffs"][-1]
            p5 = np.percentile(final_gc_coeffs, 5)
            p95 = np.percentile(final_gc_coeffs, 95)
            std = np.std(final_gc_coeffs)
            mean = np.mean(final_gc_coeffs)
            write_log(log_file, f"\nFinal GC coeffs: mean={mean:.4f}, std={std:.4f}, "
                                f"5th pct={p5:.4f}, 95th pct={p95:.4f}\n")
            if not args.no_plot:
                plot_gc_adaptation_stats(stats_np, Path(run_dir) / "gc_adaptation.png")

    write_log(log_file, f"\nRun completed. Results saved to {run_dir}\n")


if __name__ == "__main__":
    main()
