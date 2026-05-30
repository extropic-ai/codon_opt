#!/usr/bin/env python3
"""Main script for codon optimization using Potts model Gibbs sampling."""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from potts.model import run_annealing, run_sampling
from problem import (
    SPIKE_PROTEIN_SEQ,
    compute_energy,
    codons_to_nucleotides,
    create_metrics_fn,
    indices_to_codons,
)
from script_utils import (
    compute_energies_over_samples,
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
from thrml.block_sampling import SamplingSchedule


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Codon optimization using Potts model Gibbs sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Problem specification
    parser.add_argument(
        "--seq",
        type=str,
        default="MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI",
        help="Amino acid sequence (single-letter codes)",
    )
    parser.add_argument(
        "--weight_usage",
        type=float,
        default=0.1,
        help="Weight for codon usage term (c_f)",
    )
    parser.add_argument(
        "--weight_repeat",
        type=float,
        default=0.1,
        help="Weight for repeat penalty term (c_R)",
    )
    parser.add_argument(
        "--target_gc",
        type=float,
        default=0.5,
        help="Target GC fraction (0-1), omit to disable GC term",
    )
    parser.add_argument(
        "--weight_gc",
        type=float,
        default=1.0,
        help="Weight for GC content term (c_GC)",
    )
    parser.add_argument(
        "--codon_freqs",
        type=str,
        choices=["uniform", "ecoli"],
        default="ecoli",
        help="Codon frequency table: 'uniform' (equal within each AA) or 'ecoli' (E. coli K-12)",
    )

    # Sampling parameters
    parser.add_argument(
        "--n_chains",
        type=int,
        default=512,
        help="Number of parallel Gibbs chains",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples per chain (for annealing, this is n_betas)",
    )
    parser.add_argument(
        "--steps_per_sample",
        type=int,
        default=1,
        help="Gibbs sweeps between samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Annealing parameters
    parser.add_argument(
        "--anneal",
        action="store_true",
        help="Use simulated annealing instead of fixed-beta sampling",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Inverse temperature (fixed-beta mode only)",
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        default=0.3,
        help="Starting (minimum) beta for annealing schedule",
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        default=100.0,
        help="Ending (maximum) beta for annealing schedule",
    )

    # GC adaptation
    parser.add_argument(
        "--use_adaptive_gc_coeff",
        action="store_true",
        help="Use adaptive GC coefficient during annealing to match target GC fraction",
    )
    parser.add_argument(
        "--gc_coeff_adapt_mult",
        type=float,
        default=0.1,
        help="Multiplier for GC coefficient adaptation rate",
    )

    # Output
    parser.add_argument(
        "--run_name",
        type=str,
        default="codon_opt",
        help="Name for this run (used in output directory naming)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "output"),
        help="Base directory for output files (each run creates a subdirectory)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plotting",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup run directory and logging
    run_dir, log_file, timestamp = setup_run_directory(args)
    write_log(log_file, f"Run directory: {run_dir}\n\n")

    # Copy SLURM job file if available
    copy_job_file(run_dir)

    # Log hyperparameters
    write_log(log_file, "=" * 60 + "\n")
    write_log(log_file, "HYPERPARAMETERS\n")
    write_log(log_file, "=" * 60 + "\n")
    write_log(log_file, f"Run name: {args.run_name}\n")
    write_log(log_file, f"Amino acid sequence: {args.seq}\n")
    write_log(log_file, f"Codon frequencies: {args.codon_freqs}\n")
    write_log(log_file, f"Weights: usage={args.weight_usage}, repeat={args.weight_repeat}, gc={args.weight_gc}\n")
    write_log(log_file, f"Target GC fraction: {args.target_gc}\n" if args.target_gc is not None else "GC term disabled\n")
    write_log(log_file, f"Sampling: n_chains={args.n_chains}, n_samples={args.n_samples}, steps_per_sample={args.steps_per_sample}\n")
    write_log(log_file, f"Seed: {args.seed}\n")
    if args.anneal or args.use_adaptive_gc_coeff:
        write_log(log_file, f"Annealing: beta_min={args.beta_min}, beta_max={args.beta_max}\n")
        if args.use_adaptive_gc_coeff:
            write_log(log_file, f"Adaptive GC: gc_coeff_adapt_mult={args.gc_coeff_adapt_mult}\n")
    else:
        write_log(log_file, f"Fixed beta: {args.beta}\n")
    write_log(log_file, "=" * 60 + "\n\n")

    # Create problem
    if args.seq == "spike_protein":
        args.seq = SPIKE_PROTEIN_SEQ.replace(" ", "").upper()

    problem = create_problem_from_args(args)
    write_log(log_file, f"Problem: {problem.seq_length} amino acids, sequence: {problem.amino_acid_seq}\n")
    write_log(log_file, f"Codon frequencies: {args.codon_freqs}\n")
    write_log(log_file, f"Weights: usage={args.weight_usage}, repeat={args.weight_repeat}, gc={args.weight_gc}\n")
    if args.target_gc is not None:
        write_log(log_file, f"Target GC fraction: {args.target_gc}\n")
    else:
        write_log(log_file, "GC term disabled\n")

    key = jax.random.key(args.seed)

    # Determine if we should use annealing mode
    use_annealing = args.anneal or args.use_adaptive_gc_coeff

    # Create metrics function for computing energies inside the scan
    metrics_fn = create_metrics_fn(problem)

    # Run sampling (annealing or fixed-beta)
    if use_annealing:
        betas = jnp.logspace(jnp.log10(args.beta_min), jnp.log10(args.beta_max), args.n_samples)
        write_log(log_file, f"\nAnnealing: {args.n_chains} chains, {args.n_samples} beta values from {args.beta_min:.3f} to {args.beta_max:.3f}\n")
        write_log(log_file, f"  {args.steps_per_sample} Gibbs sweeps per beta\n")

        # Setup GC adaptation
        if args.use_adaptive_gc_coeff:
            if args.target_gc is None:
                raise ValueError("--use_adaptive_gc_coeff requires --target_gc to be set")
            gc_coeff_adapt_mult = jnp.array(args.gc_coeff_adapt_mult)
            write_log(log_file, f"  GC adaptation enabled: mult={args.gc_coeff_adapt_mult}, target={args.target_gc}\n")
        else:
            gc_coeff_adapt_mult = None

        # Use only_return_final=True and metrics to compute energies inside the scan
        final_states, stats, metrics_results = run_annealing(
            problem,
            key,
            n_chains=args.n_chains,
            steps_per_beta=args.steps_per_sample,
            betas=betas,
            gc_coeff_adapt_mult=gc_coeff_adapt_mult,
            only_return_final=True,
            metrics=metrics_fn,
        )

        # Extract energy trajectories from metrics results
        # metrics_results has shape (n_chains, n_betas) for each key
        total = np.array(metrics_results["total"])
        usage = np.array(metrics_results["usage"])
        gc = np.array(metrics_results["gc"])
        repeat = np.array(metrics_results["repeat"])

        final_states = np.array(final_states)
        write_log(log_file, f"Final states shape: {final_states.shape}\n")

        # Find best solution from final states (lowest energy in final column)
        final_energies = total[:, -1]
        best_chain = int(np.argmin(final_energies))
        best_indices = final_states[best_chain]
        best_codon_seq = indices_to_codons(problem, best_indices)
        best_energy, terms = compute_energy(problem, best_codon_seq, return_terms=True)

    else:
        schedule = SamplingSchedule(
            n_warmup=args.steps_per_sample,
            n_samples=args.n_samples,
            steps_per_sample=args.steps_per_sample,
        )
        write_log(log_file, f"\nFixed-beta sampling: {args.n_chains} chains, {args.n_samples} samples, "
              f"steps_per_sample={args.steps_per_sample}, beta={args.beta}\n")

        samples = run_sampling(
            problem,
            key,
            n_chains=args.n_chains,
            schedule=schedule,
            beta=args.beta,
        )
        stats = None

        samples = np.array(samples)
        write_log(log_file, f"Samples shape: {samples.shape}\n")

        # Compute energies outside the sampling loop for fixed-beta mode
        write_log(log_file, "Computing energies...\n")
        total, usage, gc, repeat = compute_energies_over_samples(problem, samples)

        # Find best solution (lowest energy across all chains and samples)
        best_idx = np.unravel_index(np.argmin(total), total.shape)
        best_chain, best_sample = best_idx
        best_indices = samples[best_chain, best_sample]
        best_codon_seq = indices_to_codons(problem, best_indices)
        best_energy, terms = compute_energy(problem, best_codon_seq, return_terms=True)

    # Print results
    print_results_summary(problem, best_codon_seq, best_energy, terms, log_file)

    # Compute qodon SeqScorer score for comparison
    best_nuc_seq = codons_to_nucleotides(best_codon_seq)
    qodon_score = compute_qodon_score(best_nuc_seq)
    write_log(log_file, f"\nqodon SeqScorer score: {qodon_score:.4f}\n")

    # Print convergence stats
    write_log(log_file, f"\nConvergence:\n")
    write_log(log_file, f"  Initial energy (mean): {total[:, 0].mean():.4f} +/- {total[:, 0].std():.4f}\n")
    write_log(log_file, f"  Final energy (mean):   {total[:, -1].mean():.4f} +/- {total[:, -1].std():.4f}\n")
    write_log(log_file, f"  Best energy: {best_energy:.4f}\n")

    # Save results to JSON
    save_results_json(
        Path(run_dir) / "results.json",
        problem,
        best_codon_seq,
        best_energy,
        terms,
        total,
        args,
    )

    # Plot results
    if not args.no_plot:
        plot_energy_convergence(
            total, usage, gc, repeat,
            Path(run_dir) / "energy_convergence.png"
        )
        plot_best_solution_stats(
            problem, best_codon_seq,
            Path(run_dir) / "solution_stats.png"
        )

    if stats is not None:
        stats_np = {k: np.array(v) for k, v in stats.items()}
        # Print GC coeffs spread at the end
        assert stats_np["gc_coeffs"].shape == (args.n_samples, args.n_chains)
        final_gc_coeffs = stats_np["gc_coeffs"][-1]
        p5 = np.percentile(final_gc_coeffs, 5)
        p95 = np.percentile(final_gc_coeffs, 95)
        std = np.std(final_gc_coeffs)
        mean = np.mean(final_gc_coeffs)
        write_log(log_file, f"\nFinal GC coeffs:"
                            f" mean={mean:.4f}, std={std:.4f},"
                            f" 5th pct={p5:.4f}, 95th pct={p95:.4f}\n")

        if not args.no_plot:
            plot_gc_adaptation_stats(
                stats_np,
                Path(run_dir) / "gc_adaptation.png"
            )

    write_log(log_file, f"\nRun completed. Results saved to {run_dir}\n")


if __name__ == "__main__":
    main()
