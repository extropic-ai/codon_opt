"""Utilities for the main codon optimization scripts (Potts and Ising)."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from problem import (
    SPIKE_PROTEIN_SEQ,
    CodonProblem,
    codons_to_nucleotides,
    create_energy_fn,
    create_metrics_fn,
    gc_count,
    get_default_rarity_scores,
    get_ecoli_codon_rarity_scores,
)


def write_log(log_path: Optional[str], text: str):
    """Write text to both log file and stdout.

    Args:
        log_path: Path to log file, or None to only print to stdout.
        text: Text to write.
    """
    if log_path is not None:
        with open(log_path, "a") as log:
            log.write(text)
    print(text, end="", flush=True)


def setup_run_directory(args) -> tuple[str, str, str]:
    """Create run directory with timestamp and SLURM job ID.

    Args:
        args: Command line arguments.

    Returns:
        Tuple of (run_dir, log_file, timestamp).
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if slurm_job_id:
        timestamp += f"_sj{slurm_job_id}"

    run_name = f"{args.run_name}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)

    os.makedirs(run_dir, exist_ok=False)
    log_file = os.path.join(run_dir, f"run_log_{timestamp}.txt")

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_log(log_file, f"Time: {time_now}\n")

    return run_dir, log_file, timestamp


def copy_job_file(save_dir: str) -> None:
    """Copy SLURM job file to run directory for reproducibility.

    Args:
        save_dir: Directory to save the job file copy.
    """
    jobfile = os.getenv("JOB_FILE")

    if jobfile and os.path.exists(jobfile):
        try:
            dest_path = os.path.join(save_dir, "jobfile_copy.job")
            shutil.copy(jobfile, dest_path)
            print(f"Successfully copied job file to {dest_path}", flush=True)
        except Exception as e:
            print(f"Failed to copy job file: {e}", flush=True)


def save_results_json(
    output_path: Path,
    problem: CodonProblem,
    best_codon_seq: list[str],
    best_energy: float,
    terms: dict[str, float],
    total_energies: np.ndarray,
    args: argparse.Namespace,
):
    """Save main results to JSON file.

    Args:
        output_path: Path to save JSON file.
        problem: The codon problem.
        best_codon_seq: Best codon sequence found.
        best_energy: Energy of best solution.
        terms: Energy terms breakdown.
        total_energies: Total energies for all samples.
        args: Command line arguments.
    """
    nuc_seq = codons_to_nucleotides(best_codon_seq)
    total_gc = sum(gc_count(c) for c in best_codon_seq)
    gc_frac = total_gc / len(nuc_seq)

    results = {
        "timestamp": datetime.now().isoformat(),
        "problem": {
            "amino_acid_seq": problem.amino_acid_seq,
            "seq_length": problem.seq_length,
            "target_gc_fraction": problem.target_gc_fraction,
            "weight_codon_usage": problem.weight_codon_usage,
            "weight_gc": problem.weight_gc,
            "weight_repeat": problem.weight_repeat,
            "codon_freqs": args.codon_freqs,
        },
        "sampling": {
            "n_chains": args.n_chains,
            "n_samples": args.n_samples,
            "steps_per_sample": args.steps_per_sample,
            "seed": args.seed,
            "anneal": args.anneal,
            "beta": args.beta if not args.anneal else None,
            "beta_min": args.beta_min if args.anneal else None,
            "beta_max": args.beta_max if args.anneal else None,
            "use_adaptive_gc_coeff": args.use_adaptive_gc_coeff,
            "gc_coeff_adapt_mult": args.gc_coeff_adapt_mult if args.use_adaptive_gc_coeff else None,
        },
        "best_solution": {
            "codon_seq": best_codon_seq,
            "nucleotide_seq": nuc_seq,
            "gc_content": float(gc_frac),
            "total_energy": float(best_energy),
            "energy_terms": {k: float(v) for k, v in terms.items()},
        },
        "convergence": {
            "initial_energy_mean": float(total_energies[:, 0].mean()),
            "initial_energy_std": float(total_energies[:, 0].std()),
            "final_energy_mean": float(total_energies[:, -1].mean()),
            "final_energy_std": float(total_energies[:, -1].std()),
            "best_energy": float(best_energy),
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {output_path}")


def create_problem_from_args(args: argparse.Namespace) -> CodonProblem:
    """Create a CodonProblem instance from command line arguments."""
    if args.codon_freqs == "ecoli":
        rarity_scores = get_ecoli_codon_rarity_scores()
    else:
        rarity_scores = get_default_rarity_scores()

    return CodonProblem(
        amino_acid_seq=args.seq,
        codon_rarity_scores=rarity_scores,
        target_gc_fraction=args.target_gc,
        weight_codon_usage=args.weight_usage,
        weight_gc=args.weight_gc,
        weight_repeat=args.weight_repeat,
    )


def compute_energies_over_samples(
    problem: CodonProblem, samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute energy terms for all samples.

    Args:
        problem: The codon problem.
        samples: Array of shape (n_chains, n_samples, L).

    Returns:
        Tuple of (total, usage, gc, repeat) arrays, each shape (n_chains, n_samples).
    """
    energy_fn = create_energy_fn(problem)
    results = energy_fn(samples)

    return (
        np.asarray(results["total"]),
        np.asarray(results["usage"]),
        np.asarray(results["gc"]),
        np.asarray(results["repeat"]),
    )


def plot_energy_convergence(
    total: np.ndarray,
    usage: np.ndarray,
    gc: np.ndarray,
    repeat: np.ndarray,
    output_path: Path,
):
    """Plot energy convergence over sampling iterations.

    Args:
        total, usage, gc, repeat: Arrays of shape (n_chains, n_samples).
        output_path: Path to save the figure.
    """
    n_samples = total.shape[1]
    x = np.arange(n_samples)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    def plot_term(ax, data, title, color):
        mean = data.mean(axis=0)
        p5 = np.percentile(data, 5, axis=0)
        p95 = np.percentile(data, 95, axis=0)
        ax.plot(x, mean, color=color, linewidth=2, label="Mean")
        ax.fill_between(x, p5, p95, color=color, alpha=0.3, label="5th-95th percentile")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Energy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_term(axes[0, 0], total, "Total Energy", "black")
    plot_term(axes[0, 1], usage, "Codon Usage Term", "blue")
    plot_term(axes[1, 0], gc, "GC Content Term", "green")
    plot_term(axes[1, 1], repeat, "Repeat Penalty Term", "red")

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def plot_gc_adaptation_stats(
    stats: dict[str, np.ndarray],
    output_path: Path,
):
    """Plot GC adaptation statistics over the annealing process.

    Args:
        stats: Dict with 'gc_fractions' (n_betas, n_chains) and optionally
               'gc_coeffs' (n_betas, n_chains).
        output_path: Path to save the figure.
    """
    gc_fractions = stats["gc_fractions"]
    gc_coeffs = stats.get("gc_coeffs", None)
    n_betas = gc_fractions.shape[0]
    x = np.arange(n_betas)

    n_plots = 2 if gc_coeffs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    def plot_stat(ax, data, title, ylabel, color):
        mean = data.mean(axis=1)
        p5 = np.percentile(data, 5, axis=1)
        p95 = np.percentile(data, 95, axis=1)
        ax.plot(x, mean, color=color, linewidth=2, label="Mean")
        ax.fill_between(x, p5, p95, color=color, alpha=0.3, label="5th-95th percentile")
        ax.set_xlabel("Beta step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_stat(axes[0], gc_fractions, "GC Fraction", "GC fraction", "green")

    if gc_coeffs is not None:
        plot_stat(axes[1], gc_coeffs, "Linear GC Coefficient", "Coefficient", "purple")

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved GC adaptation plot to {output_path}")


def plot_best_solution_stats(
    problem: CodonProblem, best_codon_seq: list[str], output_path: Path
):
    """Plot statistics about the best solution found.

    Args:
        problem: The codon problem.
        best_codon_seq: The best codon sequence.
        output_path: Path to save the figure.
    """
    L = len(best_codon_seq)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # GC content per position
    gc_per_pos = [gc_count(c) / 3.0 for c in best_codon_seq]
    axes[0].bar(range(L), gc_per_pos, color="green", alpha=0.7)
    axes[0].axhline(y=np.mean(gc_per_pos), color="red", linestyle="--", label=f"Mean: {np.mean(gc_per_pos):.2f}")
    if problem.target_gc_fraction is not None:
        axes[0].axhline(
            y=problem.target_gc_fraction, color="blue", linestyle=":", label=f"Target: {problem.target_gc_fraction:.2f}"
        )
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("GC fraction")
    axes[0].set_title("GC Content per Position")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Codon rarity scores (lower = more common = better)
    rarity_scores = [problem.codon_rarity_scores.get(c, 0) for c in best_codon_seq]
    axes[1].bar(range(L), rarity_scores, color="blue", alpha=0.7)
    axes[1].axhline(
        y=np.mean(rarity_scores), color="red", linestyle="--", label=f"Mean: {np.mean(rarity_scores):.3f}"
    )
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Rarity score (lower = better)")
    axes[1].set_title("Codon Rarity Score per Position")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved solution stats to {output_path}")


def print_results_summary(
    problem: CodonProblem,
    best_codon_seq: list[str],
    best_energy: float,
    terms: dict[str, float],
    log_file: Optional[str] = None,
):
    """Print a summary of the optimization results."""
    nuc_seq = codons_to_nucleotides(best_codon_seq)
    total_gc = sum(gc_count(c) for c in best_codon_seq)
    gc_frac = total_gc / len(nuc_seq)

    write_log(log_file, "\n" + "=" * 60 + "\n")
    write_log(log_file, "CODON OPTIMIZATION RESULTS\n")
    write_log(log_file, "=" * 60 + "\n")
    write_log(log_file, f"Amino acid sequence: {problem.amino_acid_seq}\n")
    write_log(log_file, f"Sequence length: {problem.seq_length} amino acids\n")
    write_log(log_file, "-" * 60 + "\n")
    write_log(log_file, "Best solution found:\n")
    write_log(log_file, f"  Codons: {' '.join(best_codon_seq)}\n")
    write_log(log_file, f"  Nucleotides: {nuc_seq}\n")
    write_log(log_file, f"  GC content: {gc_frac:.3f} ({total_gc}/{len(nuc_seq)})\n")
    write_log(log_file, "-" * 60 + "\n")
    write_log(log_file, "Energy terms:\n")
    write_log(log_file, f"  Total:        {best_energy:.4f}\n")
    write_log(log_file, f"  Codon usage:  {terms['usage']:.4f}\n")
    write_log(log_file, f"  GC content:   {terms['gc']:.4f}\n")
    write_log(log_file, f"  Repeat:       {terms['repeat']:.4f}\n")
    write_log(log_file, "=" * 60 + "\n")


def compute_qodon_score(nucleotide_seq: str) -> float:
    """Compute the SeqScorer score from qodon for a nucleotide sequence.

    Requires the qodon package to be available (optional dependency).

    Args:
        nucleotide_seq: The nucleotide sequence (DNA, using T not U).

    Returns:
        The qodon SeqScorer score.

    Raises:
        ImportError: If qodon is not available.
    """
    # Try importing from installed package first, then fall back to path-based import
    try:
        from scoring import SeqScorer
    except ImportError:
        qodon_path = Path(__file__).parent / "qodon"
        if str(qodon_path) not in sys.path:
            sys.path.insert(0, str(qodon_path))
        from scoring import SeqScorer

    return SeqScorer(nucleotide_seq).score
