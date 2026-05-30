# Energy-Efficient Codon Optimization on Thermodynamic Hardware

This repository is the companion code for the paper **"Energy-Efficient Codon
Optimization on Thermodynamic Hardware"** (A. Jelinčič and R. C. Walker, Extropic).
The manuscript is included in [`paper/`](paper/) (`paper/main.pdf` and
`paper/supplement.pdf`); it is being released on arXiv and has been submitted to a
journal.

The paper formulates mRNA **codon optimization** as sampling from a probabilistic
graphical model and maps it onto a **Thermodynamic Sampling Unit (TSU)** — a
sub-threshold analog CMOS Ising machine that performs single-site Gibbs sampling at
extremely high speed and low energy. It benchmarks three approaches on the SARS-CoV-2
spike protein and finds they reach comparable solution quality, while energy estimates
grounded in prototype hardware measurements indicate a TSU could solve the problem
using ~5–9 orders of magnitude less energy than a GPU.

This code provides those three approaches so that readers can reproduce, inspect, and
experiment with them:

1. **Potts model** sampler — direct categorical formulation ([`potts/`](potts/)).
2. **Ising model** sampler — Potts compiled to binary spins via **domain-wall
   encoding** for binary (p-bit) hardware ([`ising/`](ising/)).
3. **Genetic algorithm** baseline — the reference implementation from Fox et al.
   plus a tuned JAX reimplementation ([`qodon/`](qodon/)).

The Potts and Ising samplers are built on the open-source
[**THRML**](https://github.com/extropic-ai/thrml) library, which provides the
block-Gibbs sampling primitives. The current JAX implementation *simulates* what a TSU
would do, validating the algorithmic approach before hardware deployment.

---

## Installation

The code is a **flat-layout research repo**: there is no installable package, and the
scripts are run as modules from the repository root (the directory containing this
README). The one non-PyPI dependency is THRML, which is installed directly from GitHub.

### Option A — uv (recommended)

[`uv`](https://docs.astral.sh/uv/) reads `pyproject.toml`, creates a local `.venv`, and
installs everything (including THRML from GitHub):

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
uv sync                 # runtime dependencies
uv sync --extra dev     # also install pytest (for the test suite)

# Run anything with `uv run` (no manual venv activation needed):
uv run python -m potts.main --help
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest        # only needed to run the tests
```

### Notes

- **JAX backend.** The dependencies install the CPU build of JAX, which is sufficient
  to run everything here. For GPU acceleration, follow the
  [JAX install guide](https://docs.jax.dev/en/latest/installation.html) to replace
  `jax`/`jaxlib` with the CUDA wheel matching your system.
- **Headless plotting.** The scripts save PNG figures and also call `plt.show()`. On a
  machine without a display, set `MPLBACKEND=Agg` (or pass `--no_plot`) to avoid
  blocking, e.g. `MPLBACKEND=Agg uv run python -m potts.main ...`.

---

## Quickstart

Run from the repository root. (Drop the `uv run` prefix if you activated a venv via pip.)

```bash
# Potts model: simulated annealing with adaptive GC (recommended approach)
MPLBACKEND=Agg uv run python -m potts.main \
  --seq "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI" \
  --anneal --beta_min 0.3 --beta_max 100 \
  --n_chains 256 --n_samples 100 --steps_per_sample 2 \
  --use_adaptive_gc_coeff --gc_coeff_adapt_mult 0.1 \
  --target_gc 0.5 --weight_usage 0.1 --weight_gc 2e4 --weight_repeat 0.2 \
  --codon_freqs ecoli --seed 42

# Full spike protein (1273 AA) — the paper's benchmark sequence
MPLBACKEND=Agg uv run python -m potts.main --seq spike_protein --anneal --use_adaptive_gc_coeff
```

See [Usage](#usage) below for the Ising entry point and the full argument list, and the
[`qodon`](#the-qodon-folder-genetic-algorithm-baseline) section for the GA baseline.

---

## Motivation

The long-term target is deployment on a **Thermodynamic Sampling Unit (TSU)**: a
sub-threshold analog CMOS-based Ising machine that performs single-site Gibbs sampling
at extremely high speed and low energy. The goal is an early demonstration of how
thermodynamic computers can be useful for pharmaceutical applications.

The TSU's operational model has three phases: (1) **flash weights** from host to chip,
(2) **Gibbs sampling** on the chip with fixed weights, and (3) **read state** back to
host. The algorithm is designed around this: an outer loop on the host (compute
weights, flash, read back, adapt parameters) wrapping an inner loop on the TSU (many
Gibbs steps with fixed weights). The current JAX implementation simulates what the TSU
would do, validating that the algorithmic approach works before hardware deployment.

---

## Problem Specification

### Goal
Given a protein sequence of length L, choose a synonymous codon at each amino-acid position to produce an mRNA sequence that is predicted to express well in a chosen host organism. This is a discrete combinatorial optimization problem: each position p has a small categorical choice set (the synonymous codons for that amino acid, at most 6), and the objective is a weighted sum of sequence-level and local penalties.

### Inputs

1. **Protein sequence**: amino acids a_1, ..., a_L.
2. **Synonymous codon sets**: for each position p, a set S_p of codons encoding amino acid a_p. Each codon is a length-3 string over {A,C,G,T}.
3. **Host codon-usage frequencies**: for each codon c, a frequency f_host(c) describing how common that codon is in the host (e.g. E. coli K-12).
4. **Target GC fraction**: rho_T in [0,1], desired fraction of nucleotides that are G or C.
5. **Weights**: c_f >= 0 (codon usage), c_GC >= 0 (GC content), c_R >= 0 (repeat penalty).

### Energy Function

The optimization target is:

```
E(c_1,...,c_L) = c_f * sum_p u(c_p)
              + c_GC * ( (1/3L) * sum_p g(c_p) - rho_T )^2
              + c_R * sum_{p=1}^{L-1} r(c_p, c_{p+1})
```

where:
- **Codon usage cost** (unary): `u(c) = |log(f_host(c) / f_max_aa)|`, which is 0 for the most common codon of each amino acid and positive for rarer codons.
- **GC count**: `g(c)` = number of G or C nucleotides in codon c (0 to 3).
- **Repeat penalty** (pairwise, adjacent positions only): `r(c, c') = m(c||c')^2 - 1`, where m is the length of the longest contiguous run of identical nucleotides in the 6-character concatenation. Zero when no nucleotide repeats across the boundary.

**Structure of the terms:**
- The codon-usage term is **unary**: depends only on each codon independently.
- The repeat-penalty term is **local pairwise**: depends on adjacent positions only.
- The GC term is **global**: couples all positions through the squared deviation of the overall GC fraction from the target.

The output is the codon sequence minimizing E, and the resulting nucleotide string.

This formulation and its default weights follow Fox et al. (see the
[`qodon`](#the-qodon-folder-genetic-algorithm-baseline) section).

---

## Solution Methods

We implement two sampling approaches. Both use simulated annealing with parallel Gibbs
chains and an adaptive GC coefficient. (The genetic-algorithm baseline lives in
[`qodon/`](qodon/).)

### Approach 1: Direct Potts Model

The problem is naturally a **Potts model** (probabilistic graphical model with categorical variables). Each position p has a categorical variable X_p with |S_p| states (up to 6). All variables are padded to K_max=6 states, with invalid states receiving a very large negative bias (effectively -infinity).

The Potts energy decomposes into unary biases h_p(k) (codon usage + validity) and pairwise interactions J_p(k,k') (repeat penalty between adjacent positions). This gives a **chain graph** (each position interacts only with its neighbors), which is **2-colorable** (even/odd positions).

**Block Gibbs sampling.** The 2-coloring enables parallel block updates: conditioned on odd positions, all even positions are conditionally independent and can be sampled simultaneously (via softmax + categorical sampling), and vice versa. One full Gibbs sweep = update even block, then odd block.

**Simulated annealing.** Starting from high temperature (low beta), we gradually increase beta on a log-spaced schedule. At each temperature step, we run several Gibbs sweeps, then update the temperature. Multiple chains run in parallel via `jax.vmap`.

**Adaptive GC coefficient.** The global GC term would require all-to-all pairwise interactions if expanded directly. Instead, we approximate it with an adaptive linear term: a per-chain coefficient lambda that adjusts the bias on each codon proportionally to its GC count. At each annealing step, lambda is updated based on the current GC fraction error (a Lagrangian-style update). The coefficient is clamped to never exceed the gradient of the true quadratic GC penalty, preventing the approximation from dominating other terms.

### Approach 2: Ising Model via Domain Wall Encoding

For deployment on Ising hardware (spins in {-1,+1}), we compile the Potts model into an Ising model using **domain wall encoding (DWC)**. A K-state Potts variable is represented by K-1 binary spins in a "thermometer" pattern: state k means the first k spins are +1 and the rest are -1. The "domain wall" between +1s and -1s sits at position k.

**Constraint enforcement.** Invalid spin configurations (with a -1 followed by a +1) are penalized by ferromagnetic nearest-neighbor couplings of strength P/4 within each position's spin chain. The constraint graph is a chain (K-2 edges), unlike one-hot encoding which requires a K-clique (K(K-1)/2 edges).

**Ising weight compilation.** Potts unary biases become Ising local fields via first differences: `b_i = (h_i - h_{i-1})/2` plus boundary corrections. Potts pairwise interactions become inter-position couplings via mixed second differences: `J_{(p,i),(q,j)} = (V_{ij} - V_{i-1,j} - V_{i,j-1} + V_{i-1,j-1}) / 4`. This compilation is derived in `ising/potts_to_ising.md` and `paper/supplement.tex`.

**GC-sorted codon ordering.** Codons for each amino acid are sorted by `(gc_count, alphabetical)`. Since single spin flips move the Potts variable to an adjacent state (k -> k+1), this ordering makes each spin flip change GC content by at most 1, creating a smooth landscape for the domain wall dynamics.

**4-color block decomposition.** The Ising graph has two types of edges: intra-position constraint chains and inter-position couplings (between adjacent amino acid positions). A 4-coloring based on `(position_parity, spin_index_parity)` ensures no two same-color spins share an edge, enabling parallel block Gibbs updates.

**P-ramping.** Both beta (inverse temperature) and P (constraint penalty) are annealed on independent log-spaced schedules. Low P early in annealing allows invalid-state (defect) transitions that provide shortcuts around energy barriers in the bias landscape. High P at the end enforces strict validity.

**Key advantage of DWC:** mixing time through the constraint chain is P-independent. At the domain wall boundary, the ferromagnetic couplings from the two neighbors cancel, so the Gibbs conditional depends only on the bias landscape, not on P.

---

## Code Structure

```
codon_opt/                # repository root — run scripts from here
  problem.py              # Problem definition, energy functions, Potts weight computation
  script_utils.py         # CLI utilities: run directory setup, plotting, result I/O
  pyproject.toml          # uv / PEP 621 project metadata and dependencies
  requirements.txt        # pip dependency list (mirror of pyproject)
  README.md               # This file
  potts/
    model.py              # Potts model construction and sampling
    main.py               # CLI entry point for Potts sampling
  ising/
    dwc.py                # Domain Wall Encoding: weight compilation, state conversion
    ising_model.py        # Ising model construction and sampling
    ising_main.py         # CLI entry point for Ising sampling
    potts_to_ising.md     # Reference: Potts-to-Ising compilation derivations + literature review
  qodon/                  # Genetic-algorithm baseline (Fox et al.) + our JAX reimplementation
  tests/                  # Codon-specific tests (pytest)
  paper/                  # LaTeX source + compiled PDF of the companion paper
```

> **Note on imports.** Modules use flat, top-level imports (`from problem import ...`,
> `from potts.model import ...`), so commands must be run from the repository root
> (e.g. `python -m potts.main`). The `qodon/` scripts similarly expect to be run from
> within `qodon/` (see that section).

### `problem.py` — Problem Definition

Framework-agnostic module shared by both Potts and Ising models. Contains:

- `CODON_TABLE` / `SORTED_CODON_TABLE`: standard genetic code and GC-sorted codon ordering.
- `CodonProblem`: Equinox module holding the protein sequence, rarity scores, target GC, and weights.
- `compute_energy(problem, codon_seq)`: reference energy evaluation from codon strings.
- `compute_unary_biases(problem, linear_gc_coeff)`: Potts bias array `[L, K_max]` in GC-sorted ordering.
- `compute_pairwise_penalties(problem)`: pairwise repeat penalty array `[L-1, K_max, K_max]`.
- `create_metrics_fn(problem)` / `create_energy_fn(problem)`: JIT-compiled energy evaluation from codon indices (single or batched via vmap).
- `create_gc_counter(problem)`: JIT-compiled GC count from codon indices.
- `adapt_gc_coeff(...)`: Lagrangian-style adaptive GC coefficient update with clamping.
- `indices_to_codons(problem, indices)` / `codons_to_nucleotides(codon_seq)`: index-to-string conversion.
- `get_ecoli_codon_rarity_scores()` / `get_default_rarity_scores()`: host-specific codon frequency loading via `python_codon_tables`.
- `SPIKE_PROTEIN_SEQ`: SARS-CoV-2 spike protein sequence (1273 amino acids), used as the default benchmark.

### `script_utils.py` — CLI Utilities

Shared utilities for both entry points:

- `create_problem_from_args(args)`: construct `CodonProblem` from CLI arguments.
- `setup_run_directory(args)`: create timestamped output directory (with SLURM job ID if available).
- `compute_energies_over_samples(problem, samples)`: batch energy evaluation over `[n_chains, n_samples, L]` arrays.
- `save_results_json(...)`: save results, hyperparameters, and convergence stats to JSON.
- `plot_energy_convergence(...)`: 4-panel plot of total/usage/gc/repeat energy over sampling iterations.
- `plot_gc_adaptation_stats(...)`: GC fraction and coefficient trajectories over annealing.
- `plot_best_solution_stats(...)`: per-position GC content and rarity scores for the best solution.
- `print_results_summary(...)`: formatted text summary of the best solution.
- `compute_qodon_score(nucleotide_seq)`: optional comparison against the qodon `SeqScorer`.

### `potts/model.py` — Potts Model

- `create_base_codon_model(problem, linear_gc_coeff)`: builds a `FactorSamplingProgram` with `CategoricalEBMFactor`s for biases and pairwise penalties, 2-color block decomposition (even/odd), and `CategoricalGibbsConditional` samplers.
- `scale_model_by_beta(program, beta)`: scale all interaction weights by inverse temperature.
- `run_sampling(problem, key, n_chains, schedule, beta)`: fixed-temperature Gibbs sampling with vmap over chains.
- `run_annealing(problem, key, n_chains, steps_per_beta, betas, gc_coeff_adapt_mult, ...)`: simulated annealing via `jax.lax.scan` over the beta schedule. At each step: adapt GC coefficient, scale weights by beta, run Gibbs sweeps, record state. Supports per-chain GC coefficients, optional metrics computation, and trajectory or final-state-only output.
- `update_program_with_gc_coeff(program, new_gc_coeff_times_beta, editable_params)`: efficiently update program weights with the GC linear term using a precomputed diff.

### `potts/main.py` — Potts CLI Entry Point

Supports two modes:
- **Fixed-beta sampling** (default): run Gibbs sampling at a fixed inverse temperature.
- **Simulated annealing** (`--anneal`): log-spaced beta schedule from `beta_min` to `beta_max`.

With optional `--use_adaptive_gc_coeff` for GC adaptation in either mode.

### `ising/dwc.py` — Domain Wall Encoding

- `compute_spin_layout(Ks)`: precompute index arrays mapping between flat spin indices and (position, within-position index).
- `compute_ising_biases(potts_biases, potts_pairwise, Ks, P)`: compile Potts weights to Ising bias vector. JIT-compatible (Python control flow depends only on static `Ks`).
- `compute_ising_couplings(potts_pairwise, Ks, P)`: compute constraint edges (intra-position, weight P/4) and inter-position edges (weight = mixed second difference / 4).
- `potts_to_spin(potts_indices, pos_of_spin, spin_pos_index)`: convert Potts indices to thermometer spin states.
- `spin_to_potts(spin_bool, pos_matrix)`: convert spin states back to Potts indices via matrix multiply.

### `ising/ising_model.py` — Ising Model

- `prepare_ising_statics(problem)`: one-time preparation of all static structures (spin layout, node objects, edge topology, 4-color block decomposition, precomputed vectorized arrays for fast weight computation inside JIT).
- `build_ising_program(statics, beta, P, gc_coeff)`: JIT-compatible construction of a `FactorSamplingProgram` from dynamic parameters using vectorized operations. Creates `SpinEBMFactor`s for biases, constraint edges, and inter-position couplings.
- `run_ising_annealing(problem, key, n_chains, steps_per_beta, betas, Ps, gc_coeff_adapt_mult, ...)`: simulated annealing via `jax.lax.scan` with separate beta and P schedules. At each step: convert spin->Potts for GC computation, adapt GC coefficient, build Ising program with current (beta, P, gc_coeff), run Gibbs sweeps, convert back. Per-chain vmap over `build_ising_program` + `sample_states`.
- `create_ising_codon_model(problem, P, linear_gc_coeff)`: convenience wrapper for one-shot model creation.

### `ising/potts_to_ising.md` — Reference Document

Comprehensive literature review and implementation plan covering: encoding schemes (one-hot vs. domain wall vs. binary), penalty strength and mixing time analysis, penalty scheduling, codon optimization on quantum/Ising hardware, the GC content term, DWC implementation details (codon ordering, P-ramping, mixing barriers), and the TSU target hardware.

### `tests/`

- `test_potts_model.py`: tests for problem utilities (GC count, repeat penalty, sorted ordering), unary biases, pairwise penalties, model creation, energy computation, annealing output shapes and convergence, GC adaptation behavior, and linear GC coefficient differentiability.
- `test_ising_model.py`: tests for codon ordering, Potts<->spin state conversion roundtrips (including batched), energy equivalence between Potts and Ising formulations (exhaustive over all valid states), 4-color block decomposition validity, and constraint satisfaction rate under sampling.

---

## Usage

Both sampling entry points are run as Python modules **from the repository root**.
Output goes to a timestamped subdirectory under `--output_dir` (default: `./output/`).

### Potts Model

```bash
# Simulated annealing with adaptive GC (recommended)
uv run python -m potts.main \
  --seq "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI" \
  --anneal --beta_min 0.3 --beta_max 100 \
  --n_chains 256 --n_samples 100 --steps_per_sample 2 \
  --use_adaptive_gc_coeff --gc_coeff_adapt_mult 0.1 \
  --target_gc 0.5 --weight_usage 0.1 --weight_gc 2e4 --weight_repeat 0.2 \
  --codon_freqs ecoli --seed 42

# Fixed-beta sampling (simpler, no annealing)
uv run python -m potts.main --beta 1.0 --n_samples 100 --steps_per_sample 2

# Full spike protein
uv run python -m potts.main --seq spike_protein --anneal --use_adaptive_gc_coeff
```

### Ising Model

The Ising sampler always uses simulated annealing with simultaneous P-ramping.

```bash
uv run python -m ising.ising_main \
  --seq "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNI" \
  --beta_min 0.1 --beta_max 20 --P_min 1.0 --P_max 100 \
  --n_chains 256 --n_samples 1000 --steps_per_sample 10 \
  --use_adaptive_gc_coeff --gc_coeff_adapt_mult 0.1 \
  --target_gc 0.5 --weight_usage 0.1 --weight_gc 2e4 --weight_repeat 0.2 \
  --codon_freqs ecoli --seed 42
```

### Common CLI Arguments

| Argument | Description |
|----------|-------------|
| `--seq` | Amino acid sequence (`spike_protein` for full 1273-AA) |
| `--codon_freqs` | Codon frequency table: `ecoli` (E. coli K-12) or `uniform` |
| `--target_gc` | Target GC fraction |
| `--weight_usage` | Codon usage term weight (c_f) |
| `--weight_gc` | GC content term weight (c_GC) |
| `--weight_repeat` | Repeat penalty term weight (c_R) |
| `--n_chains` | Number of parallel Gibbs chains |
| `--n_samples` | Number of samples (Potts) or annealing steps (Ising) |
| `--steps_per_sample` | Gibbs sweeps per sample or annealing step |
| `--seed` | Random seed |
| `--anneal` | Enable simulated annealing (Potts; always on for Ising) |
| `--use_adaptive_gc_coeff` | Enable adaptive GC coefficient |
| `--no_plot` | Skip plot generation |
| `--output_dir` | Base output directory |

The Ising model additionally accepts `--P_min`, `--P_max`, and `--n_const_P_steps` for the constraint penalty schedule: P stays at `P_min` for the first `n_const_P_steps` annealing steps, then ramps from `P_min` to `P_max` on a log schedule over the remaining steps.

### Output

Each run creates a timestamped directory containing:
- `results.json`: problem spec, hyperparameters, best solution (codons + nucleotides + GC content), energy breakdown, and convergence stats.
- `energy_convergence.png`: 4-panel plot of energy terms over the sampling/annealing trajectory.
- `solution_stats.png`: per-position GC content and codon rarity for the best solution.
- `gc_adaptation.png` (if adaptive GC is enabled): GC fraction and coefficient trajectories.
- `run_log_*.txt`: full text log of the run.

### Running Tests

```bash
uv run pytest tests/          # or: pytest tests/  (inside an activated venv)
```

---

## The `qodon` folder (genetic-algorithm baseline)

The [`qodon/`](qodon/) directory holds the **genetic-algorithm (GA) baseline** that the
paper compares against. Its core GA, scoring, and constants modules are the reference
implementation from:

> Dillion M. Fox, Kim M. Branson, and Ross C. Walker.
> **"mRNA codon optimization with quantum computers."** *PLOS ONE* **16**(10):
> e0259101 (2021). doi:[10.1371/journal.pone.0259101](https://doi.org/10.1371/journal.pone.0259101).

That work was originally released as the `quantum_codon_opt` package under the
**GNU GPL v3** (see [`qodon/README.md`](qodon/README.md)); the license header is
retained in the source files. We use it in two ways:

- as a **scoring oracle** — `qodon/scoring.py`'s `SeqScorer` is used to cross-check our
  energy function (`script_utils.compute_qodon_score`), and
- as the **GA baseline** reported in the paper.

`qodon/fast_ga/` is our **JAX reimplementation** of the GA, with full JIT compilation of
the generation loop, multi-chain parallelism, and the tuned hyperparameters used in the
paper (population 200, 1000 generations, mutation rate 0.003).

`qodon/` modules use flat imports and expect to be run **from within the `qodon/`
directory**. `uv run` discovers the project automatically from subdirectories:

```bash
# Original classical GA (Fox et al.) on the spike protein
cd qodon
uv run python run_ga_spike.py

# Tuned JAX reimplementation (the paper's GA benchmark)
cd qodon/fast_ga
uv run python run_spike.py
```

The GA can also be called directly on any amino-acid sequence:

```python
# from within qodon/
from classical_ga import CodonOptimization
from scoring import SeqScorer

result = CodonOptimization("MFVFLVLLPLVSSQCVNLTTRT")
print(result.n_seq)                 # optimized nucleotide sequence
print(SeqScorer(result.n_seq).score)
```

> The original `qodon.py` also exposes optional D-Wave and IBM quantum backends. Those
> require proprietary D-Wave libraries / Qiskit and external account access, and are not
> installed by this repo's dependencies; see [`qodon/README.md`](qodon/README.md).

---

## Citation

If you use this code, please cite the companion paper (see [`paper/`](paper/); arXiv link
forthcoming) and, for the GA baseline, Fox et al. (2021) above.
