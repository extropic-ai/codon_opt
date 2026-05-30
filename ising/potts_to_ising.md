# Potts-to-Ising Compilation: Literature Review

## 1. Problem Statement

Given a Potts model for codon optimization — where each amino acid position p has a categorical variable $c_p \in \{0, \ldots, K_p-1\}$ with $K_p \leq 6$ — we want to compile it into a binary Ising/Boltzmann machine with spin variables $s_i \in \{0, 1\}$.

The codon optimization energy function has three terms:
- **Codon usage** (unary): prefers common codons
- **GC content** (global quadratic): penalizes deviation from target GC fraction
- **Repeat penalty** (pairwise, adjacent positions): penalizes long nucleotide runs

The Potts model has chain-structured pairwise interactions (adjacent amino acid positions) and is 2-colorable (even/odd), enabling efficient parallel block Gibbs updates.

### Core challenges

1. **Constraint enforcement**: Encoding a K-state categorical variable into binary requires constraints (e.g., one-hot: exactly one of K bits is 1). Exact enforcement requires infinite penalty weights; finite penalties introduce constraint violations.

2. **Graph colorability**: One-hot encoding creates K-cliques (penalty terms), requiring at least K colors. This destroys the 2-colorable chain structure.

3. **Mixing time**: For single-site Gibbs sampling on an Ising model, mixing time scales as $t_{\text{mix}} \geq \exp(\Omega(d \cdot |J_{\max}| \cdot \beta))$ where $d$ is degree and $|J_{\max}|$ is the largest coupling (Levin, Peres & Wilmer 2009; Martinelli 1999). Large constraint penalties directly cause exponentially slow mixing.

---

## 2. Encoding Schemes

### 2.1 One-Hot Encoding

Each K-state Potts variable becomes K binary variables $(x_{p,1}, \ldots, x_{p,K})$ with the constraint $\sum_k x_{p,k} = 1$ enforced via a penalty $P(\sum_k x_{p,k} - 1)^2$.

**Properties (K=6):**
- Variables per position: 6
- Constraint edges per position: K(K-1)/2 = 15 (clique)
- Interaction terms between two positions: $K^2 = 36$
- Chromatic number impact: increases to $\geq K$ (destroys 2-colorability)

**Advantages:**
- Conceptually simple
- Potts pairwise interactions map cleanly to quadratic binary terms

**Disadvantages:**
- K-clique penalty structure is expensive (15 edges for K=6)
- Requires large penalty P that dominates the energy landscape
- Single-spin-flip dynamics must traverse invalid intermediate states (0 or 2 active bits)
- The penalty clique requires $\geq K$ colors, destroying parallel block update structure
- For K>2, the penalty landscape itself has local minima (e.g., (1,1,0,0,0,0) is a local minimum of the penalty but violates the constraint)

**Key references:**
- Lucas (2014), "Ising formulations of many NP problems," Frontiers in Physics 2:5 — foundational QUBO encoding catalog
- Fox, Branson & Walker (2021), "mRNA codon optimization with quantum computers," PLoS ONE 16(10):e0259101 — directly applies one-hot to codon optimization with `combo_penalty=40`; D-Wave handles ~30 AAs with full GC coupling
- Glover, Kochenberger & Du (2019), "A Tutorial on Formulating and Using QUBO Models," arXiv:1811.11538

### 2.2 Domain Wall Encoding

A K-state variable is represented by K-1 binary variables $(q_1, \ldots, q_{K-1})$ arranged in a chain. Valid states are "thermometer" patterns: $11\ldots100\ldots0$. State k means $q_1 = \ldots = q_k = 1$ and $q_{k+1} = \ldots = q_{K-1} = 0$ (with all-zeros = state 0, all-ones = state K-1).

The constraint (no reverse domain walls, i.e., no "01" subsequences) is enforced via nearest-neighbor penalties: $P \cdot q_{i+1}(1 - q_i)$ for each consecutive pair.

**Properties (K=6):**
- Variables per position: K-1 = 5
- Constraint edges per position: K-2 = 4 (chain)
- Interaction terms between two positions: $(K-1)^2 = 25$
- Chromatic number impact: **none** (chain is 2-colorable)

The single-variable Hamiltonian (Chancellor 2019):

$$H = \sum_{k=1}^{K-1} (h_k - h_{k-1}) \cdot q_k + \sum_{i=1}^{K-2} P \cdot q_{i+1}(1 - q_i) + h_0$$

**Advantages over one-hot:**
1. **Sparser constraint graph**: Chain (4 edges) vs. clique (15 edges) for K=6
2. **Lower penalty requirements**: P must exceed max single-variable energy difference, not global energy benefit of constraint violation. In practice 2-5x smaller than one-hot.
3. **Preserves colorability**: Constraint chains are 2-colorable, so the combined graph (domain wall chains + inter-position interactions) has much better coloring properties.
4. **Tighter spectral gap**: Berwald et al. (2022) proved the energy separation between valid and invalid states is larger for the same P, meaning better optimization and sampling.
5. **Convex constraint landscape**: Any domain wall violation can be fixed by a single bit flip (unlike one-hot, where some violations require coordinated multi-bit flips). This directly improves MCMC mixing.
6. **Natural dynamics**: Single-spin flips move the Potts variable to an adjacent state ($k \to k\pm1$), creating smooth exploration in category space.
7. **Ordering opportunity**: Codons can be ordered by GC content or usage frequency so domain wall moves traverse states meaningfully.
8. **Optimality**: Berwald et al. (2022) proved domain wall is optimal in number of binary variables for quadratic-only interactions between discrete variables.

**Disadvantages:**
- Pairwise interaction expansion involves differences of J values (moderately more complex than one-hot)
- State ordering matters for interaction efficiency
- Distant Potts state changes (e.g., $k \to k+3$) require multiple spin flips

**Key references:**
- Chancellor (2019), "Domain wall encoding of discrete variables for quantum annealing and QAOA," QST 4(4):045004, arXiv:1903.05068 — **foundational paper**
- Chen, Stollenwerk & Chancellor (2021), "Performance of Domain-Wall Encoding for Quantum Annealing," IEEE Trans. Quantum Eng., arXiv:2102.12224 — experimental validation; domain wall outperformed one-hot on three problems; D-Wave 2000Q with domain wall beat next-gen Advantage with one-hot
- Berwald, Chancellor & Dridi (2022), "Understanding domain-wall encoding theoretically and experimentally," Phil. Trans. Royal Society A 381(2241):20210410, arXiv:2108.12004 — proves tighter spectral gap; dynamics freeze later than one-hot; optimal variable count

### 2.3 Binary (Logarithmic) Encoding

Represent K states using $\lceil \log_2 K \rceil$ binary variables (3 bits for K=6, representing values 0-7).

**Properties (K=6):**
- Variables per position: 3
- Invalid states (6, 7) need additional penalties
- Pairwise Potts interactions become higher-order (cubic+), requiring auxiliary variables for quadratization
- Single bit flip can change Potts state by up to 4 — extremely rugged landscape

**Not recommended** for Gibbs sampling due to rugged landscape, complex interaction structure, and invalid-state handling.

**Key reference:**
- Tamura, Shirai, Katsura, Tanaka & Togawa (2021), "Performance Comparison of Typical Binary-Integer Encodings in an Ising Machine," IEEE Access 9:81032-81039 — binary encoding has most complex interaction structure; unary/domain wall performs best for small K

### 2.4 Gray Code Encoding

A variant of binary encoding where successive states differ by exactly one bit. For K=6 with 3 bits: 000, 001, 011, 010, 110, 111 (100 and 101 invalid).

Adjacent Potts states are Hamming distance 1 apart (improving over standard binary), but interaction structure remains complex and invalid-state penalties are still needed. Not widely adopted.

### 2.5 Summary Table

| Encoding | Vars/K | Constraint edges | Penalty strength | Chromatic impact | Gibbs mixing |
|----------|--------|-----------------|------------------|-----------------|-------------|
| **Native Potts** | 1 categorical | 0 | N/A | None | Optimal |
| **One-hot** | K (6) | K(K-1)/2 = 15 (clique) | Large (must exceed max obj. benefit) | Increases to $\geq$ K | Poor |
| **Domain wall** | K-1 (5) | K-2 = 4 (chain) | ~2-5x smaller than one-hot | **None** (chain is 2-colorable) | Moderate |
| **Binary** | $\lceil\log_2 K\rceil$ (3) | Complex | Complex | Unpredictable | Very poor |

---

## 3. Penalty Strength and Mixing Time

### 3.1 Theoretical Bounds on Mixing Time

The fundamental result (Levin, Peres & Wilmer 2009; Martinelli 1999) for Glauber dynamics (single-site Gibbs) on an Ising model:

$$t_{\text{mix}} \geq \exp\bigl(\Omega(d \cdot |J_{\max}| \cdot \beta)\bigr)$$

Slow mixing occurs when $\beta \cdot |J_{\max}| > 1/(2d)$.

**Concrete estimates for the codon problem:**

| Encoding | Penalty P | Degree d | Slow mixing threshold $\beta$ | $t_{\text{mix}}$ at $\beta=1$ |
|----------|-----------|----------|-------------------------------|-------------------------------|
| One-hot (qodon) | 40 | 5 | $> 1/(2 \cdot 5 \cdot 40) = 0.0025$ | $\sim \exp(200)$ |
| One-hot (min theory) | ~1.2 | 5 | $> 1/12 \approx 0.083$ | $\sim \exp(6)$ |
| Domain wall | ~5 | 2 | $> 1/(2 \cdot 2 \cdot 5) = 0.05$ | $\sim \exp(10)$ |

The `combo_penalty=40` used in Fox et al. (qodon) is ~30x larger than the Lucas (2014) theoretical minimum ($P > K \cdot |J_{\max}| \approx 6 \times 0.2 = 1.2$).

### 3.2 Domain Wall Mixing is P-Independent

**The standard mixing time bound does not apply to the domain wall constraint chain.** Although the couplings $J = P/4$ grow with $P$, a Gibbs update at the domain wall boundary (where $s_i = +1, s_{i+1} = -1$) sees an effective field of:

$$h_{\text{eff}} = b_i + J \cdot s_{i-1} + J \cdot s_{i+1}$$

For the spin at the wall, one neighbour is $+1$ and the other is $-1$, so the coupling contributions cancel: $J(+1) + J(-1) = 0$. The Gibbs conditional depends only on the bias $b_i$, not on $P$. The domain wall therefore performs an unbiased random walk in the bias potential, and its mixing time is governed by the bias landscape alone.

**Experimental verification** (`domain_wall_mixing_experiment.py`): For a $K=6$ Potts variable with random biases, we measured total variation distance between the empirical Gibbs distribution and the exact Boltzmann distribution for $P \in [0.5, 1000]$ (a 2000$\times$ range). Results:

- TV distance is constant at $\approx 0.006$ (pure finite-sample noise from $50{,}000$ samples) across all $P$ values.
- The chain is already mixed at **zero warmup** — TV is identical for $n_{\text{warmup}} \in \{0, 5, 10, \ldots, 500\}$.
- The only effect of $P$ is on constraint enforcement: $P \geq 10$ gives $> 99.98\%$ probability on valid states.

This is a crucial advantage over one-hot, where the $K$-clique penalty creates genuine energy barriers that scale with $P$.

### 3.3 Key Theoretical References

- **Levin, Peres & Wilmer (2009)**, *Markov Chains and Mixing Times* — standard reference; mixing time exponential in max coupling $\times$ degree $\times$ $\beta$
- **Martinelli (1999)**, "Lectures on Glauber dynamics for discrete spin models" — slow mixing when $\beta \cdot J > 1/(2d)$; bottleneck at strongest couplings
- **Borgs, Chayes & Tetali (2012)**, "Tight bounds for mixing of Swendsen-Wang at the Potts transition" — single-site Gibbs exponentially slow at phase transition; cluster algorithms polynomial
- **Ge & Risteski (2018)** / **Sly (2010)** — all local MCMC exponentially slow beyond uniqueness threshold; information-theoretic barrier, not just algorithmic

### 3.4 Penalty Strength Selection Rules

| Source | Rule |
|--------|------|
| Lucas (2014) | $P > K \cdot |J_{\max}|$ where K is constraint size, $J_{\max}$ largest objective coupling |
| Glover et al. (2019) | $P \geq$ max absolute coefficient in objective; increase if violations persist |
| Bian et al. (2010/2014) | $P > \Delta_{\text{obj}} / 2$; one-hot with K>2 creates local minima in penalty landscape |
| D-Wave community | Chain strength $\approx 1.5$-$2\times$ max coupling; going beyond 5x is counterproductive |
| Ayodele (2022) | Proposes better static methods for penalty calculation; too large $\Rightarrow$ slow convergence, too small $\Rightarrow$ infeasible solutions |

**Reference:** Ayodele (2022), "Penalty Weights in QUBO Formulations: Permutation Problems," EvoCOP 2022, arXiv:2206.11040

---

## 4. Penalty Scheduling (Separate Schedules for Constraints vs. Objective)

### 4.1 The Core Insight

Standard approach (fixed penalty + temperature annealing) fails because:
- At high temperature: constraint violations common (good mixing but infeasible solutions)
- At low temperature: large penalty creates energy barriers that trap the sampler

Several papers propose **decoupling** the constraint penalty schedule from the temperature schedule.

### 4.2 Key Papers

**Hen & Spedalieri (2016)**, "Quantum Annealing for Constrained Optimization," Phys. Rev. Applied 5(3):034007
- Constraint Hamiltonian kept separate from objective; constraint remains "on" while objective is annealed in
- Penalty magnitude required to maintain spectral gap grows with system size, degrading performance
- **Takeaway**: Keep constraint penalties high and fixed while ramping $\beta$ on the objective

**Ohzeki (2020)**, "Breaking limitation of quantum annealer in solving optimization problems under constraints," Scientific Reports 10:3126
- Treats constraint penalty as a separate parameter from annealing schedule
- Proposes soft constraints at high T, tightened as system cools, on a different schedule than objective temperature
- **Takeaway**: Start with moderate constraint penalties, ramp up alongside but faster than $\beta$

**Vyskočil & Djidjev (2019)**, "Embedding Equality Constraints of Optimization Problems into a Quantum Annealer"
- Penalty schedule and temperature schedule interact nonlinearly
- Simply increasing penalty does not monotonically improve constraint satisfaction beyond a threshold
- **Takeaway**: Joint tuning of penalty and temperature schedules is needed

**Smith & Coit (1997)**, "Penalty Functions," Handbook of Evolutionary Computation
- "Annealing penalty": $P(t) = P_0 \cdot (t/T)^\alpha$ with $\alpha = 2$
- Early iterations focus on objective with relaxed constraints; later enforce feasibility
- **Takeaway**: Ramp penalty from small to large over the annealing schedule, faster than temperature

**Ayanzadeh et al. (2020)**, "An Ensemble Approach to Quantum Annealing with Constraint Relaxation"
- Run annealing with relaxed constraints (smaller P), post-select/repair feasible solutions
- Reducing penalty improves objective quality via better exploration
- **Takeaway**: Use smaller penalties and repair constraint violations in post-processing

### 4.3 Adaptive Penalty Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| Static penalty | Fixed weight (e.g., `combo_penalty=40`) | Fox et al. (2021) |
| Annealing penalty | $P(t) = P_0 \cdot (t/T)^\alpha$ | Smith & Coit (1997) |
| Feasibility feedback | Adjust P to maintain target feasibility rate | Yokota et al. (1996) |
| Lagrangian relaxation | $\lambda_{t+1} = \max(0, \lambda_t + \alpha \cdot g_t)$ | Convex optimization duality |
| Self-adaptive | Penalty weight itself subject to optimization | Coello Coello (2002) survey |

Note: The adaptive GC coefficient already implemented in `problem.py` (`adapt_gc_coeff`) is a Lagrangian-style adaptive penalty — the same approach could handle one-hot/domain-wall constraints.

### 4.4 Note on Domain Wall and Penalty Scheduling

The penalty scheduling literature above is primarily relevant for **one-hot encoding**, where large $P$ creates energy barriers in the $K$-clique that degrade mixing. For **domain wall encoding**, constraint chain mixing is $P$-independent (§3.2), so separate penalty scheduling for constraint enforcement is unnecessary.

However, P-ramping is still useful for domain wall for a *different* reason: enabling invalid-state shortcuts around energy barriers in the bias landscape. See §7.4 for details.

---

## 5. Codon Optimization on Quantum/Ising Hardware

### 5.1 Existing Work

**Fox, Branson & Walker (2021)**, "mRNA codon optimization with quantum computers," PLoS ONE
- One-hot QUBO encoding of codon optimization
- Tested on D-Wave and IBM quantum simulators
- GC content creates all-to-all coupling — main scalability bottleneck
- D-Wave handles ~30 AAs with full GC; ~1000 AAs with local terms only
- `combo_penalty=40`, `gc_constant=2e4`, `rarity_constant=0.1`, `repeat_constant=0.2`
- Reference implementation in `qodon/` directory of this repo

**arxiv:2507.18817 (2025)**, "Co-optimization of codon usage and mRNA secondary structure using quantum computing"
- Dual-objective: CAI + minimum free energy (MFE)
- Hybrid quantum-classical; IBM 127-qubit Eagle processor
- Validated on sequences up to 30 nucleotides
- Scales to 156 qubits / 950 non-local gates

**arxiv:2404.14858 (2024)**, "A resource-efficient variational quantum algorithm for mRNA codon optimization"
- VQE-based approach to codon optimization

### 5.2 Structurally Analogous Problems

**Mulligan, Malone et al. (2020)**, "Quantum Annealing for Computational Protein Design"
- Protein side-chain packing (rotamer selection) as QUBO
- **Most directly analogous**: choose one categorical option per position with pairwise energy terms
- Structurally identical to codon optimization with codon-pair interactions
- One-hot encoding; benchmarked on D-Wave

**Perdomo-Ortiz et al. (2012)**, "Finding low-energy conformations of lattice protein models by quantum annealing," Scientific Reports 2:571
- Lattice protein folding as QUBO; one-hot encoding of categorical choices

---

## 6. The GC Content Term

Regardless of encoding, the GC content term $c_{\text{GC}} \cdot \bigl(\frac{1}{N}\sum_p g(c_p) - \rho_T\bigr)^2$ creates **all-to-all quadratic coupling** between all positions/variables. This is the primary scalability bottleneck:

- Under one-hot: expands to $O(N^2)$ quadratic terms where $N$ is total binary variables
- Fox et al. found D-Wave can only handle ~30 amino acids with full GC coupling
- Our adaptive linear GC coefficient approach (in `problem.py`, `adapt_gc_coeff`) avoids this by linearizing the GC term and adapting the coefficient during annealing — this works at the Potts level and would also work after domain wall compilation

---

## 7. Implementation Plan (Domain Wall)

We use domain wall encoding. The evidence is overwhelming (§2.2): tighter spectral gap, optimal variable count, 2-colorable constraint graph, and — crucially — $P$-independent mixing (§3.2).

**Implementation status key:** Items marked **[IMPLEMENTED]** are in the current codebase. Items marked **[NOT YET IMPLEMENTED]** are future enhancements described here for reference.

### 7.1 Annealing strategy: ramp $P$ alongside $\beta$ — [IMPLEMENTED]

We anneal both $\beta$ and $P$ together. The $\beta$ schedule drives the usual simulated annealing (high temperature → low temperature). The $P$ schedule serves a dual purpose:

- **Low $P$ (early)**: invalid-state (defect) transitions are accessible, providing shortcuts around energy barriers in the bias landscape (see §7.4). Good exploration.
- **High $P$ (late)**: constraint strictly enforced, the domain wall is confined to valid thermometer states.

Since each annealing step already requires re-computing and re-flashing Ising weights (to update $\beta$ and the adaptive GC coefficient), changing $P$ costs nothing extra. Both can be log-spaced schedules. The constraint chain mixing remains $P$-independent (§3.2) at all stages — $P$-ramping only affects whether invalid states act as accessible shortcuts.

Implementation: `ising/ising_model.py` (`run_ising_annealing`) accepts separate `betas` and `Ps` arrays.

### 7.2 Codon ordering within domain wall — [IMPLEMENTED]

Adjacent domain wall states differ by a single spin flip ($c \to c \pm 1$). We order codons by **ascending GC count** (ties broken alphabetically) so that each spin flip changes GC content by at most 1. This makes the GC landscape smooth under the natural dynamics.

Amino acids with $K = 1$ (Met, Trp) have no synonymous codons and need no encoding.

| AA | $K$ | Spins | Ordered codons (GC count) |
|----|-----|-------|---------------------------|
| A | 4 | 3 | GCA(2), GCT(2), GCC(3), GCG(3) |
| C | 2 | 1 | TGT(1), TGC(2) |
| D | 2 | 1 | GAT(1), GAC(2) |
| E | 2 | 1 | GAA(1), GAG(2) |
| F | 2 | 1 | TTT(0), TTC(1) |
| G | 4 | 3 | GGA(2), GGT(2), GGC(3), GGG(3) |
| H | 2 | 1 | CAT(1), CAC(2) |
| I | 3 | 2 | ATA(0), ATT(0), ATC(1) |
| K | 2 | 1 | AAA(0), AAG(1) |
| L | 6 | 5 | TTA(0), CTA(1), CTT(1), TTG(1), CTC(2), CTG(2) |
| M | 1 | 0 | ATG(1) — *fixed* |
| N | 2 | 1 | AAT(0), AAC(1) |
| P | 4 | 3 | CCA(2), CCT(2), CCC(3), CCG(3) |
| Q | 2 | 1 | CAA(1), CAG(2) |
| R | 6 | 5 | AGA(1), AGG(2), CGA(2), CGT(2), CGC(3), CGG(3) |
| S | 6 | 5 | AGT(1), TCA(1), TCT(1), AGC(2), TCC(2), TCG(2) |
| T | 4 | 3 | ACA(1), ACT(1), ACC(2), ACG(2) |
| V | 4 | 3 | GTA(1), GTT(1), GTC(2), GTG(2) |
| W | 1 | 0 | TGG(2) — *fixed* |
| Y | 2 | 1 | TAT(0), TAC(1) |

Average: 2.05 domain-wall spins per position. For the spike protein (1273 AA), this gives ~2610 Ising spins (vs. 1273 Potts variables).

Implementation: `problem.py` (`SORTED_CODON_TABLE`). Both Potts and Ising models use this single ordering.

### 7.3 Keep the adaptive GC approach — [IMPLEMENTED]

The Lagrangian-style adaptive GC coefficient (in `problem.py`, `adapt_gc_coeff`) avoids the all-to-all coupling bottleneck. It works at the Potts level and transfers directly to the Ising model: the GC count per codon is a function of the domain wall state, so the adaptive linear GC term becomes a position-dependent bias adjustment applied each annealing step.

Implementation: `problem.py` (`adapt_gc_coeff`), used in both `potts/model.py` (`run_annealing`) and `ising/ising_model.py` (`run_ising_annealing`).

### 7.4 Mixing barriers within domain wall chains

The $P$-independence result (§3.2) shows that the constraint chain couplings do not impede mixing. However, the **bias landscape** within a DWC can itself create barriers that degrade mixing.

**The problem.** The domain wall performs a 1D birth-death random walk on Potts states $0, 1, \ldots, K_p{-}1$. Single-site Gibbs can only move the wall by $\pm 1$ (assuming $P$ is large enough to suppress invalid states). If some intermediate state $k$ has very high energy, the domain wall must pass through it to reach states on the other side, picking up an acceptance penalty of $\sim \exp(-\beta \cdot h_k)$. For example, with $K=6$ and a very rare codon at DWC position 3: transitioning from $c=1$ to $c=5$ requires passing through $c=3$, which becomes nearly impossible at high $\beta$.

This is most relevant for amino acids with $K = 6$ (Leu, Arg, Ser) where some codons can be orders of magnitude rarer than others. For $K = 2$ there is no intermediate state, so no barrier is possible. For $K = 3$ or $K = 4$ it can occur but is less severe.

**Mitigation 1: Adaptive DWC ordering — [NOT YET IMPLEMENTED].** The order of Potts states within each DWC determines the shape of the 1D landscape. By sorting states so that the highest-energy states sit at the chain ends, interior barriers are eliminated. The sort key is the **total Potts bias** $h_k^{(p)}$ at each position, which includes all contributions: codon usage, the current adaptive GC coefficient, and pairwise effects from neighbors. Since these biases change during annealing (the adaptive GC coefficient changes, pairwise contributions shift), the optimal ordering can change between annealing steps. Each step, we recompute biases, re-sort, and re-encode the Ising weights accordingly.

An even more adaptive variant: after each annealing step, observe the **empirical frequencies** of each Potts state from the sampled batch and use those to order the DWC for the next step. States that were rarely visited go to the chain edges. This is model-free and adapts to the actual landscape the sampler experienced.

When the ordering changes between steps, the current state must be reshuffled: decode each position's DWC to its Potts state (in the old ordering), then re-encode in the new ordering.

**Mitigation 2: Gradual P-ramping — [IMPLEMENTED].** With moderate $P$, invalid (defect) states in the DWC become accessible. A single defect costs energy $\sim P$, so the Gibbs probability of accepting a defect transition is $\sim \exp(-\beta P)$. Crucially, if a valid intermediate state $k$ has energy $h_k > P$, then the defect route (cost $P$) is **cheaper** than the valid route (cost $h_k$). Moderate $P$ therefore provides shortcuts around exactly the high-energy states that cause barriers.

Schedule: start with moderate $P$ so that $\beta P$ is not too large, allowing defect-mediated jumps; ramp to large $P$ to enforce strict validity at the end. The optimal $P$ at each stage should be roughly comparable to the typical energy differences between Potts states, not dramatically larger.

**Mitigation 3: Diverse orderings across parallel chains — [NOT YET IMPLEMENTED].** When running multiple parallel Gibbs chains, use different DWC orderings across chains. Different orderings create different barrier structures, so the ensemble explores the Potts state space more effectively than identically-ordered chains. After read-back, states can be decoded to Potts and re-encoded in each chain's ordering for the next step.

**JAX implementation notes.** To support per-chain orderings, maintain an array `ordering_spec: Int[Array, "n_chains L K"]` storing the codon permutation for each chain at each amino acid position. Two utility functions map between the "canonical" codon ordering (as defined by the codon table) and the per-chain DWC ordering:

- `canonical_to_dwc(potts_indices, ordering_spec)` — permute canonical Potts indices to DWC order
- `dwc_to_canonical(dwc_indices, ordering_spec)` — inverse permutation

Since all chains share the same Ising **graph structure** (only weights and biases differ between chains), Gibbs sampling remains efficiently parallelizable via `vmap` over the chain dimension. The Potts-to-Ising weight compilation is also `vmap`-able: given per-chain biases and orderings, produce per-chain Ising weights.

**V1 simplification.** The first implementation will **not** use adaptive DWC reordering. All chains use the fixed GC-sorted ordering from §7.2. Mixing barriers are addressed solely through gradual P-ramping. This avoids the considerable complexity of per-chain ordering management, state reshuffling between annealing steps, and the `ordering_spec` machinery. Adaptive reordering can be added as a future enhancement once the base implementation is validated.

---

## 8. Target Hardware: Thermodynamic Sampling Unit (TSU)

The long-term target for this algorithm is a **Thermodynamic Sampling Unit (TSU)**: a sub-threshold analog CMOS-based Ising machine with sparse, local connectivity. The current JAX implementation is a software prototype intended to demonstrate that the algorithmic approach works, validating that a future TSU deployment would be viable. We are not currently writing code for a TSU.

The TSU's operational model has three phases with very different costs:

1. **Flash weights** ($\sim 100 \times$ a single Gibbs step): Upload biases and couplings from a host computer to the TSU.
2. **Gibbs sampling** (baseline cost): The TSU performs single-site Ising Gibbs sampling at extremely high speed and low energy. During this phase, weights and biases are **fixed** — the only operation is pure Ising Gibbs with the flashed parameters. No algorithmic tricks are possible during this phase.
3. **Read state** ($\sim 10-100 \times$ a single Gibbs step): Read the current spin configuration back to the host.

The natural algorithm is an **outer loop on the host** (compute weights from current annealing parameters, flash, read back, update parameters) with an **inner loop on the TSU** (many Gibbs steps with fixed weights). Steps 1 and 3 are expensive relative to a single Gibbs iteration but cheap in absolute terms, so the outer loop can run frequently — e.g., once per annealing step.

Key implications for algorithm design:
- All "intelligence" (annealing schedules, adaptive GC, ordering decisions, P-ramping) happens on the host between TSU sampling bursts.
- The TSU graph structure is assumed to match the Ising model we construct (same nodes, same edges).
- Multiple parallel chains can run simultaneously on the TSU.

---

## 9. Complete Reference List

### Encoding Theory
1. Lucas, A. (2014). "Ising formulations of many NP problems." *Frontiers in Physics* 2:5.
2. Chancellor, N. (2019). "Domain wall encoding of discrete variables for quantum annealing and QAOA." *Quantum Science and Technology* 4(4):045004. arXiv:1903.05068.
3. Berwald, J., Chancellor, N., Dridi, R. (2022). "Understanding domain-wall encoding theoretically and experimentally." *Phil. Trans. Royal Society A* 381(2241):20210410. arXiv:2108.12004.
4. Chen, J., Stollenwerk, T., Chancellor, N. (2021). "Performance of Domain-Wall Encoding for Quantum Annealing." *IEEE Trans. Quantum Engineering*. arXiv:2102.12224.
5. Tamura, R., Shirai, T., Katsura, H., Tanaka, S., Togawa, N. (2021). "Performance Comparison of Typical Binary-Integer Encodings in an Ising Machine." *IEEE Access* 9:81032-81039.
6. Glover, F., Kochenberger, G., Du, Y. (2019). "A Tutorial on Formulating and Using QUBO Models." arXiv:1811.11538.

### Penalty Strength and Mixing Time
7. Levin, D.A., Peres, Y., Wilmer, E.L. (2009). *Markov Chains and Mixing Times*. AMS. (2nd ed. 2017)
8. Martinelli, F. (1999). "Lectures on Glauber dynamics for discrete spin models." *Springer LNM* 1717.
9. Ayodele, M. (2022). "Penalty Weights in QUBO Formulations: Permutation Problems." *EvoCOP 2022*. arXiv:2206.11040.
10. Bian, Z., Chudak, F., Macready, W.G., Clark, L., Gaitan, F. (2010/2014). "The Ising model: teaching an old problem new tricks."

### Penalty Scheduling and Adaptive Methods
11. Hen, I., Spedalieri, F.M. (2016). "Quantum Annealing for Constrained Optimization." *Phys. Rev. Applied* 5(3):034007.
12. Ohzeki, M. (2020). "Breaking limitation of quantum annealer in solving optimization problems under constraints." *Scientific Reports* 10:3126.
13. Vyskočil, T., Djidjev, H. (2019). "Embedding Equality Constraints of Optimization Problems into a Quantum Annealer."
14. Smith, A.E., Coit, D.W. (1997). "Penalty Functions." *Handbook of Evolutionary Computation*.
15. Coello Coello, C.A. (2002). "Theoretical and numerical constraint-handling techniques used with evolutionary algorithms: a survey." *CMAME* 191(11-12):1245-1287.

### Codon Optimization and Related Biological QUBO Problems
16. Fox, D.M., Branson, K.M., Walker, R.C. (2021). "mRNA codon optimization with quantum computers." *PLoS ONE* 16(10):e0259101.
17. "Co-optimization of codon usage and mRNA secondary structure using quantum computing." (2025). arXiv:2507.18817.
18. "A resource-efficient variational quantum algorithm for mRNA codon optimization." (2024). arXiv:2404.14858.
19. Mulligan, V.K., Malone, B.D., et al. (2020). "Quantum Annealing for Computational Protein Design."
20. Perdomo-Ortiz, A., et al. (2012). "Finding low-energy conformations of lattice protein models by quantum annealing." *Scientific Reports* 2:571.

### Additional
21. Choi, V. (2011). "Minor-embedding in adiabatic quantum computation: II." *QIP* 10(3):343-353.
22. Dattani, N. (2019). "Quadratization in discrete optimization and quantum mechanics." arXiv:1901.04405.
23. Karimi, S., Rosenberg, G. (2019). "Practical Integer-to-Binary Mapping for Quantum Annealers." *QIP* 18(4).
24. Borgs, C., Chayes, J., Tetali, P. (2012). "Tight bounds for mixing of the Swendsen-Wang algorithm at the Potts transition point." *Prob. Theory Related Fields* 152:509-557.
