"""Codon optimization problem utilities (thrml-agnostic).

Provides the CodonProblem definition, energy computation, codon ordering tables,
and unified Potts weight computation used by both Potts and Ising models.
"""

from dataclasses import field
from typing import Callable

import equinox as eqx
import jax
import numpy as np
import python_codon_tables as pct
from jax import numpy as jnp

# ---------------------------------------------------------------------------
# Genetic code tables
# ---------------------------------------------------------------------------

# Standard genetic code (DNA codons, T instead of U)
CODON_TABLE: dict[str, list[str]] = {
    "A": ["GCT", "GCC", "GCA", "GCG"],  # Alanine - 4 codons
    "C": ["TGT", "TGC"],  # Cysteine - 2 codons
    "D": ["GAT", "GAC"],  # Aspartic acid - 2 codons
    "E": ["GAA", "GAG"],  # Glutamic acid - 2 codons
    "F": ["TTT", "TTC"],  # Phenylalanine - 2 codons
    "G": ["GGT", "GGC", "GGA", "GGG"],  # Glycine - 4 codons
    "H": ["CAT", "CAC"],  # Histidine - 2 codons
    "I": ["ATT", "ATC", "ATA"],  # Isoleucine - 3 codons
    "K": ["AAA", "AAG"],  # Lysine - 2 codons
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],  # Leucine - 6 codons
    "M": ["ATG"],  # Methionine - 1 codon (start)
    "N": ["AAT", "AAC"],  # Asparagine - 2 codons
    "P": ["CCT", "CCC", "CCA", "CCG"],  # Proline - 4 codons
    "Q": ["CAA", "CAG"],  # Glutamine - 2 codons
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],  # Arginine - 6 codons
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],  # Serine - 6 codons
    "T": ["ACT", "ACC", "ACA", "ACG"],  # Threonine - 4 codons
    "V": ["GTT", "GTC", "GTA", "GTG"],  # Valine - 4 codons
    "W": ["TGG"],  # Tryptophan - 1 codon
    "Y": ["TAT", "TAC"],  # Tyrosine - 2 codons
}

MAX_SYNONYMOUS_CODONS = 6  # Maximum number of synonymous codons for any amino acid


def gc_count(codon: str) -> int:
    """Count G and C nucleotides in a codon."""
    return sum(1 for nt in codon if nt in "GC")


def _gc_sorted_codons(aa: str) -> list[str]:
    """Return codons for amino acid sorted by (gc_count, alphabetical)."""
    return sorted(CODON_TABLE[aa], key=lambda c: (gc_count(c), c))


# GC-sorted codon ordering used by both Potts and Ising models.
# Adjacent indices differ smoothly in GC content, which is important for
# domain wall encoding dynamics (see ising/potts_to_ising.md §7.2).
SORTED_CODON_TABLE: dict[str, list[str]] = {aa: _gc_sorted_codons(aa) for aa in CODON_TABLE}

# Very negative bias for invalid codon slots (effectively -infinity)
INVALID_BIAS = -1e10

# SARS-CoV-2 spike protein sequence (1273 amino acids).
# Contains spaces for readability; callers should use .replace(" ", "").
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
    "IAIPTNVTI SVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRAL TGI"
    "AVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTL"
    "ADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWT"
    "F GAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALG"
    "KLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQ"
    "TYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLH"
    "VTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVS"
    "G NCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKE"
    "IDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCS"
    "CLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def longest_run(s: str) -> int:
    """Find the longest contiguous run of identical characters."""
    if not s:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def repeat_penalty(codon1: str, codon2: str) -> float:
    """Compute the repeated nucleotide penalty between adjacent codons.

    Returns m^2 - 1 where m is the longest run in the concatenated string.
    """
    concat = codon1 + codon2
    m = longest_run(concat)
    return float(m * m - 1)


# ---------------------------------------------------------------------------
# Codon frequency / rarity tables
# ---------------------------------------------------------------------------

def get_default_codon_frequencies() -> dict[str, float]:
    """Get default (uniform within each amino acid) codon frequencies.

    For real applications, use organism-specific codon usage tables.
    """
    freqs = {}
    for aa, codons in CODON_TABLE.items():
        freq = 1.0 / len(codons)
        for codon in codons:
            freqs[codon] = freq
    return freqs


def _load_codon_table_from_pct(species: str = "e_coli_316407") -> tuple[dict[str, float], dict[str, float]]:
    """Load codon frequencies and rarity scores from python_codon_tables.

    Returns:
        Tuple of (frequencies, rarity_scores) where:
        - frequencies: dict mapping codon -> raw frequency from the table
        - rarity_scores: dict mapping codon -> |log(freq / max_freq_for_aa)|
          This is 0 for the most common codon of each amino acid, positive for rarer codons.
    """
    table = pct.get_codons_table(species)

    frequencies = {}
    rarity_scores = {}

    for aa, codon_dict in table.items():
        if aa == "*":  # Skip stop codons
            continue

        # Get all codons and their frequencies for this amino acid
        codons_and_freqs = [(codon, freq) for codon, freq in codon_dict.items()]
        max_freq = max(freq for _, freq in codons_and_freqs)

        for codon, freq in codons_and_freqs:
            frequencies[codon] = float(freq)
            # Rarity score: |log(freq / max_freq)| = log(max_freq) - log(freq) when freq <= max_freq
            # Use a small epsilon to avoid log(0)
            eps = 1e-10
            rarity_scores[codon] = float(abs(np.log((freq + eps) / (max_freq + eps))))

    return frequencies, rarity_scores


# Cache the loaded codon tables to avoid repeated file I/O
_CODON_TABLE_CACHE: dict[str, tuple[dict[str, float], dict[str, float]]] = {}


def get_codon_data(species: str = "e_coli_316407") -> tuple[dict[str, float], dict[str, float]]:
    """Get codon frequencies and rarity scores for a species.

    Args:
        species: Species identifier for python_codon_tables (default: E. coli K-12).

    Returns:
        Tuple of (frequencies, rarity_scores) where:
        - frequencies: dict mapping codon -> raw frequency
        - rarity_scores: dict mapping codon -> |log(freq / max_freq_for_aa)|
    """
    if species not in _CODON_TABLE_CACHE:
        _CODON_TABLE_CACHE[species] = _load_codon_table_from_pct(species)
    return _CODON_TABLE_CACHE[species]


def get_ecoli_codon_frequencies() -> dict[str, float]:
    """Get E. coli K-12 codon usage frequencies.

    Returns the raw frequencies from python_codon_tables (e_coli_316407).
    """
    freqs, _ = get_codon_data("e_coli_316407")
    return freqs


def get_ecoli_codon_rarity_scores() -> dict[str, float]:
    """Get E. coli K-12 codon rarity scores.

    Returns |log(freq / max_freq_for_aa)| for each codon, which is 0 for
    the most common codon of each amino acid and positive for rarer codons.
    """
    _, rarity = get_codon_data("e_coli_316407")
    return rarity


def get_default_rarity_scores() -> dict[str, float]:
    """Get default (zero) rarity scores - uniform codon preference."""
    return {codon: 0.0 for codons in CODON_TABLE.values() for codon in codons}


# Keep underscore alias for backward compatibility
_get_default_rarity_scores = get_default_rarity_scores


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class CodonProblem(eqx.Module):
    """Contains all inputs for a codon optimization problem.

    Attributes:
        amino_acid_seq: The protein sequence (single-letter codes).
        codon_rarity_scores: Mapping from codon to rarity score |log(freq/max_freq)|.
            This is 0 for the most common codon of each amino acid and positive for rarer.
        target_gc_fraction: Target GC content (0-1), or None to disable.
        weight_codon_usage: Weight for codon usage term (c_f).
        weight_gc: Weight for GC content term (c_GC).
        weight_repeat: Weight for repeat penalty term (c_R).
    """

    amino_acid_seq: str
    codon_rarity_scores: dict[str, float] = field(default_factory=get_default_rarity_scores)
    target_gc_fraction: float | None = None
    weight_codon_usage: float = 1.0
    weight_gc: float = 1.0
    weight_repeat: float = 1.0

    def __post_init__(self):
        for aa in self.amino_acid_seq:
            if aa not in CODON_TABLE:
                raise ValueError(f"Unknown amino acid: {aa}")

    @property
    def seq_length(self) -> int:
        return len(self.amino_acid_seq)

    @property
    def nucleotide_length(self) -> int:
        return 3 * self.seq_length

    def num_codons_at(self, position: int) -> int:
        """Number of synonymous codons at a given position."""
        aa = self.amino_acid_seq[position]
        return len(CODON_TABLE[aa])

    def codons_at(self, position: int) -> list[str]:
        """List of synonymous codons at a given position (GC-sorted)."""
        aa = self.amino_acid_seq[position]
        return SORTED_CODON_TABLE[aa]


# ---------------------------------------------------------------------------
# Energy computation (codon-string based, ordering-independent)
# ---------------------------------------------------------------------------

def compute_energy(
    problem: CodonProblem, codon_seq: list[str], return_terms: bool = False
) -> float | tuple[float, dict[str, float]]:
    """Compute the energy of a codon sequence.

    The codon usage term uses rarity scores: |log(freq / max_freq_for_aa)|.
    This is 0 for the most common codon of each amino acid (optimal) and
    positive for rarer codons. This matches the qodon reference implementation.

    Args:
        problem: The problem instance.
        codon_seq: List of codons (one per amino acid position).
        return_terms: If True, also return individual energy terms.

    Returns:
        Total energy, or (total_energy, terms_dict) if return_terms=True.
    """
    L = len(codon_seq)
    N = 3 * L

    # Validate
    if L != problem.seq_length:
        raise ValueError(f"Codon sequence length {L} != amino acid sequence length {problem.seq_length}")

    # Codon usage term: c_f * sum(|log(freq / max_freq)|)
    # Common codons have rarity score 0, rare codons have positive rarity score
    usage_term = 0.0
    for codon in codon_seq:
        rarity = problem.codon_rarity_scores.get(codon, 0.0)
        usage_term += rarity
    usage_term *= problem.weight_codon_usage

    # GC content term: c_GC * (gc_fraction - target)^2
    if problem.target_gc_fraction is not None:
        total_gc = sum(gc_count(c) for c in codon_seq)
        gc_fraction = total_gc / N
        gc_term = problem.weight_gc * (gc_fraction - problem.target_gc_fraction) ** 2
    else:
        gc_term = 0.0

    # Repeated nucleotide term: c_R * sum(r(c_p, c_{p+1}))
    repeat_term = 0.0
    for i in range(L - 1):
        repeat_term += repeat_penalty(codon_seq[i], codon_seq[i + 1])
    repeat_term *= problem.weight_repeat

    total = usage_term + gc_term + repeat_term

    if return_terms:
        terms = {
            "usage": float(usage_term),
            "gc": float(gc_term),
            "repeat": float(repeat_term),
            "total": float(total),
        }
        return float(total), terms
    return float(total)


def indices_to_codons(problem: CodonProblem, indices: np.ndarray) -> list[str]:
    """Convert codon indices (GC-sorted ordering) to a codon sequence.

    Args:
        problem: The codon problem instance.
        indices: Array of shape (L,) with codon indices in SORTED_CODON_TABLE ordering.

    Returns:
        List of codon strings.
    """
    codon_seq = []
    for i, aa in enumerate(problem.amino_acid_seq):
        codons = SORTED_CODON_TABLE[aa]
        idx = int(indices[i])
        if idx >= len(codons):
            raise ValueError(f"Invalid codon index {idx} for amino acid {aa} (has {len(codons)} codons)")
        codon_seq.append(codons[idx])
    return codon_seq


def codons_to_nucleotides(codon_seq: list[str]) -> str:
    """Convert a codon sequence to a nucleotide string."""
    return "".join(codon_seq)


# ---------------------------------------------------------------------------
# Potts weight computation (uses SORTED_CODON_TABLE)
# ---------------------------------------------------------------------------

def compute_unary_biases(problem: CodonProblem, linear_gc_coeff=0.0):
    """Compute Potts unary biases in GC-sorted codon ordering.

    The bias for codon c is: -c_f * rarity_score(c) + linear_gc_coeff * gc_count(c).
    Invalid codon slots (padding) get INVALID_BIAS.

    If linear_gc_coeff is a JAX array, the result supports gradient flow.

    Args:
        problem: The codon optimization problem.
        linear_gc_coeff: Coefficient for linear GC term.

    Returns:
        (biases, Ks) where biases has shape [L, K_max] and Ks[p] is the
        number of valid codons at position p.
    """
    L = problem.seq_length
    K_max = MAX_SYNONYMOUS_CODONS

    base_biases = np.full((L, K_max), INVALID_BIAS, dtype=np.float32)
    gc_biases = np.zeros((L, K_max), dtype=np.float32)
    Ks = []

    for p, aa in enumerate(problem.amino_acid_seq):
        codons = SORTED_CODON_TABLE[aa]
        Ks.append(len(codons))
        for j, codon in enumerate(codons):
            rarity = problem.codon_rarity_scores.get(codon, 0.0)
            base_biases[p, j] = -problem.weight_codon_usage * rarity
            gc_biases[p, j] = gc_count(codon)

    biases = jnp.array(base_biases) + linear_gc_coeff * jnp.array(gc_biases)
    return biases, Ks


def compute_pairwise_penalties(problem: CodonProblem) -> np.ndarray:
    """Compute pairwise repeat penalties in GC-sorted ordering.

    The weight for pair (c1, c2) is: -c_R * r(c1, c2).
    The negative sign converts the penalty to a THRML weight.

    Args:
        problem: The codon optimization problem.

    Returns:
        Array of shape [L-1, K_max, K_max] with pairwise weights.
    """
    L = problem.seq_length
    K_max = MAX_SYNONYMOUS_CODONS

    pairwise = np.zeros((L - 1, K_max, K_max), dtype=np.float32)

    for i in range(L - 1):
        aa1 = problem.amino_acid_seq[i]
        aa2 = problem.amino_acid_seq[i + 1]
        codons1 = SORTED_CODON_TABLE[aa1]
        codons2 = SORTED_CODON_TABLE[aa2]

        for j, c1 in enumerate(codons1):
            for k, c2 in enumerate(codons2):
                penalty = repeat_penalty(c1, c2)
                pairwise[i, j, k] = -problem.weight_repeat * penalty

    return pairwise


def create_gc_counter(problem: CodonProblem):
    """Create a jitted function to compute total GC count from codon indices.

    Uses the GC-sorted codon ordering (SORTED_CODON_TABLE).

    Args:
        problem: The codon optimization problem.

    Returns:
        A jitted function: indices (L,) -> total GC count (scalar int).
    """
    L = problem.seq_length
    K_max = MAX_SYNONYMOUS_CODONS

    gc_lookup = np.zeros((L, K_max), dtype=np.int32)
    for i, aa in enumerate(problem.amino_acid_seq):
        codons = SORTED_CODON_TABLE[aa]
        for j, codon in enumerate(codons):
            gc_lookup[i, j] = gc_count(codon)

    gc_lookup_jax = jnp.array(gc_lookup)

    @jax.jit
    def compute_total_gc(indices):
        gc_per_position = gc_lookup_jax[jnp.arange(L), indices]
        return jnp.sum(gc_per_position)

    return compute_total_gc


# ---------------------------------------------------------------------------
# JAX energy / metrics functions
# ---------------------------------------------------------------------------

def _build_metrics_fn(problem: CodonProblem) -> Callable:
    """Build a function to compute energy metrics for a single codon index sequence.

    Uses the GC-sorted codon ordering (SORTED_CODON_TABLE).
    The returned function is NOT jitted — callers apply jit/vmap as needed.

    Args:
        problem: The codon optimization problem.

    Returns:
        A function: indices (L,) -> dict with 'total', 'usage', 'gc', 'repeat'.
    """
    L = problem.seq_length
    N = problem.nucleotide_length
    K_max = MAX_SYNONYMOUS_CODONS

    # Precompute usage costs
    usage_lookup = np.full((L, K_max), 0.0, dtype=np.float32)
    for i, aa in enumerate(problem.amino_acid_seq):
        codons = SORTED_CODON_TABLE[aa]
        for j, codon in enumerate(codons):
            rarity = problem.codon_rarity_scores.get(codon, 0.0)
            usage_lookup[i, j] = rarity
    usage_lookup_jax = jnp.array(usage_lookup)

    # Precompute GC counts
    gc_lookup = np.zeros((L, K_max), dtype=np.float32)
    for i, aa in enumerate(problem.amino_acid_seq):
        codons = SORTED_CODON_TABLE[aa]
        for j, codon in enumerate(codons):
            gc_lookup[i, j] = gc_count(codon)
    gc_lookup_jax = jnp.array(gc_lookup)

    # Precompute repeat penalties
    repeat_lookup = np.zeros((L - 1, K_max, K_max), dtype=np.float32)
    for i in range(L - 1):
        aa1 = problem.amino_acid_seq[i]
        aa2 = problem.amino_acid_seq[i + 1]
        codons1 = SORTED_CODON_TABLE[aa1]
        codons2 = SORTED_CODON_TABLE[aa2]
        for j, c1 in enumerate(codons1):
            for k, c2 in enumerate(codons2):
                repeat_lookup[i, j, k] = repeat_penalty(c1, c2)
    repeat_lookup_jax = jnp.array(repeat_lookup)

    target_gc = problem.target_gc_fraction
    weight_usage = problem.weight_codon_usage
    weight_gc = problem.weight_gc
    weight_repeat = problem.weight_repeat

    if target_gc is not None:
        def compute_gc_term(gc_frac):
            return weight_gc * (gc_frac - target_gc) ** 2
    else:
        def compute_gc_term(gc_frac):
            return jnp.array(0.0)

    def compute_metrics(indices):
        usage_costs = usage_lookup_jax[jnp.arange(L), indices]
        usage_term = weight_usage * jnp.sum(usage_costs)

        gc_counts = gc_lookup_jax[jnp.arange(L), indices]
        total_gc = jnp.sum(gc_counts)
        gc_frac = total_gc / N
        gc_term = compute_gc_term(gc_frac)

        pos_indices = jnp.arange(L - 1)
        repeat_penalties = repeat_lookup_jax[pos_indices, indices[:-1], indices[1:]]
        repeat_term = weight_repeat * jnp.sum(repeat_penalties)

        total = usage_term + gc_term + repeat_term

        return {"total": total, "usage": usage_term, "gc": gc_term, "repeat": repeat_term}

    return compute_metrics


def create_metrics_fn(problem: CodonProblem) -> Callable:
    """Create a jitted function to compute energy metrics for a single codon index sequence.

    Args:
        problem: The codon optimization problem.

    Returns:
        A jitted function: indices (L,) -> dict with 'total', 'usage', 'gc', 'repeat'.
    """
    return jax.jit(_build_metrics_fn(problem))


def create_energy_fn(problem: CodonProblem) -> Callable:
    """Create a jitted vmapped function to compute energies for batched samples.

    Args:
        problem: The codon optimization problem.

    Returns:
        A function that takes indices of shape (n_chains, n_samples, L) and
        returns a dict with 'total', 'usage', 'gc', 'repeat' arrays of shape
        (n_chains, n_samples).
    """
    return jax.jit(jax.vmap(jax.vmap(_build_metrics_fn(problem))))


# ---------------------------------------------------------------------------
# GC coefficient adaptation
# ---------------------------------------------------------------------------

def adapt_gc_coeff(gc_fraction, prev_coeff, target_gc_fraction, weight_gc, nucleotide_length, gc_coeff_adapt_mult):
    """Adapt linear GC coefficient based on GC fraction error.

    The coefficient is bounded to never exceed the bias that the true
    quadratic GC term would impose (gc_coeff_true). This prevents the
    linear GC approximation from overpowering the usage and repeat terms.

    All parameters except gc_fraction and prev_coeff may be Python scalars
    (treated as compile-time constants in JIT).

    Args:
        gc_fraction: Current GC fraction (JAX array, possibly batched).
        prev_coeff: Previous GC coefficient (JAX array, same shape).
        target_gc_fraction: Target GC fraction (Python float).
        weight_gc: Weight for GC content term (Python float).
        nucleotide_length: Total nucleotide length 3*L (Python int).
        gc_coeff_adapt_mult: Adaptation rate multiplier (JAX scalar).

    Returns:
        Updated GC coefficient (same shape as prev_coeff).
    """
    gc_error = gc_fraction - target_gc_fraction

    # Compute the "true" coefficient from the quadratic GC term's gradient.
    # The penalty is E_GC = c_GC * (gc_frac - target)^2, but THRML works with
    # weights (negative energy), so the effective bias is -dE/d(gc_count).
    gc_coeff_true = -2 * weight_gc * gc_error / nucleotide_length

    move = gc_error * gc_coeff_adapt_mult
    new_coeff = prev_coeff - move

    # Clamp to range [prev_coeff, gc_coeff_true] (in either order).
    lower = jnp.minimum(prev_coeff, gc_coeff_true)
    upper = jnp.maximum(prev_coeff, gc_coeff_true)
    new_coeff = jnp.clip(new_coeff, lower, upper)

    return new_coeff
