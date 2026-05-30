"""Ising model (Domain Wall Encoding) for codon optimization."""

from .dwc import (
    compute_ising_biases,
    compute_ising_couplings,
    compute_spin_layout,
    potts_to_spin,
    spin_to_potts,
)
from .ising_model import (
    build_ising_program,
    create_ising_codon_model,
    prepare_ising_statics,
    run_ising_annealing,
)

__all__ = [
    "compute_ising_biases",
    "compute_ising_couplings",
    "compute_spin_layout",
    "potts_to_spin",
    "spin_to_potts",
    "build_ising_program",
    "create_ising_codon_model",
    "prepare_ising_statics",
    "run_ising_annealing",
]
