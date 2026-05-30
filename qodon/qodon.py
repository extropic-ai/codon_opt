'''

    mRNA Codon Optimization with Quantum Computers 
    Copyright (C) 2021  Dillion M. Fox, Ross C. Walker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

from classical_ga import CodonOptimization
from codon_bqm import DWaveBQM, QiskitBQM, ibm_score
from constants import *
from Bio import SeqIO
import numpy as np

# In order to use D-Wave or IBM, you must have access to appropriate
# libraries/devices.
use_dwave = False
use_ibm = False
use_ga = True

# Path to fasta
fasta_path = 'covid_sequences.fasta'
seq_list = [str(bio_seq.seq) for bio_seq in SeqIO.parse(fasta_path, 'fasta')]
seq_name = [str(bio_seq.id) for bio_seq in SeqIO.parse(fasta_path, 'fasta')]

for p_seq, seq_name in zip(seq_list, seq_name):

    if use_dwave:
        # Must have D-Wave libraries installed. If using hybrid or QPU methods,
        # must run code on Leap interface with appropriate resource access.
        # Quantum code is programmed to run n_execs times
        dwave_results = DWaveBQM(p_seq, target_score, exact=True, hybrid=False)
        print(dwave_results.score, dwave_results.score_mean, dwave_results.score_std)

    if use_ibm:
        # Must have qiskit installed or run code on IBM Experience.
        # QiskitBQM will run 1 time.
        ibm_results = QiskitBQM(p_seq, target_score)
        print(ibm_results.score, ibm_results.exact_score)

    if use_ga:
        # Run classical GA n_execs times
        c_scores = [CodonOptimization(p_seq).score for _ in range(n_execs)]
        print(f"{seq_name}: {min(c_scores),np.mean(c_scores), np.std(c_scores)}")

