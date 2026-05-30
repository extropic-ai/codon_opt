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
from collections import Counter, OrderedDict
from itertools import groupby, combinations
from constants import *
import numpy as np


class OneBodyPenaltyMatrix(object):
    def __init__(self, p_seq, target_score=0.5):
        self.p_seq = p_seq
        self.target_score = target_score
        self.gc_constant = gc_constant
        self.rarity_constant = rarity_constant
        self._check()
        self._build_matrix()

    def __repr__(self):
        return '''
        One-Body scoring function for BQM.
        
                 (1)         (2)          (3)
        h_i = r_i*c_i - 2*c_cg*p*s_i + c_cg*s_i^2 - E
        
        '''

    def _build_matrix(self):
        '''
        Compute the values of 'h' in the BQM.
        
        '''

        # Write out all possible codons in flattened list
        codons = [__ for _ in codon_table.values() for __ in _]

        # Create penalty dictionary for all codons
        gc_penalties = {i: (i.count('G') + i.count('C')) / 3.0 for i in codons}

        # Assign penalties to each codon in sequence
        s = [gc_penalties[c] for a in self.p_seq for c in codon_table[a]]
        self.s = s

        # Look up codon scores (penalize rare codons)
        c = [codon_scores[_] for a in self.p_seq for _ in codon_table[a]]
        N = float(len(self.p_seq))

        # Compute terms in sum. Maps to expression in __repr__
        self.term_1 = np.array([self.rarity_constant * _ for _ in c])
        self.term_2 = np.array(
            [-2 * self.target_score * self.gc_constant * _ / N for _ in s])
        self.term_3 = np.array([self.gc_constant * _ * _ / (N**2) for _ in s])

        self.matrix = self.term_1 + self.term_2 + self.term_3 - combo_factor

    @property
    def dwave_format(self):
        '''
        Convert vector to dictionary to submit to QPU
        
        '''

        return {
            ind: val
            for ind, val in zip(range(len(self.matrix)), self.matrix)
        }

    def _check(self):

        allowed_characters = 'ACDEFGHIKLMNPQRSTVWY'
        if any(x not in allowed_characters for x in self.p_seq):
            raise ValueError('''
            Invalid character detected in input sequence
            
            ''')

        if not isinstance(self.target_score, float):
            raise TypeError('''
            Target score must be a float between 0 and 1.
            
            ''')

        if not 0 <= self.target_score <= 1.0:
            raise ValueError('''
            Target score must be a float between 0 and 1.
            
            ''')


class TwoBodyPenaltyMatrix(object):
    def __init__(self, p_seq):
        self.p_seq = p_seq
        self.gc_constant = gc_constant
        self.repeat_constant = repeat_constant
        self._check()
        self._build_matrix()

    def __repr__(self):
        return '''
        Two-body scoring terms for BQM.
        
               (1)            (2)         (3)
        J = 2*gc*sigma + c_R*r*delta + deltaprime
        
        '''

    def _build_matrix(self):
        '''
        Compute values in Matrix J.
        
        '''

        # Write out all possible codons in order of sequence
        self._codon_code = [c for a in self.p_seq for c in codon_table[a]]

        # Precompute codon positions relative to aa positions
        self._codon_pos = list(
            zip([
                it_a for it_a, a in enumerate(self.p_seq)
                for c in codon_table[a]
            ], range(len(self._codon_code))))

        # Compute penalties
        self._compute_gc_penalties()
        self._compute_repeat_penalties()
        self._compute_deltaprime_penalties()

        N = float(len(self.p_seq))

        # Compute the terms in the sum
        self.term_1 = 2 * self.gc_constant * self._gc_tensor / (N**2)
        self.term_2 = self.repeat_constant * self.r_tensor
        self.term_3 = self.dp_tensor

        # Compute total 2-body interaction matrix
        self.matrix = self.term_1 + self.term_2 + self.term_3

    def _compute_gc_penalties(self):
        '''
        Compute the rank-2 tensor that comes from the calculation
        of GC-Content.
        
        '''

        # Retrieve (recompute) vector s and convert to numpy array
        s = np.array(OneBodyPenaltyMatrix(self.p_seq).s)

        # Compute outer product
        self._gc_tensor = np.outer(s, s)

    def _compute_repeat_penalties(self):
        '''
        Assign penalties to maximum number of repeated nucleotides
        in sequential codons.

        '''

        # Identify positions we want to penalize for too many repeated nucleotides
        self._twobody_penalty_terms()
        self.r_tensor = np.zeros(
            (len(self._codon_code), len(self._codon_code)))

        for i in range(len(self._codon_code)):
            for j in range(len(self._codon_code)):
                if (j, i) not in self._twobody_penalty_pairs: continue
                s = self._codon_code[j] + self._codon_code[i]
                repeats = sorted([(letter, len(list(group)))
                                  for letter, group in groupby(s)],
                                 key=lambda i: i[1],
                                 reverse=True)
                self.r_tensor[i][j] = (max(repeats, key=lambda i: i[1])[1]**2-1)

    def _compute_deltaprime_penalties(self):
        '''
        Set all position-conflicting codons to high values

        '''

        pos = 0
        self.dp_tensor = np.zeros(
            (len(self._codon_code), len(self._codon_code)))
        for aa in self.p_seq:
            n_co = len(codon_table[aa])
            for i in range(n_co):
                for j in range(n_co):
                    if i != j:
                        self.dp_tensor[pos + i][pos + j] = combo_penalty
            pos += n_co

    @property
    def dwave_format(self):
        '''
        Convert matrix to dictionary to ship to QPU.
        
        '''
        return {(i, j): self.matrix[j][i]
                for i in range(self.matrix.shape[1])
                for j in range(self.matrix.shape[0])
                if self.matrix[j][i] != 0 and i < j}

    def _check(self):
        allowed_characters = 'ACDEFGHIKLMNPQRSTVWY'
        if any(x not in allowed_characters for x in self.p_seq):
            raise ValueError('''
            Invalid character detected in input sequence
            
            ''')

    def _get_codon_pos(self, aa_pos):
        '''
        Map positions in amino acid sequence to positions
        in codon array.
        
        '''

        return [
            _[1] for _ in filter(lambda x: x[0] == aa_pos, self._codon_pos)
        ]

    def _twobody_penalty_terms(self):
        '''
        Precompute all positions that we want to compute two-body interactions
        for. Only counting repeated nucleotides, so only need to consider
        neighboring codons. However, need to consider all possible combinations
        of codons when filling out matrix.

        '''

        int_terms = []
        for pos in range(1, len(self.p_seq)):
            _prev = self._get_codon_pos(pos - 1)
            _curr = self._get_codon_pos(pos)
            _next = self._get_codon_pos(pos + 1)
            int_terms += [(_, __) for _ in _prev for __ in _curr]
            int_terms += [(_, __) for _ in _curr for __ in _next]
        self._twobody_penalty_pairs = sorted(list(set(int_terms)))
