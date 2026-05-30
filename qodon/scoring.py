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
from itertools import groupby
from constants import *


class SeqScorer(object):
    def __init__(self, n_seq):
        self.n_seq = n_seq
        self.gc_constant = gc_constant
        self.repeat_constant = repeat_constant
        self.rarity_constant = rarity_constant
        self.execute()

    def __repr__(self):
        return 'Classical scoring function.'

    def execute(self):
        self._gc_score()
        self._repeat_score()
        self._rarity_score()
        self.score = self.gc_constant * self.gc_score + self.repeat_constant * self.rep_score + self.rarity_constant * self.rarity_score

    def _gc_score(self):
        self.gc_score = ((self.n_seq.count('C') + self.n_seq.count('G')) /
                         float(len(self.n_seq)) - 0.5)**2

    def _repeat_score(self):
        score = 0
        for _i, c in enumerate(list(self.n_seq)[:-3][::3]):
            ind = 3 * _i
            s = self.n_seq[ind:ind + 6]
            repeats = sorted([(letter, len(list(group)))
                              for letter, group in groupby(s)],
                             key=lambda i: i[1],
                             reverse=True)
            score += max(repeats, key=lambda i: i[1])[1]**2 - 1
        self.rep_score = score

    def _rarity_score(self):
        self.rarity_score = sum([
            codon_scores[self.n_seq[i:i + 3]]
            for i in range(len(self.n_seq))[::3]
        ])
