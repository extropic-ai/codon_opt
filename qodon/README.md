# quantum_codon_opt

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

## USAGE:

The main module is contained in qodon.py, and can be run by:

```python
python qodon.py
```

The options for specifying sequences and backends are set in qodon.py.
The possible execution frameworks include:

* Classical genetic algorithm
* D-Wave exact solver, hybrid solver, and QPU solver
* IBM exact solver and noisy simulator

To access D-Wave and IBM resources, users are reponsible for installing
libraries and/or establishing device access. The easiest way to access
these resources is to create accounts on D-Waves Leap platform
and IBM's IBM Experience platform.

## API Calls
The execution frameworks can also be called directly. The API calls are
as follows

* D-Wave:
    * DWaveBQM(_sequence: str, hybrid: bool, exact: bool_)

* IBM
    * QiskitBQM(_sequence: str, exact: bool_)

* Classical Genetic Algorithm
    * CodonOptimization(_sequence: str_)

## Setting scoring parameters
The scoring constants are defined in constants.py. These values are 
referenced by all other scripts, including D-Wave, IBM, and classical
GA scoring algorithms.

## Requirements
The code was written so that it could be executed on both the Leap
Platform and the IBM Experience as long as the appropriate options
are set (i.e. D-Wave resources are not requested on the IBM platform
and vice-versa). That being said, the required proprietary D-Wave
libraries cannot be packaged into a conda library. In lieu of a
conda requirements.yaml, the dependencies are listed below

General:
numpy 1.19.4, pandas 1.1.4, Bio 1.78, python_codon_tables 0.1.10

D-Wave:
dimod 0.9.10, dwave (proprietary)

IBM:
qiskit 0.16.1
