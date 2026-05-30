#!/usr/bin/env python3
"""Run the genetic algorithm on the spike protein sequence and report results."""

import time
from classical_ga import CodonOptimization
from scoring import SeqScorer
from constants import gc_constant, repeat_constant, rarity_constant

# Spike protein sequence (from codons/script_utils.py)
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
    "IAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGI"
    "AVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTL"
    "ADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWT"
    "FGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALG"
    "KLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQ"
    "TYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLH"
    "VTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVS"
    "GNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKE"
    "IDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCS"
    "CLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
).replace(" ", "")


def main():
    print("=" * 70)
    print("Running Genetic Algorithm on Spike Protein Sequence")
    print("=" * 70)
    print(f"\nSequence length: {len(SPIKE_PROTEIN_SEQ)} amino acids")
    print(f"Nucleotide length: {3 * len(SPIKE_PROTEIN_SEQ)} bp")
    print(f"\nGA parameters:")
    print(f"  elitelist: 10")
    print(f"  randomlist: 2")
    print(f"  ntrials: 50")
    print(f"  numgens: 100")
    print(f"\nScoring weights:")
    print(f"  gc_constant: {gc_constant}")
    print(f"  repeat_constant: {repeat_constant}")
    print(f"  rarity_constant: {rarity_constant}")

    print("\nRunning GA...")
    start_time = time.time()

    result = CodonOptimization(SPIKE_PROTEIN_SEQ)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")

    # Get detailed scoring
    scorer = SeqScorer(result.n_seq)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # GC content
    gc_count = result.n_seq.count('G') + result.n_seq.count('C')
    gc_fraction = gc_count / len(result.n_seq)
    print(f"\nGC Content:")
    print(f"  GC count: {gc_count} / {len(result.n_seq)}")
    print(f"  GC fraction: {gc_fraction:.4f}")
    print(f"  GC deviation from 0.5: {gc_fraction - 0.5:.4f}")

    print(f"\nScore breakdown (raw values before weighting):")
    print(f"  GC score (deviation^2):      {scorer.gc_score:.6f}")
    print(f"  Repeat score:                {scorer.rep_score:.1f}")
    print(f"  Rarity score:                {scorer.rarity_score:.4f}")

    print(f"\nWeighted score terms:")
    print(f"  GC term ({gc_constant} * {scorer.gc_score:.6f}):      {gc_constant * scorer.gc_score:.6f}")
    print(f"  Repeat term ({repeat_constant} * {scorer.rep_score:.1f}):    {repeat_constant * scorer.rep_score:.4f}")
    print(f"  Rarity term ({rarity_constant} * {scorer.rarity_score:.4f}):  {rarity_constant * scorer.rarity_score:.4f}")

    print(f"\nTotal score: {scorer.score:.6f}")

    # Also print the first 100 nucleotides of the optimized sequence
    print(f"\nOptimized nucleotide sequence (first 120 bp):")
    print(f"  {result.n_seq[:120]}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
