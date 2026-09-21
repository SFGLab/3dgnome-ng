"""Contact singletons for the trio pipeline from the HGSVC Hi-C of the same nine individuals.

The trio runs read a contact map built from the CTCF ChIA-PET read pairs, which holds nothing
the loops do not. The Human Genome Structural Variation Consortium sequenced Hi-C on exactly
these nine lymphoblastoid lines, two biological replicates each, GRCh38, and publishes
normalised 40 kb matrices per chromosome, one dense text matrix per file, at
ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/hgsv_sv_discovery/working/20160817_HiC_contact_matrices/.
This turns one such matrix into the singletons file the pipeline reads, one row per contact,
`chr s e chr s e 1`, by drawing contacts with probability proportional to the matrix value.
The draw is seeded and the number of contacts is given, so a run on this map can be matched in
depth to the run it replaces.

    python playground/trio/hgsvc_singletons.py <sample>.40Kb.nor.qq.chr1.mat chr1 1535847 out.bedpe
"""

import argparse
from pathlib import Path

import numpy as np

BIN = 40_000
SEED = 20260921


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matrix", type=Path)
    ap.add_argument("chrom")
    ap.add_argument("n_contacts", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    m = np.loadtxt(args.matrix, dtype=np.float64)
    if m.shape[0] != m.shape[1]:
        raise SystemExit(f"{args.matrix} is {m.shape}, not square")
    iu = np.triu_indices(m.shape[0])
    w = m[iu]
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    if w.sum() <= 0:
        raise SystemExit(f"{args.matrix} holds no positive contacts")
    p = w / w.sum()
    rng = np.random.default_rng(args.seed)
    counts = rng.multinomial(args.n_contacts, p)
    hit = counts > 0
    i, j, c = iu[0][hit], iu[1][hit], counts[hit]
    order = np.lexsort((j, i))
    with args.out.open("w") as fh:
        for a, b, k in zip(i[order], j[order], c[order]):
            row = f"{args.chrom}\t{a * BIN}\t{(a + 1) * BIN}\t{args.chrom}\t{b * BIN}\t{(b + 1) * BIN}\t1\n"
            fh.write(row * int(k))
    n_bins = int(hit.sum())
    print(f"[hgsvc] {args.matrix.name}: {m.shape[0]} bins, {args.n_contacts:,} contacts drawn over "
          f"{n_bins:,} bin pairs, {int((w > 0).sum()):,} bin pairs positive, wrote {args.out}")


if __name__ == "__main__":
    main()
