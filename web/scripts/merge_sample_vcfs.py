"""Merges a directory of single sample VCFs of one call set into one multi sample VCF.

    python merge_sample_vcfs.py <directory> <out.vcf.gz>

The directory holds `<sample>/<sample>.vcf`. Two records are one site when chromosome,
position, ID, REF and ALT agree. A sample without a site gets 0|0. The output is sorted,
compressed with bgzip and indexed with tabix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    samples = sorted(d.name for d in src.iterdir() if (d / f"{d.name}.vcf").is_file())
    n = len(samples)
    header: list[str] = []
    contigs: dict[str, int] = {}
    sites: dict[tuple[str, int, str, str, str], int] = {}
    fixed: list[tuple[str, str, str]] = []
    calls: list[list[str]] = []
    for i, s in enumerate(samples):
        with (src / s / f"{s}.vcf").open() as fh:
            for line in fh:
                if line.startswith("##"):
                    if i == 0:
                        header.append(line)
                        if line.startswith("##contig=<ID="):
                            contigs[line[13:].split(",", 1)[0].rstrip(">\n")] = len(contigs)
                    continue
                if line.startswith("#"):
                    continue
                x = line.rstrip("\n").split("\t")
                key = (x[0], int(x[1]), x[2], x[3], x[4])
                j = sites.get(key)
                if j is None:
                    j = sites[key] = len(fixed)
                    fixed.append((x[5], x[6], x[7]))
                    calls.append(["0|0"] * n)
                calls[j][i] = x[9].split(":", 1)[0]
        if (i + 1) % 200 == 0:
            print(f"  {i + 1} of {n} samples, {len(fixed):,} sites", flush=True)
    order = sorted(sites, key=lambda k: (contigs.get(k[0], len(contigs)), k[1], k[2]))
    with out.open("wb") as raw:
        z = subprocess.Popen(["bgzip", "-c"], stdin=subprocess.PIPE, stdout=raw)
        assert z.stdin is not None
        z.stdin.write("".join(header).encode())
        z.stdin.write(f"##merged_from={n} single-sample files, a site a sample does not list is 0|0\n".encode())
        z.stdin.write(("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples) + "\n").encode())
        for k in order:
            j = sites[k]
            q, f, info = fixed[j]
            z.stdin.write(("\t".join((k[0], str(k[1]), k[2], k[3], k[4], q, f, info, "GT", *calls[j])) + "\n").encode())
        z.stdin.close()
        if z.wait() != 0:
            raise SystemExit("bgzip failed")
    subprocess.run(["tabix", "-f", "-p", "vcf", str(out)], check=True)
    print(f"{out}: {n} samples, {len(order):,} sites")


if __name__ == "__main__":
    main()
