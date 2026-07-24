"""Model-vs-Hi-C correlation. An invented test, beyond the 2016 paper, whose Fig. 2 is data-level
ChIA-PET to Hi-C, not model to Hi-C. For each of the three variants, reference, python parity,
and python tuned, we reconstruct a region, build the ensemble simulated contact map, and correlate
it against experimental Hi-C.

Scale coverage vs the paper's Fig. 2.
  * Scale B, intra-chromosomal at 1 Mb, the paper's Fig. 2B at ρ=0.67, is supported here.
  * Scale A, inter-chromosomal at chromosome level, Fig. 2A at ρ=0.73, is not possible with our
    data. The GM12878 ChIA-PET has no inter-chromosomal rows, checked at 0/265597 singletons and
    0/123274 clusters, so no whole-genome model can position chromosomes relative to each other.
    Reported as unavailable rather than faked.

Note this is a harder test than the paper's. It compares the reconstructed 3D structure's contacts
to Hi-C, not the input data to Hi-C. The paper validated models via reconstruction fidelity and
noise robustness, see validation/studies/synthetic.py, not a model-vs-Hi-C correlation.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from validation import metrics
from validation.core import variants
from validation.metrics import hic as contacts
from validation.studies import Context, Study, register

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
import integration as ig  # noqa: E402  (dev tooling; harness is not a package)


def _arrays(ens: list) -> tuple[list, list]:  # type: ignore[type-arg]
    cl, ml = [], []
    for beads in ens:
        c, m = metrics.to_arrays(beads)
        cl.append(c)
        ml.append(m)
    return cl, ml


def _radius(cl: list) -> float:  # type: ignore[type-arg]
    return float(np.median(metrics.bond_lengths(cl[0])))


class ModelHiC(Study):
    name = "model-hic"
    help = "model-vs-Hi-C correlation, reference/parity/tuned, scale B at 1Mb"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", default="chr3:1-30000000", help="intra-chr region, scale B")
        p.add_argument(
            "--binsize", type=int, default=1_000_000, help="Hi-C resolution, paper Fig.2B is 1Mb"
        )
        p.add_argument("--variants", default="reference,parity,tuned")

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        variant_names = args.variants.split(",")
        if "reference" in variant_names and not ig.CPP_BIN.exists():
            sys.exit(f"[error] reference binary not found: {ig.CPP_BIN}\n  run: make -C 3dnome")

        tmp = Path(tempfile.mkdtemp(prefix="mhic_"))
        cfg = variants.write_parity_ini(tmp, fast=ctx.fast)  # reference and parity base

        print(f"model-vs-Hi-C @ {args.binsize // 1000}kb  region {args.region}  n={ctx.n}")
        print("  paper Fig. 2B intra-chromosomal @ 1Mb has ρ=0.67, but that is data-level ChIA-PET↔Hi-C.")
        print("   this is the harder model structure↔Hi-C correlation\n")
        print(f"  {'variant':<16}{'Pearson(log1p)':>16}{'SCC':>9}")
        for v in variant_names:
            try:
                ens = variants.reconstruct(
                    v,
                    args.region,
                    cell=ctx.cell,
                    data_root=ctx.data_root,
                    quality=ctx.quality,
                    config=cfg,
                    data=None,
                    n=ctx.n,
                    py_arcs=ctx.py_arcs,
                    py_workers=ctx.py_workers,
                    ref_workers=ctx.ref_workers,
                    fast=ctx.fast,
                    label="mhic",
                )
                cl, ml = _arrays(ens)
                rad = _radius(cl)
                hc = contacts.ensemble_hic_correlation(cl, ml, ctx.hic, args.region, args.binsize, rad)
                print(f"  {v:<16}{hc['pearson']:>16.3f}{hc['scc']:>9.3f}")
            except Exception as e:  # noqa: BLE001
                print(f"  {v:<16}  ERROR: {e}")

        print("\n  Scale A, inter-chromosomal, paper Fig. 2A ρ=0.73, is not available. The GM12878 ChIA-PET")
        print("  has 0 inter-chromosomal contacts, so no whole-genome inter-chr model can be built.")
        print("  Pearson on log1p simulated against observed contacts, off-diagonal, ensemble-aggregated.")


register(ModelHiC())
