"""The one reference/parity/tuned reconstruction path the studies share.

reference runs the 3dnome reference binary. parity runs the python port with feature flags off.
tuned runs the unified canonical config. Every study reconstructs through here."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

from gnome3d.data import ContactData
from gnome3d.settings import Settings
from validation.core.config import settings_for_cell, with_arcs_executor
from validation.core.ensemble import run_ensemble
from validation.core.regions import parse_region_arg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
import integration as ig  # noqa: E402

Variant = Literal["reference", "parity", "tuned"]
MAX_LEVEL = 2
FEATURES_OFF: dict[str, object] = {
    "use_excluded_volume": False,
    "use_confinement": False,
    "use_dynamic_loop_density": False,
    "use_ib_mc": False,
}


def write_parity_ini(tmp: Path, fast: bool = False) -> Path:
    """Write the parity.ini base the reference binary and the parity variant both read."""
    cfg = tmp / "parity.ini"
    ig.write_config(cfg, fast=fast)
    return cfg


def tuned_settings(cell, data_root, quality, py_arcs="batch", py_workers=0) -> Settings:
    s = settings_for_cell(cell, data_root, quality)
    return with_arcs_executor(s, py_arcs, py_workers)


def parity_settings(config_path, py_arcs="batch", py_workers=0) -> Settings:
    s = Settings()
    s.load_ini(str(config_path))
    return with_arcs_executor(s, py_arcs, py_workers)


def reconstruct(variant, region, *, cell, data_root, quality, config, data=None,
                n, py_arcs="batch", py_workers=0, ref_workers=0, fast=False, label="v"):
    """Reconstruct n structures for one variant on one region."""
    chrs_list, bed = parse_region_arg(region)
    if variant == "reference":
        rdir = Path(tempfile.mkdtemp(prefix="ref_"))
        ens, _ = ig.run_cpp_ensemble_parallel(rdir, config, n, MAX_LEVEL, region, label, workers=ref_workers)
        return ens
    if variant == "parity":
        s = parity_settings(config, py_arcs, py_workers)
    else:
        s = tuned_settings(cell, data_root, quality, py_arcs, py_workers)
    d = data if data is not None else ContactData.from_files(s, chrs_list, bed)
    return run_ensemble(s, d, chrs_list, bed, n)


def reconstruct_all(variants, region, ctx) -> dict:
    """Reconstruct each requested variant on one region. ctx carries the shared knobs."""
    out = {}
    for v in variants:
        out[v] = reconstruct(
            v, region, cell=ctx.cell, data_root=ctx.data_root, quality=ctx.quality,
            config=ctx.config, data=(ctx.data if v != "reference" else None), n=ctx.n,
            py_arcs=ctx.py_arcs, py_workers=ctx.py_workers, ref_workers=ctx.ref_workers,
            fast=ctx.fast, label=ctx.label,
        )
    return out


def run_reference(config, region, n, *, ref_workers=0, label="v"):
    """Run the reference binary on one region and return its ensemble of bead lists.

    Shared by studies that build their own reference .ini for custom data, so the dispatch to
    run_cpp_ensemble_parallel stays in one place even when the config is not the parity.ini base."""
    rdir = Path(tempfile.mkdtemp(prefix="ref_"))
    ens, _ = ig.run_cpp_ensemble_parallel(rdir, config, n, MAX_LEVEL, region, label, workers=ref_workers)
    return ens
