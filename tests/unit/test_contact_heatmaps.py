"""Unit checks for the singleton contact heatmaps."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg
from gnome3d.pipeline.coarse import build_state
from gnome3d.pipeline.coarse.build import build_contact_heatmaps
from gnome3d.settings import Settings


def test_the_region_has_blocks_to_test_on_blocks() -> None:
    s = Settings()
    s.load_ini("data/GM12878/config.ini")
    s.data_dir = "data/GM12878"
    chrs, region = parse_chrs_arg("chr1:18288319-20307135")
    data = ContactData.from_files(s, chrs, region)
    state = build_state(s, data, chrs, region)
    ibs = [c for c in state.clusters if int(c.level) == 3 and len(c.children) > 1]
    assert len(ibs) > 0, f"the region has blocks to test on: {len(ibs)} blocks"
    for k, ib in enumerate(ibs[:3]):
        active = list(ib.children)
        full_anchor, sub = build_contact_heatmaps(state, active, "chr1")
        direct_anchor, none = build_contact_heatmaps(state, active, "chr1", with_subanchor=False)
        assert full_anchor.shape == direct_anchor.shape and np.array_equal(
            full_anchor, direct_anchor
        ), (
            f"block {k}: the direct anchor heatmap is the cut out one, exactly"
            + ": "
            + str(
                f"{full_anchor.shape}, max abs diff {np.abs(full_anchor - direct_anchor).max():.3g}"
                if full_anchor.shape == direct_anchor.shape
                else f"{full_anchor.shape} vs {direct_anchor.shape}"
            )
        )
        assert none is None and sub is not None, (
            f"block {k}: no subanchor matrix without the heat term"
        )
