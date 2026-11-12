"""Engine inputs for one request.

    python web/tests/test_inputs.py
"""

from __future__ import annotations

import configparser
import sys
import tempfile
from pathlib import Path

from _check import check, done
from fixture import make_data
from personal3d.variants import SV

from gnome3d_web.catalog import Catalog
from gnome3d_web.inputs import build_inputs
from gnome3d_web.regions import Region


def rows(path: Path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text().splitlines() if ln.strip()]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        cat = Catalog.load(make_data(Path(d) / "data"))
        ct = cat.cell_types["GMX"]
        region = Region("chr1", 0, 3_000_000)

        inp = build_inputs(ct, region, "CTCF", [], Path(d) / "plain")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        data = cfg["data"]
        check("a CTCF run reads one loop file and names no factors", "," not in data["clusters"] and "factors" not in data)
        loops = rows(Path(data["data_dir"]) / data["clusters"])
        check("the loops are those inside the region", len(loops) == 14, str(len(loops)))
        check("the region is unchanged without variants", inp.region == "chr1:0-3000000")
        check("the rest of the config is the cell type's", cfg["main"]["max_pet_length"] == "1000000")

        inp = build_inputs(ct, region, "CTCF+RNAPII", [], Path(d) / "both")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        check(
            "a CTCF and RNAPII run reads two loop files, CTCF first",
            cfg["data"]["factors"] == "CTCF,RNAPOL2" and cfg["data"]["clusters"].split(",")[0].endswith("ctcf_clusters.bedpe"),
        )

        inp = build_inputs(ct, region, "RNAPII", [], Path(d) / "rnapii")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        data = cfg["data"]
        check("an RNAPII run reads the RNAPII loops alone", data["clusters"] == "rnapii_clusters.bedpe" and data["factors"] == "RNAPOL2", dict(data).__repr__())
        anchors = rows(Path(data["data_dir"]) / data["anchors"])
        check("its anchors are the RNAPII loop ends", [(a[1], a[2]) for a in anchors] == [("1300000", "1301000"), ("1500000", "1501000")], str(anchors))
        check("every anchor has orientation N", all(a[3] == "N" for a in anchors), str(anchors))
        check("the CTCF orientation term is off", cfg["motif_orientation"]["use_motif_orientation"] == "no")

        inp = build_inputs(ct, region, "CTCF", [], Path(d) / "none", background="none")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        check("background none turns both contact terms off", cfg["springs"]["use_contact_background"] == "no" and cfg["anchor_heatmap"]["use_anchor_heatmap"] == "no")
        check("background none keeps the singletons for the exponent fit", (Path(cfg["data"]["data_dir"]) / "singletons.bedpe").stat().st_size > 0)
        inp = build_inputs(ct, region, "CTCF", [], Path(d) / "hic", background="hic")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        check("background hic leaves the config as the cell type has it", not cfg.has_section("springs") and not cfg.has_section("anchor_heatmap"))

        dele = SV("chr1", 900_001, 960_001, "DEL")
        inp = build_inputs(ct, region, "CTCF", [dele], Path(d) / "del")
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        anchors = rows(Path(cfg["data"]["data_dir"]) / cfg["data"]["anchors"])
        starts = sorted(int(a[1]) for a in anchors)
        check("a deletion shifts the anchors after it by its length", 1_040_000 in starts and 1_100_000 not in starts, str(starts[:8]))
        check("the region shrinks by the deletion", inp.region == "chr1:0-2940000", inp.region)
        check("the variants applied are counted", inp.n_svs == 1)

        # On HG00512's first haplotype over chr1:1-3 Mb the rules' own cutoff of 0 turned 54 CTCF
        # loops into 101, 46 of them on 19 bp anchors at weak motifs inside inserted repeats;
        # triosformer's cutoff of 10 leaves 54 and one new anchor.
        from personal3d import motifs, rewrite

        seen: list[float] = []
        real = rewrite.apply

        def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            seen.append(kwargs.get("motif_cut", args[6] if len(args) > 6 else 0.0))
            return real(*args, **kwargs)

        rewrite.apply = spy
        try:
            build_inputs(ct, region, "CTCF", [dele], Path(d) / "spy")
        finally:
            rewrite.apply = real
        check("the rules run with triosformer's motif cutoff", seen == [motifs.THRESHOLD], str(seen))
    return done()


if __name__ == "__main__":
    sys.exit(main())
