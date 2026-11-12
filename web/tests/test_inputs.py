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
from gnome3d_web.inputs import Own, build_inputs
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

        own_dir = Path(d) / "own_files"
        own_dir.mkdir()
        (own_dir / "a.bedpe").write_text("chr1\t100000\t101000\tchr1\t300000\t301000\t5\nchr1\t100500\t101500\tchr1\t500000\t501000\t3\n")
        (own_dir / "b.bedpe").write_text("chr1\t700000\t701000\tchr1\t900000\t901000\t4\n")
        (own_dir / "c.bedpe").write_text("chr1\t100000\t125000\tchr1\t300000\t325000\t5\n")
        own = Own(loops=[("CTCF", own_dir / "a.bedpe"), ("H3K27ac", own_dir / "b.bedpe")], contacts=own_dir / "c.bedpe")
        inp = build_inputs(ct, region, "own", [], Path(d) / "own", own=own)
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        data = cfg["data"]
        check("own loops of two factors write two files with the factors named", len(data["clusters"].split(",")) == 2 and data["factors"] == "CTCF,H3K27AC", dict(data).__repr__())
        anchors = rows(Path(data["data_dir"]) / data["anchors"])
        check("without an anchors upload the anchors are the merged loop ends with N", [(a[1], a[2], a[3]) for a in anchors] == [("100000", "101500", "N"), ("300000", "301000", "N"), ("500000", "501000", "N"), ("700000", "701000", "N"), ("900000", "901000", "N")], str(anchors))
        sing = rows(Path(data["data_dir"]) / "singletons.bedpe")
        check("own contacts replace the singletons", len(sing) == 1 and sing[0][6] == "5", str(sing))
        deep = "".join(f"chr1\t{a}\t{a + 25000}\tchr1\t{a + 50000}\t{a + 75000}\t500\n" for a in range(0, 2_900_000, 25_000))
        (own_dir / "deep.bedpe").write_text(deep)
        inp = build_inputs(ct, region, "own", [], Path(d) / "deep", own=Own(loops=own.loops, contacts=own_dir / "deep.bedpe"))
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        sing = rows(Path(cfg["data"]["data_dir"]) / "singletons.bedpe")
        total = sum(int(r[6]) for r in sing)
        hosted = 39 * 3 / 3.875
        check("deep own contacts are thinned to the hosted map's contacts per Mb", total == round(hosted * 2.95), f"{total} against {hosted * 2.95:.1f}")
        check("thinning keeps every pair at or under its count", all(0 < int(r[6]) <= 500 for r in sing), str(sing[:3]))
        (own_dir / "short.bedpe").write_text("chr1\t100000\t102000\tchr1\t104000\t106000\t50\n")
        inp = build_inputs(ct, region, "own", [], Path(d) / "short", own=Own(loops=own.loops, contacts=own_dir / "short.bedpe"))
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        sing = rows(Path(cfg["data"]["data_dir"]) / "singletons.bedpe")
        check("contacts over a span too short for one hosted contact keep at least one", len(sing) == 1 and int(sing[0][6]) >= 1, str(sing))
        own2 = Own(loops=[("H3K27ac", own_dir / "b.bedpe")])
        inp = build_inputs(ct, region, "own", [], Path(d) / "own2", own=own2)
        cfg = configparser.ConfigParser()
        cfg.read(inp.config)
        check("a first factor that is not CTCF turns the orientation term off", cfg["motif_orientation"]["use_motif_orientation"] == "no")

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
