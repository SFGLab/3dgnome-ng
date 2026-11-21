"""Engine inputs for one request.

    pytest web/tests/test_inputs.py
"""

from __future__ import annotations

import configparser
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data

from gnome3d_web.catalog.model import Catalog
from gnome3d_web.engine.inputs import Own, build_inputs, loop_network
from gnome3d_web.regions import Region
from personal3d.variants import SV


def rows(path: Path) -> list[list[str]]:
    return [ln.split("\t") for ln in path.read_text().splitlines() if ln.strip()]


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
    cat = Catalog.load(make_data(Path(d) / "data"))
    ct = cat.cell_types["GMX"]
    region = Region("chr1", 0, 3_000_000)

    inp = build_inputs(ct, region, "CTCF", [], Path(d) / "plain")
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    data = cfg["data"]
    return SimpleNamespace(data=data, cfg=cfg, ct=ct, region=region, d=d, inp=inp)


def test_a_ctcf_run_reads_one_loop_file_and_names_no_factors(shared: SimpleNamespace) -> None:
    data = shared.data
    assert "," not in data["clusters"] and "factors" not in data, "a CTCF run reads one loop file and names no factors"


def test_the_cell_type_s_compartments_of_the_chromosome_are_an_input(shared: SimpleNamespace) -> None:
    data, cfg = shared.data, shared.cfg
    comp = rows(Path(data["data_dir"]) / data.get("compartments", "none"))
    assert [c[1:] for c in comp] == [["0", "1000000", "A"], ["1000000", "2500000", "B"], ["2500000", "4000000", "A"]], "the cell type's compartments of the chromosome are an input: " + str(comp)
    assert not cfg.has_section("compartments"), "the term stays off unless the request turns it on"


def test_a_cell_type_without_compartments_names_none(shared: SimpleNamespace) -> None:
    ct, region, d = shared.ct, shared.region, shared.d
    bare = build_inputs(replace(ct, files={k: v for k, v in ct.files.items() if k != "compartments"}), region, "CTCF", [], Path(d) / "bare")
    cfg_bare = configparser.ConfigParser()
    cfg_bare.read(bare.config)
    assert "compartments" not in cfg_bare["data"], "a cell type without compartments names none"


def test_the_loops_are_those_inside_the_region(shared: SimpleNamespace) -> None:
    data, inp, cfg = shared.data, shared.inp, shared.cfg
    loops = rows(Path(data["data_dir"]) / data["clusters"])
    assert len(loops) == 14, "the loops are those inside the region: " + str(len(loops))
    assert inp.region == "chr1:0-3000000", "the region is unchanged without variants"
    assert cfg["main"]["max_pet_length"] == "1000000", "the rest of the config is the cell type's"


def test_a_ctcf_and_rnapii_run_reads_two_loop_files_ctcf_first(shared: SimpleNamespace) -> None:
    ct, region, d = shared.ct, shared.region, shared.d
    inp = build_inputs(ct, region, "CTCF+RNAPII", [], Path(d) / "both")
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    assert cfg["data"]["factors"] == "CTCF,RNAPOL2" and cfg["data"]["clusters"].split(",")[0].endswith("ctcf_clusters.bedpe"), "a CTCF and RNAPII run reads two loop files, CTCF first"


def test_an_rnapii_run_reads_the_rnapii_loops_alone(shared: SimpleNamespace) -> None:
    ct, region, d = shared.ct, shared.region, shared.d
    inp = build_inputs(ct, region, "RNAPII", [], Path(d) / "rnapii")
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    data = cfg["data"]
    assert data["clusters"] == "rnapii_clusters.bedpe" and data["factors"] == "RNAPOL2", "an RNAPII run reads the RNAPII loops alone: " + str(dict(data).__repr__())
    anchors = rows(Path(data["data_dir"]) / data["anchors"])
    assert [(a[1], a[2]) for a in anchors] == [("1300000", "1301000"), ("1500000", "1501000")], "its anchors are the RNAPII loop ends: " + str(anchors)
    assert all(a[3] == "N" for a in anchors), "every anchor has orientation N: " + str(anchors)
    assert cfg["motif_orientation"]["use_motif_orientation"] == "no", "the CTCF orientation term is off"


def test_background_none_turns_both_contact_terms_off(shared: SimpleNamespace) -> None:
    ct, region, d = shared.ct, shared.region, shared.d
    inp = build_inputs(ct, region, "CTCF", [], Path(d) / "none", background="none")
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    assert cfg["springs"]["use_contact_background"] == "no" and cfg["anchor_heatmap"]["use_anchor_heatmap"] == "no", "background none turns both contact terms off"
    assert (Path(cfg["data"]["data_dir"]) / "singletons.bedpe").stat().st_size > 0, "background none keeps the singletons for the exponent fit"
    inp = build_inputs(ct, region, "CTCF", [], Path(d) / "hic", background="hic")
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    assert not cfg.has_section("springs") and not cfg.has_section("anchor_heatmap"), "background hic leaves the config as the cell type has it"


@pytest.fixture(scope="module")
def data_and_own_dir_and_own(shared: SimpleNamespace) -> SimpleNamespace:
    d, ct, region = shared.d, shared.ct, shared.region
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
    return SimpleNamespace(data=data, own_dir=own_dir, own=own)


def test_own_loops_of_two_factors_write_two_files_with_the_factors_named(data_and_own_dir_and_own: SimpleNamespace) -> None:
    data = data_and_own_dir_and_own.data
    assert len(data["clusters"].split(",")) == 2 and data["factors"] == "CTCF,H3K27AC", "own loops of two factors write two files with the factors named: " + str(dict(data).__repr__())


def test_without_an_anchors_upload_the_anchors_are_the_merged_loop_ends_with_n(data_and_own_dir_and_own: SimpleNamespace) -> None:
    data = data_and_own_dir_and_own.data
    anchors = rows(Path(data["data_dir"]) / data["anchors"])
    assert [(a[1], a[2], a[3]) for a in anchors] == [("100000", "101500", "N"), ("300000", "301000", "N"), ("500000", "501000", "N"), ("700000", "701000", "N"), ("900000", "901000", "N")], "without an anchors upload the anchors are the merged loop ends with N: " + str(anchors)


def test_own_contacts_replace_the_singletons(data_and_own_dir_and_own: SimpleNamespace) -> None:
    data = data_and_own_dir_and_own.data
    sing = rows(Path(data["data_dir"]) / "singletons.bedpe")
    assert len(sing) == 1 and sing[0][6] == "5", "own contacts replace the singletons: " + str(sing)
    assert "compartments" not in data, "own contacts are not read with the cell type's compartments"


def test_deep_own_contacts_are_thinned_to_the_hosted_map_s_contacts_per_mb(data_and_own_dir_and_own: SimpleNamespace, shared: SimpleNamespace) -> None:
    own_dir, own = data_and_own_dir_and_own.own_dir, data_and_own_dir_and_own.own
    ct, region, d = shared.ct, shared.region, shared.d
    deep = "".join(f"chr1\t{a}\t{a + 25000}\tchr1\t{a + 50000}\t{a + 75000}\t500\n" for a in range(0, 2_900_000, 25_000))
    (own_dir / "deep.bedpe").write_text(deep)
    inp = build_inputs(ct, region, "own", [], Path(d) / "deep", own=Own(loops=own.loops, contacts=own_dir / "deep.bedpe"))
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    sing = rows(Path(cfg["data"]["data_dir"]) / "singletons.bedpe")
    total = sum(int(r[6]) for r in sing)
    hosted = 39 * 3 / 3.875
    assert total == round(hosted * 2.95), "deep own contacts are thinned to the hosted map's contacts per Mb: " + f"{total} against {hosted * 2.95:.1f}"
    assert all(0 < int(r[6]) <= 500 for r in sing), "thinning keeps every pair at or under its count: " + str(sing[:3])


def test_contacts_over_a_span_too_short_for_one_hosted_contact_keep_at_least(data_and_own_dir_and_own: SimpleNamespace, shared: SimpleNamespace) -> None:
    own_dir, own = data_and_own_dir_and_own.own_dir, data_and_own_dir_and_own.own
    ct, region, d = shared.ct, shared.region, shared.d
    (own_dir / "short.bedpe").write_text("chr1\t100000\t102000\tchr1\t104000\t106000\t50\n")
    inp = build_inputs(ct, region, "own", [], Path(d) / "short", own=Own(loops=own.loops, contacts=own_dir / "short.bedpe"))
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    sing = rows(Path(cfg["data"]["data_dir"]) / "singletons.bedpe")
    assert len(sing) == 1 and int(sing[0][6]) >= 1, "contacts over a span too short for one hosted contact keep at least one: " + str(sing)


def test_a_first_factor_that_is_not_ctcf_turns_the_orientation_term_off(data_and_own_dir_and_own: SimpleNamespace, shared: SimpleNamespace) -> None:
    own_dir = data_and_own_dir_and_own.own_dir
    ct, region, d = shared.ct, shared.region, shared.d
    own2 = Own(loops=[("H3K27ac", own_dir / "b.bedpe")])
    inp = build_inputs(ct, region, "own", [], Path(d) / "own2", own=own2)
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    assert cfg["motif_orientation"]["use_motif_orientation"] == "no", "a first factor that is not CTCF turns the orientation term off"


@pytest.fixture(scope="module")
def starts_and_inp_and_cfg(shared: SimpleNamespace) -> SimpleNamespace:
    ct, region, d = shared.ct, shared.region, shared.d
    dele = SV("chr1", 900_001, 960_001, "DEL")
    inp = build_inputs(ct, region, "CTCF", [dele], Path(d) / "del")
    cfg = configparser.ConfigParser()
    cfg.read(inp.config)
    anchors = rows(Path(cfg["data"]["data_dir"]) / cfg["data"]["anchors"])
    starts = sorted(int(a[1]) for a in anchors)
    return SimpleNamespace(starts=starts, inp=inp, cfg=cfg, dele=dele)


def test_a_deletion_shifts_the_anchors_after_it_by_its_length(starts_and_inp_and_cfg: SimpleNamespace, shared: SimpleNamespace) -> None:
    starts, inp, cfg = starts_and_inp_and_cfg.starts, starts_and_inp_and_cfg.inp, starts_and_inp_and_cfg.cfg
    ct, region, d = shared.ct, shared.region, shared.d
    assert 1_040_000 in starts and 1_100_000 not in starts, "a deletion shifts the anchors after it by its length: " + str(starts[:8])
    assert inp.region == "chr1:0-2940000", "the region shrinks by the deletion: " + str(inp.region)
    assert inp.n_svs == 1, "the variants applied are counted"
    comp = rows(Path(cfg["data"]["data_dir"]) / cfg["data"]["compartments"])
    assert [c[1:] for c in comp] == [["0", "940000", "A"], ["940000", "2440000", "B"], ["2440000", "3940000", "A"]], "the compartments move with the variants and stay joined: " + str(comp)
    inside = build_inputs(ct, region, "CTCF", [SV("chr1", 950_001, 1_050_001, "DEL")], Path(d) / "del2")
    cfg2 = configparser.ConfigParser()
    cfg2.read(inside.config)
    comp = rows(Path(cfg2["data"]["data_dir"]) / cfg2["data"]["compartments"])
    assert [c[1:] for c in comp] == [["0", "950000", "A"], ["950000", "2400000", "B"], ["2400000", "3900000", "A"]], "a deletion over a compartment's edge joins the two sides there: " + str(comp)


def test_the_rules_run_with_triosformer_s_motif_cutoff(shared: SimpleNamespace, starts_and_inp_and_cfg: SimpleNamespace) -> None:
    ct, region, d = shared.ct, shared.region, shared.d
    dele = starts_and_inp_and_cfg.dele
    # The cutoff of 0 of the rules made 101 loops of 54 on this haplotype. The cutoff of 10
    # leaves 54 and one new anchor.
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
    assert seen == [motifs.THRESHOLD], "the rules run with triosformer's motif cutoff: " + str(seen)


def test_the_loops_the_job_and_the_preview_of_variants_rewire(tmp_path: Path) -> None:
    ct = Catalog.load(make_data(tmp_path / "data")).cell_types["GMX"]
    region = Region("chr1", 0, 3_000_000)
    anchors, length, network = loop_network(ct, region, "RNAPII", [], None)
    assert sorted(set(anchors[1])) == [1_300_000, 1_500_000] and set(anchors[3]) == {"N"}, "RNAPII alone keeps the anchors of its loop ends, without orientation: " + str(anchors.values.tolist())
    assert network == [("RNAPII", [["chr1", 1_300_000, 1_301_000, "chr1", 1_500_000, 1_501_000, "N", "N", 4]])], "and its loops have no orientation: " + str(network)
    assert length == 1_501_000 + 10_000_000, "the chromosome ends 10 Mb after the last anchor"
    anchors, _, network = loop_network(ct, region, "CTCF+RNAPII", [], None)
    assert len(anchors) == 16 and [(f, len(c)) for f, c in network] == [("CTCF", 14), ("RNAPII", 1)], "two factors keep every anchor and give their loops, CTCF first"
    mine = tmp_path / "mine.bedpe"
    mine.write_text("chr1\t900200\t900400\tchr1\t1300200\t1300400\t5\nchr1\t2000000\t2000400\tchr1\t2400000\t2400400\t3\n")
    own_anchors = tmp_path / "anchors.bed"
    own_anchors.write_text("chr1\t900000\t901000\tR\nchr1\t1300000\t1301000\tL\n")
    anchors, _, network = loop_network(ct, region, "CTCF", [("H3K27ac", mine)], own_anchors)
    assert network == [("H3K27ac", [["chr1", 900_000, 901_000, "chr1", 1_300_000, 1_301_000, "R", "L", 5], ["chr1", 2_000_000, 2_000_400, "chr1", 2_400_000, 2_400_400, "N", "N", 3]])], (
        "an end of an own loop is moved to the anchor that holds it and has its orientation, an end on no anchor stays: " + str(network)
    )
    anchors, _, _ = loop_network(ct, region, "CTCF", [("H3K27ac", mine)], None)
    assert anchors.values.tolist() == [["chr1", 900_200, 900_400, "N"], ["chr1", 1_300_200, 1_300_400, "N"], ["chr1", 2_000_000, 2_000_400, "N"], ["chr1", 2_400_000, 2_400_400, "N"]], "without own anchors the loop ends are the anchors: " + str(anchors.values.tolist())
