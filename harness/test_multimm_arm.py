"""Unit checks for the MultiMM arm of the validation battery.

    python harness/test_multimm_arm.py

MultiMM writes beads with no genomic coordinates, so the arm has to give each bead a span before
the battery can bin it against Hi-C. It also needs a compartment bed in the labelled form MultiMM
reads, a config that pins the bead count to ours, and a collector that turns an ensemble's output
tree into one directory of cifs per arm. Each of those is checked here on hand built inputs, and
the relabelled cif is read back through the battery's own loader so the two cannot drift apart.
"""

from __future__ import annotations

import configparser
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.io import write_cif  # noqa: E402
from playground.multimm_arm import (  # noqa: E402
    collect_ensemble,
    compartment_bed,
    phase_by_loops,
    read_multimm_cif,
    uniform_beads,
    write_multimm_config,
)
from playground.validation_battery import cell_nu, load, read_singletons  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


MULTIMM_HEADER = """data_multimm
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
"""


def multimm_cif(path: Path, xyz: np.ndarray) -> None:
    """The layout `initial_structure_tools.write_mmcif` produces, thirteen fields, x y z last."""
    lines = [MULTIMM_HEADER]
    for i, (x, y, z) in enumerate(xyz, start=1):
        comp, atom = ("ALB", "CB") if i in (1, len(xyz)) else ("ALA", "CA")
        lines.append(f"ATOM {i} D {atom} . {comp} A 0 {i} ? {x:.3f} {y:.3f} {z:.3f}\n")
    path.write_text("".join(lines))


def test_reads_multimm_cif() -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 2.0, 0.0], [1.5, 2.0, -3.25]])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "MultiMM_minimized.cif"
        multimm_cif(p, xyz)
        got = read_multimm_cif(p)
    check(
        "MultiMM ATOM records read back as positions", got.shape == (4, 3) and np.allclose(got, xyz)
    )


def test_uniform_beads_tile_the_region() -> None:
    xyz = np.zeros((4, 3))
    beads = uniform_beads(xyz, 0, 100)
    starts = [b.start for b in beads]
    ends = [b.end for b in beads]
    check(
        "bead spans tile the region end to end",
        starts == [0, 25, 50, 75] and ends == [25, 50, 75, 100],
    )
    check("every bead is a subanchor", all(b.kind == "subanchor" for b in beads))
    beads = uniform_beads(np.zeros((3, 3)), 1, 100)
    check(
        "an uneven division still tiles without gaps",
        [b.start for b in beads] == [1, 34, 67] and [b.end for b in beads] == [34, 67, 100],
    )


def test_relabelled_cif_round_trips_through_the_battery() -> None:
    rng = np.random.default_rng(1)
    xyz = rng.normal(size=(50, 3))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s1.cif"
        write_cif(str(p), uniform_beads(xyz, 1_000_000, 2_000_000))
        pos, start, mid, anchor = load(p)
    check(
        "the battery loads a relabelled cif in file order",
        pos.shape == (50, 3) and np.allclose(pos, xyz, atol=1e-6) and start[0] == 1_000_000,
    )
    check("bead starts are unique so the order is total", len(set(start.tolist())) == 50)
    check("no bead is an anchor, so overlaps land in the smooth column", not anchor.any())


def test_compartment_bed_from_the_track() -> None:
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "c.bedGraph"
        t.write_text(
            "chr1\t0\t100000\t0.5\n"  # straddles the region start, clipped
            "chr1\t100000\t200000\t1.2\n"
            "chr1\t200000\t300000\t-0.7\n"
            "chr1\t300000\t400000\t0\n"  # unassigned, dropped
            "chr1\t400000\t500000\tnan\n"  # unassigned, dropped
            "chr1\t500000\t600000\t-0.1\n"  # straddles the region end, clipped
            "chr2\t100000\t200000\t2.0\n"  # other chromosome
        )
        rows = compartment_bed(t, "chr1", 50_000, 550_000)
    check(
        "signed values become A and B rows inside the region only",
        rows
        == [
            ("chr1", 50_001, 100_000, "A"),
            ("chr1", 100_000, 200_000, "A"),
            ("chr1", 200_000, 300_000, "B"),
            ("chr1", 500_000, 549_999, "B"),
        ],
        str(rows),
    )


def test_phasing_puts_the_loop_anchors_on_a() -> None:
    """The eigenvector sign is arbitrary. The pipeline phases by anchor counts and so does this."""
    rows = [("chr1", 0, 100, "A"), ("chr1", 100, 200, "B")]
    with tempfile.TemporaryDirectory() as d:
        loops = Path(d) / "l.bedpe"
        loops.write_text(
            "chr1\t110\t120\tchr1\t150\t160\t5\n"  # both anchors in the B row
            "chr1\t10\t20\tchr1\t130\t140\t5\n"  # one each
            "chr2\t10\t20\tchr2\t30\t40\t5\n"  # another chromosome, ignored
        )
        flipped = phase_by_loops(rows, loops, "chr1", 0, 200)
    check(
        "labels flip when the B side holds more loop anchors",
        [r[3] for r in flipped] == ["B", "A"],
        str(flipped),
    )
    rows = [("chr1", 0, 100, "A"), ("chr1", 100, 200, "B")]
    with tempfile.TemporaryDirectory() as d:
        loops = Path(d) / "l.bedpe"
        loops.write_text("chr1\t10\t20\tchr1\t50\t60\t5\n")
        same = phase_by_loops(rows, loops, "chr1", 0, 200)
    check("and stay when the A side already holds them", [r[3] for r in same] == ["A", "B"])


def test_config_pins_the_bead_count() -> None:
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "mm.ini"
        write_multimm_config(
            cfg,
            loops=Path("/x/loops.bedpe"),
            out_path=Path(d) / "run",
            chrom="chr1",
            lo=1,
            hi=60_000_000,
            n_beads=42_480,
            n_ensemble=5,
            platform="OpenCL",
            compartments=None,
        )
        c = configparser.ConfigParser()
        c.read(cfg)
        m = c["Main"]
        check(
            "the config carries our bead count and the region",
            m.getint("N_BEADS") == 42_480
            and m["CHROM"] == "chr1"
            and m.getint("LOC_START") == 1
            and m.getint("LOC_END") == 60_000_000,
        )
        check(
            "an ensemble of five with molecular dynamics, plots off",
            m.getboolean("GENERATE_ENSEMBLE")
            and m.getint("N_ENSEMBLE") == 5
            and m.getboolean("SIM_RUN_MD")
            and not m.getboolean("SAVE_PLOTS"),
        )
        check(
            "no modelling level preset, which would reset the bead count",
            "MODELLING_LEVEL" not in m,
        )
        check("compartments off without a bed", not m.getboolean("COB_USE_COMPARTMENT_BLOCKS"))
        write_multimm_config(
            cfg,
            loops=Path("/x/loops.bedpe"),
            out_path=Path(d) / "run",
            chrom="chr1",
            lo=1,
            hi=60_000_000,
            n_beads=100,
            n_ensemble=2,
            platform="CPU",
            compartments=Path("/x/comp.bed"),
        )
        c = configparser.ConfigParser()
        c.read(cfg)
        m = c["Main"]
        check(
            "compartments on with a bed",
            m.getboolean("COB_USE_COMPARTMENT_BLOCKS") and m["COMPARTMENT_PATH"] == "/x/comp.bed",
        )


def test_collects_an_ensemble_into_arm_dirs() -> None:
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "mm"
        for i in (1, 2, 3):
            m = root.parent / f"mm_{i}" / "model"
            m.mkdir(parents=True)
            multimm_cif(m / "MultiMM_minimized.cif", rng.normal(size=(20, 3)))
            multimm_cif(m / "MultiMM_afterMD.cif", rng.normal(size=(20, 3)))
        arm_min = Path(d) / "arm_min"
        arm_md = Path(d) / "arm_md"
        n = collect_ensemble(root, 3, 1, 1_000_000, arm_min, arm_md)
        mins = sorted(arm_min.glob("*.cif"))
        mds = sorted(arm_md.glob("*.cif"))
        check(
            "one minimised and one after MD cif per member",
            n == 3 and len(mins) == 3 and len(mds) == 3,
        )
        pos, start, _, _ = load(mds[0])
        check(
            "collected cifs carry spans over the region",
            pos.shape == (20, 3) and start[-1] < 1_000_000,
        )


def test_cell_nu_is_the_runs_own_fit() -> None:
    """The battery's yardstick is the fit a run makes on the same file, or the named fallback."""
    from gnome3d.polymer import FALLBACK_NU, fit_contact_exponent

    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.bedpe"
        rows = []
        grid = 25_000
        for k in range(1, 200):
            sep = k * grid
            count = max(1, int(round(4000.0 * (sep / grid) ** -0.9)))
            for _ in range(min(count, 60)):
                a = int(rng.integers(0, 4000)) * grid
                rows.append(
                    f"chr1\t{a}\t{a + grid}\tchr1\t{a + sep}\t{a + sep + grid}\t{max(1, count // 60)}\n"
                )
        p.write_text("".join(rows))
        got = cell_nu(p)
        want = fit_contact_exponent(read_singletons(p))
    check(
        "cell_nu returns the polymer fit on the file",
        got.ok == want.ok and got.nu == want.nu,
        f"nu {got.nu:.3f} ok {got.ok} {got.reason}",
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "empty.bedpe"
        p.write_text("")
        got = cell_nu(p)
    check("an unfittable file yields the named fallback", not got.ok and got.nu == FALLBACK_NU)


def main() -> int:
    print("MultiMM arm checks\n")
    test_reads_multimm_cif()
    test_uniform_beads_tile_the_region()
    test_relabelled_cif_round_trips_through_the_battery()
    test_compartment_bed_from_the_track()
    test_phasing_puts_the_loop_anchors_on_a()
    test_config_pins_the_bead_count()
    test_collects_an_ensemble_into_arm_dirs()
    test_cell_nu_is_the_runs_own_fit()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
