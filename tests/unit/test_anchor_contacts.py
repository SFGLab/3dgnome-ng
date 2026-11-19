"""Unit checks for the anchor contact table and the two sided contact background.

    python tests/unit/test_anchor_contacts.py

The table shares each pixel between the anchors it overlaps in proportion to the overlap, and a
run reads it by anchor interval. Under `contact_background_z` an arcless pair whose count lies
beyond z Poisson deviations of its expectation is held in when above and out when below, and
every other entry of the target matrix is left as it was.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d import anchor_contacts as ac  # noqa: E402
from gnome3d.pipeline.coarse.build import add_contact_background  # noqa: E402
from gnome3d.polymer import PolymerLaw  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def test_overlap() -> None:
    print("\n[overlap] a pixel is shared in proportion to the overlap")
    bins = np.arange(0, 10_000, 1000, dtype=np.int64)
    w = ac.overlap_weights(bins, 1000, np.array([500, 5000]), np.array([2500, 6000])).toarray()
    check(
        "an anchor over two and a half bins",
        np.allclose(w[:3, 0], [0.5, 1.0, 0.5]) and w[3:, 0].sum() == 0,
    )
    check("an anchor on one whole bin", w[5, 1] == 1.0 and w[:, 1].sum() == 1.0)
    st, en = np.array([500, 5000]), np.array([1500, 6000])
    w = ac.overlap_weights(bins, 1000, st, en)
    b1 = np.array([0, 1, 5, 0])
    b2 = np.array([5, 5, 5, 1])
    cnt = np.array([4.0, 2.0, 9.0, 7.0])
    c = ac.aggregate_pixels(w, b1, b2, cnt).toarray()
    check("pair count is the overlap weighted sum", c[0, 1] == 0.5 * 4 + 0.5 * 2, f"{c[0, 1]}")
    check("a pixel inside one anchor and a diagonal pixel count for no pair", c.sum() == c[0, 1])


def test_table() -> None:
    print("\n[table] a run reads the table by anchor interval")
    t = ac.AnchorContacts(
        starts=np.array([0, 100, 100, 900], dtype=np.int64),
        ends=np.array([50, 150, 180, 950], dtype=np.int64),
        i=np.array([0, 1, 0], dtype=np.int32),
        j=np.array([2, 3, 3], dtype=np.int32),
        count=np.array([5.0, 7.0, 2.0], dtype=np.float32),
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.npz"
        ac.save(p, {"chr1": t}, 1000, "test")
        back = ac.load(str(p), {"chr1", "chr2"})
    check("round trip", set(back) == {"chr1"} and np.array_equal(back["chr1"].count, t.count))
    st = np.array([0, 100, 500, 900], dtype=np.int64)
    en = np.array([50, 180, 600, 950], dtype=np.int64)
    idx = ac.match(back["chr1"], st, en)
    check(
        "anchors sharing a start are told apart by the end",
        list(idx) == [0, 2, -1, 3],
        f"{list(idx)}",
    )
    m = ac.dense_counts(back["chr1"], st, en)
    want = np.zeros((4, 4))
    want[0, 1] = want[1, 0] = 5.0
    want[0, 3] = want[3, 0] = 2.0
    check("dense counts, symmetric, nothing for an anchor the table lacks", np.array_equal(m, want))


def settings(z: float) -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    s.use_contact_background = True
    s.background_weight = 0.1
    s.contact_background_z = z
    return s


def test_two_sided() -> None:
    print("\n[two sided] enriched pairs held in, depleted held out, the rest untouched")
    n = 60
    mids = [int(k * 50_000) for k in range(n)]
    lens = np.full(n, 1000.0)
    counts = np.full((n, n), 100.0)
    np.fill_diagonal(counts, 0.0)
    counts[10, 14] = counts[14, 10] = 400.0
    counts[20, 24] = counts[24, 20] = 0.0
    counts[30, 34] = counts[34, 30] = 400.0
    heat = counts / (np.outer(lens, lens) / 1e6)
    mat = np.full((n, n), -0.5)
    np.fill_diagonal(mat, 0.0)
    mat[30, 34] = mat[34, 30] = 2.0
    s = settings(3.0)
    out = add_contact_background(mat, mids, heat, s, lens)
    bg = s.polymer_law().background(200_000)
    check(
        "an enriched arcless pair is held in",
        -bg < out[10, 14] < -1.0,
        f"{out[10, 14]:.3f} against {-bg:.3f}",
    )
    check(
        "a depleted arcless pair is held out",
        out[20, 24] < -bg,
        f"{out[20, 24]:.3f} against {-bg:.3f}",
    )
    check("symmetric", out[10, 14] == out[14, 10] and out[20, 24] == out[24, 20])
    check("an arc pair is never touched", out[30, 34] == 2.0)
    check("a pair at its expectation keeps the repulsion", out[5, 9] == -0.5)
    check("the diagonal stays zero", float(np.abs(np.diag(out)).max()) == 0.0)
    hi = s.heatmap_distance_stretching * bg
    check(
        "a pair with no contact sits at a finite distance",
        np.isfinite(out[20, 24]) and -out[20, 24] <= hi + 1e-9,
    )
    thin = add_contact_background(mat, mids, heat * 0.05, s, lens)
    check(
        "a pair whose expectation is under z squared is never tested, even when enriched",
        thin[10, 14] == -0.5 and thin[20, 24] == -0.5,
        f"expected 5 counts, {thin[10, 14]:.3f}",
    )
    old = add_contact_background(mat, mids, heat, settings(0.0), lens)
    check(
        "at z zero only enriched pairs are held, the old rule",
        old[20, 24] == -0.5 and old[10, 14] < -1.0,
    )
    try:
        add_contact_background(mat, mids, heat, s)
        check("refuses without the anchor lengths", False)
    except ValueError:
        check("refuses without the anchor lengths", True)


def test_settings() -> None:
    print("\n[settings] the two keys are read")
    s = Settings.from_dict(
        {"data": {"anchor_contacts": "x.npz"}, "springs": {"contact_background_z": "3"}}
    )
    check("read", s.data_anchor_contacts == "x.npz" and s.contact_background_z == 3.0)
    d = Settings()
    check("defaults off", d.data_anchor_contacts == "" and d.contact_background_z == 0.0)


def main() -> int:
    test_overlap()
    test_table()
    test_two_sided()
    test_settings()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
