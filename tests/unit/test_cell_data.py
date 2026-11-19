"""Unit checks for the data section of a cell line."""

from __future__ import annotations

from pathlib import Path

from gnome3d.production import cell_data_section


def test_without_a_hi_c_file_the_chia_pet_singletons_are_the_input(tmp_path: Path) -> None:
    tmp = str(tmp_path)
    root = Path(tmp)
    (root / "GM12878").mkdir()
    d = cell_data_section("GM12878", str(root))
    assert d["singletons"] == "GM12878_singletons_lessthan3.bedpe", (
        "without a Hi-C file the ChIA-PET singletons are the input: " + str(d["singletons"])
    )
    (root / "GM12878" / "GM12878_hic_25kb_singletons.bedpe").write_text("")
    d = cell_data_section("GM12878", str(root))
    assert d["singletons"] == "GM12878_hic_25kb_singletons.bedpe", (
        "with a Hi-C file it is the input: " + str(d["singletons"])
    )
