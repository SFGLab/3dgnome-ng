"""Contact loading shared by the validation studies."""

from __future__ import annotations

from gnome3d.io import load_singletons
from gnome3d.settings import Settings
from gnome3d.types import BedRegion


def load_contacts(
    s: Settings, chrs_list: list[str], region: BedRegion | None
) -> list[tuple[int, int, float]]:
    """Input singleton contacts, genomic pos and score, for the self-consistency check, via the public loader."""
    path = s.data_path(s.data_singletons)
    raw = load_singletons(path, set(chrs_list), region)
    return [(p1, p2, float(sc)) for _c1, p1, _c2, p2, sc in raw]


def load_chiapet_contacts(
    s: Settings, chrs_list: list[str], region: BedRegion | None
) -> list[tuple[int, int, float]]:
    """Full input ChIA-PET contacts as (pos_a, pos_b, score), combining singleton weak contacts and
    cluster loop-arcs. This is the model's input heat map, for the cross-data correlation check of
    ChIA-PET versus Hi-C. Arcs are the strong PET clusters, singletons the dense Hi-C-like background."""
    from gnome3d.io import load_arcs

    chr_set = set(chrs_list)
    out = load_contacts(s, chrs_list, region)  # singleton weak contacts
    arcs, _long = load_arcs(s.data_path(s.data_pet_clusters), chr_set, region)
    for chrom in chrs_list:
        out.extend((a.start, a.end, float(a.score)) for a in arcs.get(chrom, []))
    return out
