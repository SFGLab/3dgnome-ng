"""Console report formatting shared by the validation studies."""

from __future__ import annotations

import numpy as np

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
PASS, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"


def verdict(ok: bool, warn: bool = False) -> str:
    return WARN if warn else (PASS if ok else FAIL)


def fmt(label: str, value: float, unit: str = "") -> str:
    return f"    {label:<26}{value:10.4f} {unit}"


def print_single(name: str, m: dict[str, float]) -> None:
    print(f"\n  [{name}]  {int(m['n_beads'])} beads/structure")
    print(fmt("Rg", m["rg"]))
    print(fmt("median bond length", m["bond"]))
    print(fmt("max extent (centroid)", m["max_extent"]))
    print(fmt("overlap fraction", m["overlap_frac"], "(non-bonded pairs < radius)"))
    print(fmt("self-consistency rho", m["selfconsistency_rho"], "(want < 0)"))
    print(fmt("distance scaling exp", m["dist_scaling_exp"], "(want > 0)"))
    print(fmt("contact-decay exp", m["contact_prob_exp"], "(want < 0)"))
    print(fmt("ensemble diversity", m["diversity_dab"], "(median d_AB)"))


def print_comparison(target: str, base: dict[str, float], treat: dict[str, float]) -> bool:
    print(f"\n{'=' * 70}\n  prove '{target}', flags off baseline against on treatment\n{'=' * 70}")
    print_single("baseline (off)", base)
    print_single("treatment (on)", treat)

    results: list[bool] = []
    print("\n  [verdict]")

    # Each divergence has its own success axis.
    #   ev / all      excluded volume should reduce overlaps, which is its purpose.
    #   confinement   a containment envelope, so it should reduce spatial extent, Rg or max extent.
    #                 It is not an anti-overlap term. Compaction may even raise overlaps slightly,
    #                 so overlaps are reported as info only. Pair it with EV.
    #   dynamic       changes bead spacing. Overlaps should at least not inflate.
    ov_b, ov_t = base["overlap_frac"], treat["overlap_frac"]
    ext_b, ext_t = base["max_extent"], treat["max_extent"]
    if target == "confinement":
        ok = ext_t <= ext_b + 1e-9
        print(
            f"  {verdict(ok)}  max extent {ext_b:.2f} -> {ext_t:.2f}"
            f"  {'reduced' if ext_t < ext_b else 'not reduced'}. confinement should compact"
        )
        results.append(ok)
        print(
            f"  {WARN}  overlaps {ov_b:.4f} -> {ov_t:.4f}. confinement alone need not cut these"
        )
    elif target in ("ev", "all"):
        ok = ov_t <= ov_b + 1e-9
        print(
            f"  {verdict(ok)}  overlaps {ov_b:.4f} -> {ov_t:.4f}"
            f"  {'reduced' if ov_t < ov_b else 'not reduced'}. excluded volume should cut overlaps"
        )
        results.append(ok)
    else:  # dynamic
        ok = ov_t <= ov_b * 1.5 + 1e-9
        print(f"  {verdict(ok)}  overlaps {ov_b:.4f} -> {ov_t:.4f}  (should not inflate)")
        results.append(ok)

    # Self-consistency must not degrade, so rho stays comparably negative.
    rb, rt = base["selfconsistency_rho"], treat["selfconsistency_rho"]
    if np.isfinite(rb) and np.isfinite(rt):
        ok = rt <= rb + 0.10  # treatment no more than 0.10 worse, meaning less negative
        print(f"  {verdict(ok)}  self-consistency {rb:+.3f} -> {rt:+.3f}  (must not degrade)")
        results.append(ok)
    else:
        print(f"  {WARN}  self-consistency unavailable, too few in-region contacts")

    # Scaling laws stay in sane polymer bands in both runs.
    for key, lo, hi, want in (
        ("dist_scaling_exp", 0.05, 1.0, "distance grows with separation"),
        ("contact_prob_exp", -2.5, -0.2, "contacts decay with separation"),
    ):
        vb, vt = base[key], treat[key]
        ok = bool(np.isfinite(vt) and lo <= vt <= hi)
        print(
            f"  {verdict(ok, warn=not np.isfinite(vt))}  scaling law {key} {vb:+.3f} -> {vt:+.3f}"
            f"  (sane: [{lo}, {hi}]; {want})"
        )
        if np.isfinite(vt):
            results.append(ok)

    # Diversity should not collapse to ~0 or explode.
    db, dt = base["diversity_dab"], treat["diversity_dab"]
    if np.isfinite(dt):
        ok = dt > 1e-6
        print(f"  {verdict(ok)}  ensemble diversity {db:.4f} -> {dt:.4f}  (must not collapse)")
        results.append(ok)

    all_ok = all(results)
    print(f"\n  {'=' * 66}\n  {target}: {PASS if all_ok else FAIL}")
    return all_ok
