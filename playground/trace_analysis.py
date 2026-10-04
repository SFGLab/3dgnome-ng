"""The energy trajectory of a traced arcs solve, for choosing [simulation_arcs] solver_tol.
    python trace_analysis.py TRACE OUT_PNG"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
lines = open(sys.argv[1]).read().splitlines()
blocks, cur, hdr = [], [], None
for ln in lines:
    if ln.startswith("#"):
        if cur: blocks.append((hdr, np.array(cur))); cur = []
        hdr = ln
    else: cur.append(float(ln))
if cur: blocks.append((hdr, np.array(cur)))
for hdr, e in blocks:
    print(hdr)
    best = np.minimum.accumulate(e); final = best[-1]
    n = len(best)
    for k in (100, 200, 400, 800, 1500, 3000, 5000, n - 1):
        if k < n: print(f"  evaluation {k:5d}: energy {best[k]:,.1f}, {100 * (best[k] - final) / final:.3f} percent above the last")
    rel = (best[:-1] - best[1:]) / np.maximum(np.abs(best[1:]), 1.0)
    for tol in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        w = np.where(rel < tol)[0]
        first = int(w[0]) if len(w) else None
        # first evaluation after which the relative improvement stays under tol for 20 in a row
        run = np.convolve((rel < tol).astype(int), np.ones(20, int), "valid"); stable = np.where(run == 20)[0]
        st = int(stable[0]) if len(stable) else None
        print(f"  tol {tol:.0e}: first hit at evaluation {first}, stays under for 20 in a row from {st}, energy there {best[st] if st is not None else float('nan'):,.1f} ({100 * (best[st] - final) / final if st is not None else float('nan'):.3f} percent above the last)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(best); ax[0].set_xlabel("evaluation"); ax[0].set_ylabel("best energy"); ax[0].set_yscale("log"); ax[0].set_title(hdr[2:60], fontsize=9)
    ax[1].plot(np.maximum(rel, 1e-12)); ax[1].set_yscale("log"); ax[1].set_xlabel("evaluation"); ax[1].set_ylabel("relative improvement per evaluation")
    for tol in (1e-4, 1e-5, 1e-6): ax[1].axhline(tol, color="grey", lw=0.5)
    fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110); print(f"wrote {sys.argv[2]}")
