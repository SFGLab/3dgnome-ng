"""Turn a stage trace into the model creation video.

    python playground/figures/render_video.py TRACE.npz OUT_DIR [--fps 25] [--title "GM12878 chr1:1-8 Mb"]

Reads the npz `trace_stages.py` writes and renders one frame per moment of the reconstruction:
the block layout, the Hilbert curve the anchors start on, the anchors the loops pull into
place, the chain densified on straight lines, the coil start, the smooth stage's annealing
milestone by milestone, the boundary stitch and the cross block relaxation. Stage changes are
shown as short morphs so the eye can follow what moved. The chain is coloured by genomic
position and the camera orbits slowly. Frames go to OUT_DIR/frames and ffmpeg writes
OUT_DIR/model_creation.mp4 and a 720 pixel GIF beside it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.figures.tube import draw, project, tube_pieces  # noqa: E402
from playground.validation_battery import _flag, _str_flag  # noqa: E402

HOLD = 22
MORPH = 28


def lerp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    t = (1 - np.cos(np.linspace(0, np.pi, n))) / 2  # ease in and out
    return [a + (b - a) * ti for ti in t]


class Scene:
    def __init__(self, npz: dict[str, np.ndarray]) -> None:
        self.n = int(npz["n_blocks"])
        self.blocks = list(range(self.n))
        self.start = [npz[f"b{k}_start"] for k in self.blocks]
        self.fixed = [npz[f"b{k}_fixed"] for k in self.blocks]
        self.amid = [npz[f"b{k}_anchor_mid"] for k in self.blocks]
        self.d = npz
        lo = min(int(s.min()) for s in self.start)
        hi = max(int(s.max()) for s in self.start)
        self.lo, self.hi = lo, hi
        # one colour per bead by genomic position, and per anchor for the anchor stages
        self.bead_col = [plt.get_cmap("turbo")((s - lo) / (hi - lo)) for s in self.start]
        self.anchor_col = [plt.get_cmap("turbo")((m - lo) / (hi - lo)) for m in self.amid]

    def anchors(self, key: str) -> list[np.ndarray]:
        return [self.d[f"b{k}_{key}"] for k in self.blocks]

    def chains(self, key: str) -> list[np.ndarray]:
        return [self.d[f"b{k}_{key}"] for k in self.blocks]

    def frames(self) -> list[tuple[str, str, list[np.ndarray], bool]]:
        """(caption, sub caption, per block positions, is_chain) per frame."""
        out: list[tuple[str, str, list[np.ndarray], bool]] = []

        def hold(cap: str, sub: str, pos: list[np.ndarray], chain: bool, n: int = HOLD) -> None:
            out.extend([(cap, sub, pos, chain)] * n)

        def morph(cap: str, sub: str, a: list[np.ndarray], b: list[np.ndarray], chain: bool) -> None:
            for i in range(MORPH):
                out.append((cap, sub, [lerp(x, y, MORPH)[i] for x, y in zip(a, b, strict=True)], chain))

        layout, hilbert, solved = self.anchors("layout"), self.anchors("hilbert"), self.anchors("anchors")
        hold("Block layout", "each interaction block placed by its chain of blocks", layout, False)
        morph("Hilbert start", "every anchor of the chromosome on one space filling curve", layout, hilbert, False)
        hold("Hilbert start", "every anchor of the chromosome on one space filling curve", hilbert, False)
        morph("Loops placed", "the arcs solve pulls loop anchors together", hilbert, solved, False)
        hold("Loops placed", "the arcs solve pulls loop anchors together", solved, False)
        line, coil = self.chains("line"), self.chains("coil")
        hold("Chain densified", "one bead per kilobase between the anchors", line, True)
        morph("Coil start", "each gap's beads on a compact bridge, clear of each other", line, coil, True)
        hold("Coil start", "each gap's beads on a compact bridge, clear of each other", coil, True)
        traj = [self.d[f"b{k}_frames"] for k in self.blocks]
        n_ms = max(t.shape[0] for t in traj)
        for i in range(n_ms):
            pos = [t[min(i, t.shape[0] - 1)] for t in traj]
            sub = f"Monte Carlo on the chain, milestone {i + 1} of {n_ms}, 50,000 moves each, hard wall on"
            out.append(("Smoothing", sub, pos, True))
            out.append(("Smoothing", sub, pos, True))
        final = self.chains("final")
        hold("Smoothing", "the smooth stage's own end point", final, True)
        if self.n > 1:
            stitched, relaxed = self.chains("stitched"), self.chains("relaxed")
            morph("Boundary stitch", "blocks moved rigidly so boundary pairs sit at the structure's own distance", final, stitched, True)
            hold("Boundary stitch", "blocks moved rigidly so boundary pairs sit at the structure's own distance", stitched, True)
            morph("Cross block relaxation", "coils of different blocks re route around each other", stitched, relaxed, True)
            hold("Cross block relaxation", "coils of different blocks re route around each other", relaxed, True, n=2 * HOLD)
        else:
            hold("Done", "one block, so nothing to stitch", final, True, n=2 * HOLD)
        return out


def main() -> None:
    fps = int(_flag("--fps", 25))
    title = _str_flag("--title")
    npz_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.png"):
        f.unlink()
    scene = Scene(dict(np.load(npz_path)))
    frames = scene.frames()
    # The camera's base frame is the finished structure's principal axes, longest across the
    # screen, and the film orbits slowly about the screen's vertical. One box for the whole
    # film, from the projected extent of every keyframe under its own orbit angle.
    last = np.concatenate(frames[-1][2])
    c = last.mean(0)
    _, _, vt = np.linalg.svd(last - c, full_matrices=False)
    base = vt if np.linalg.det(vt) > 0 else vt * np.array([[1.0], [1.0], [-1.0]])

    def rotation(i: int) -> np.ndarray:
        t = np.radians(0.18 * i)
        orbit = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
        return orbit @ base

    ext_x = ext_y = 0.0
    for i in range(0, len(frames), max(1, len(frames) // 120)):
        pts = np.concatenate(frames[i][2])
        xy, _ = project(pts, rotation(i), c)
        ext_x = max(ext_x, float(np.abs(xy[:, 0]).max()))
        ext_y = max(ext_y, float(np.abs(xy[:, 1]).max()))
    half_w = max(ext_x, ext_y * 16 / 9) * 1.04
    print(f"{len(frames)} frames, {len(frames) / fps:.1f} s at {fps} fps", flush=True)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    cap_text = fig.text(0.03, 0.93, "", fontsize=22, weight="bold", color="#202020")
    sub_text = fig.text(0.03, 0.885, "", fontsize=12.5, color="#404040")
    if title:
        fig.text(0.97, 0.04, title, fontsize=11, color="#606060", ha="right")
    for i, (cap, sub, pos, chain) in enumerate(frames):
        ax.cla()
        ax.set_axis_off()
        rot = rotation(i)
        if chain:
            pieces = [tube_pieces(p, scene.bead_col[k], rot, c, width=1.6) for k, p in enumerate(pos)]
            draw(ax, pieces, background="white", halo=1.4, shade=0.5, thin=0.4)
            for k, p in enumerate(pos):
                a = scene.fixed[k]
                xy, _ = project(p[a], rot, c)
                ax.scatter(xy[:, 0], xy[:, 1], s=5, c="black", zorder=5, linewidths=0)
        else:
            for k, p in enumerate(pos):
                xy, z = project(p, rot, c)
                ax.plot(xy[:, 0], xy[:, 1], color="#b8b8b8", lw=0.6, alpha=0.8, zorder=1)
                order = np.argsort(z)
                ax.scatter(xy[order, 0], xy[order, 1], s=12, c=scene.anchor_col[k][order], zorder=3, linewidths=0)
        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(-half_w * 9 / 16, half_w * 9 / 16)
        ax.set_aspect("equal")
        cap_text.set_text(cap)
        sub_text.set_text(sub)
        fig.savefig(frames_dir / f"frame_{i:05d}.png", dpi=100, facecolor="white")
        if i % 100 == 0:
            print(f"  frame {i}", flush=True)
    plt.close(fig)
    mp4 = out_dir / "model_creation.mp4"
    gif = out_dir / "model_creation.gif"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vf", "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer",
         str(gif)],
        check=True,
    )  # fmt: skip
    print(f"wrote {mp4} and {gif}")


if __name__ == "__main__":
    main()
