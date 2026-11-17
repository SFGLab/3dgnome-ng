"""Render the Method page equations of the design mock as images with matplotlib mathtext.

    python web/design/render_equations.py web/design/images/eq

Writes one transparent PNG per equation at twice its display size and `sizes.json` with the
display width and height of each in pixels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["mathtext.fontset"] = "cm"

INK = "#151514"
DISPLAY = 15
CELL = 12

EQUATIONS: dict[str, tuple[str, int]] = {
    "background": (r"$b(s) = \max\left(1,\ \left(\frac{s}{s_0}\right)^{\nu}\right),\quad s_0 = 1\ \mathrm{kb}$", DISPLAY),
    "loop_target": (r"$d(s, q) = 1 + \frac{b(s) - 1}{1 + q},\quad q = \frac{n}{\bar{n}_f(s)}$", DISPLAY),
    "contact_prob": (r"$P(s) \propto s^{-3\nu}$", DISPLAY),
    "anchor_energy": (
        r"$E_{\mathrm{anchor}} = \sum_{(i,j) \in L} k \left(\frac{d_{ij} - t_{ij}}{t_{ij}}\right)^2"
        r" + w_b \sum_{(i,j) \in B} \left(\frac{d_{ij} - b_{ij}}{\min(d_{ij}, b_{ij})}\right)^2"
        r"$" "\n" r"$\qquad + w_g \sum_{w \in W} n_w \left(\frac{R_g(w)}{R_w} - 1\right)^2"
        r" + w_{ev} \sum_{|i - j| > 1} \left(\frac{r_0 - d_{ij}}{r_0}\right)_{+}^{2}$",
        DISPLAY,
    ),
    "chain_energy": (
        r"$E_{\mathrm{chain}} = \sum_i w_d k \left(\frac{d_{i,i+1} - t_i}{t_i}\right)^2"
        r" + \sum_i w_a \left(\frac{1 - \cos\theta_i}{2}\right)^3"
        r"$" "\n" r"$\qquad + \sum_{(i,j) \in C} w_m \left(\frac{1 + \mathbf{o}_i \cdot \mathbf{o}_j}{2}\right)^2"
        r" + w_{ev} \sum_{|i - j| > 1} \left(\frac{r_0 - d_{ij}}{r_0}\right)_{+}^{2}"
        r" + w_c \sum_i \left(\frac{r_i - R}{R}\right)_{+}^{2}$",
        DISPLAY,
    ),
    "acceptance_plain": (r"$p_{\mathrm{accept}} = \min\left(1,\ \exp\left(-\frac{c\, E'}{E\, T}\right)\right)\quad \mathrm{for}\ \Delta E \geq 0$", DISPLAY),
    "t_loop_simple": (r"$k \left(\frac{d_{ij} - t_{ij}}{t_{ij}}\right)^2$", CELL),
    "t_background": (r"$w_b \left(\frac{d_{ij} - b_{ij}}{\min(d_{ij}, b_{ij})}\right)^2$", CELL),
    "t_repulsion": (r"$\max\left(0,\ \frac{1}{d_{ij}} - \frac{1}{r_c}\right)$", CELL),
    "t_gyration": (r"$w_g\, n_w \left(\frac{R_g(w)}{R_w} - 1\right)^2$", CELL),
    "t_ev": (r"$w_{ev} \left(\frac{r_0 - d_{ij}}{r_0}\right)^2,\ d_{ij} < r_0$", CELL),
    "t_bond": (r"$w_d k \left(\frac{d_{i,i+1} - t_i}{t_i}\right)^2$", CELL),
    "t_bending": (r"$w_a \left(\frac{1 - \cos\theta_i}{2}\right)^3$", CELL),
    "t_orientation": (r"$w_m \left(\frac{1 + \mathbf{o}_i \cdot \mathbf{o}_j}{2}\right)^2$", CELL),
    "t_confinement": (r"$w_c \left(\frac{r_i - R}{R}\right)^2,\ r_i > R$", CELL),
}


def render(tex: str, size: int, path: Path) -> tuple[int, int]:
    """Draws one equation and returns its display width and height in pixels."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color=INK)
    fig.savefig(path, dpi=192, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    w, h = plt.imread(path).shape[1::-1]
    return w // 2, h // 2


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, (tex, size) in EQUATIONS.items():
        try:
            sizes[name] = render(tex, size, out / f"{name}.png")
        except ValueError as err:
            print(f"{name}: {err}")
    (out / "sizes.json").write_text(json.dumps(sizes, indent=1))
    print(json.dumps(sizes))


if __name__ == "__main__":
    main()
