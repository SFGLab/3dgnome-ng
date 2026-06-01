"""Generic JAX driver (smooth recipe) == the hand-written smooth kernel, byte-exact.

Builds a REAL smooth `Problem` (run Seeded->arcs->densify->heat, then
SmoothStage.to_problem — so it carries genuine orientation + subanchor-heat data),
preps it exactly like `_mc_smooth_jax_batch_chunk`, then runs the OLD
`_build_smooth_kernel().kernel_full_mp` and the NEW
`build_mc_kernel([CHAIN, EXCLUDED_VOLUME, SUBANCHOR_HEAT, ORIENTATION, CONFINEMENT])`
with the SAME init scores + base key, asserting identical final pos + score.

Smooth uses strict acceptance, no freeze, and no ratio guard — the generic driver
matches by `strict_accept=True, freeze_converged=False, stop_ratio=inf` (the
ratio term then never fires, a no-op).  The recipe order reproduces the old
ss+se+sh+so+sc sum exactly.

    JAX_PLATFORMS=cpu .venv/bin/python -u playground/refactor/validate_driver_smooth.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "playground/refactor")

from _common import load_region  # noqa: E402

from gnome3d import skeleton  # noqa: E402
from gnome3d.mc import jax as mc_jax  # noqa: E402
from gnome3d.mc.jax_driver import build_mc_kernel  # noqa: E402
from gnome3d.mc.terms import (  # noqa: E402
    CHAIN,
    CONFINEMENT,
    EXCLUDED_VOLUME,
    ORIENTATION,
    SUBANCHOR_HEAT,
)
from gnome3d.mc.terms.chain import ChainP  # noqa: E402
from gnome3d.mc.terms.confinement import ConfP  # noqa: E402
from gnome3d.mc.terms.excluded_volume import ExclP  # noqa: E402
from gnome3d.mc.terms.orientation import OrnP  # noqa: E402
from gnome3d.mc.terms.subanchor_heat import HeatP  # noqa: E402
from gnome3d.pipeline import coarse as cb  # noqa: E402
from gnome3d.pipeline.coarse.stages import build_coarse_dag  # noqa: E402
from gnome3d.pipeline.executor import SerialExecutor  # noqa: E402
from gnome3d.pipeline.ib import arcs as ib_arcs  # noqa: E402
from gnome3d.pipeline.ib import densify as ib_densify  # noqa: E402
from gnome3d.pipeline.ib import heat as ib_heat  # noqa: E402
from gnome3d.pipeline.ib.arcs import ArcsStage  # noqa: E402
from gnome3d.pipeline.ib.densify import DensifyStage  # noqa: E402
from gnome3d.pipeline.ib.heat import HeatDistStage  # noqa: E402
from gnome3d.pipeline.ib.smooth import SmoothStage  # noqa: E402


def _smooth_problem(ibseed):
    """Run the IB chain to the smooth stage's input and return its Problem."""
    seeded = ibseed.seed
    arced = ArcsStage().apply((seeded,), ib_arcs._run(ArcsStage().to_problem((seeded,))))
    densified = DensifyStage().apply((arced,), ib_densify._run(DensifyStage().to_problem((arced,))))
    st = densified
    if ibseed.wants_heat:
        st = HeatDistStage().apply((densified,), ib_heat._run(HeatDistStage().to_problem((densified,))))
    return SmoothStage().to_problem((st,))


def main() -> int:
    s, bed, data, _ = load_region()
    s.mc_backend = "jax"
    s.jax_bucket_shapes = True
    s.mc_stop_steps_smooth = 500  # small batch -> quick convergence for the test

    state = cb.build_state(s, data, [bed.chr], bed)
    spine, _ = build_coarse_dag(state, 0, fan_out=False)
    SerialExecutor().run(spine)
    ibseeds = skeleton.gather_all_ib_seeds(state, 0)
    problems = [_smooth_problem(ib) for ib in ibseeds]
    print(f"smooth IBs: {len(problems)} (beads {[len(p['pos']) for p in problems]})")

    assert mc_jax._ensure_jax()
    jax, jnp = mc_jax._jax, mc_jax._jnp
    f32 = jnp.float32

    p0 = problems[0]
    use_orn = (
        p0.get("char_orientations") is not None
        and p0.get("anchor_neighbors") is not None
        and p0.get("anchor_neighbor_weights") is not None
    )
    use_heat = p0.get("heat_dist") is not None
    use_excl = bool(s.use_excluded_volume) and bool(s.exclusion_apply_to_smooth)
    use_conf = bool(s.use_confinement) and bool(s.confinement_apply_to_smooth)
    print(f"  use_heat={use_heat} use_orn={use_orn} use_excl={use_excl} use_conf={use_conf}")

    # common (B, A, M) buckets — mirror _mc_smooth_jax_batch_chunk
    Bs, As, Ms = [], [], []
    for p in problems:
        n_i = int(p["pos"].shape[0])
        Bs.append(mc_jax._bucket_for(n_i))
        if use_orn:
            anchors_i = int(np.count_nonzero(p["fixed"]))
            nbrs_i = max((len(p["anchor_neighbors"].get(k, [])) for k in range(anchors_i)), default=1)
            As.append(mc_jax._bucket_for(max(anchors_i, 1), mc_jax._ANCHOR_BUCKETS))
            Ms.append(mc_jax._bucket_for(max(nbrs_i, 1), mc_jax._NBR_BUCKETS))
        else:
            As.append(1)
            Ms.append(1)
    B, A, M = max(Bs), max(As), max(Ms)
    K = len(problems)

    preps = [
        mc_jax._prep_smooth_problem_np(
            p["pos"], p["dtn"], p["fixed"], s, p.get("char_orientations"),
            p.get("anchor_neighbors"), p.get("anchor_neighbor_weights"), p.get("heat_dist"), B, A, M,
        )
        for p in problems
    ]

    def st_(key):
        return jnp.asarray(np.stack([pr[key] for pr in preps], axis=0))

    def arr_(key, dt):
        return jnp.asarray(np.array([pr[key] for pr in preps], dtype=dt))

    pos_k, dtn_k, movable_k, heat_k = st_("pos"), st_("dtn"), st_("movable"), st_("heat")
    anchor_ar_k, b2a_k = st_("anchor_ar"), st_("bead_to_anchor_k")
    nbr_idx_k, nbr_w_k, nbr_valid_k, is_L_k = st_("nbr_idx"), st_("nbr_w"), st_("nbr_valid"), st_("is_L")
    n_active_k, n_movable_k = arr_("n_active", np.int32), arr_("n_movable", np.int32)
    excl_r0_k = arr_("excl_r0", np.float32)
    cx, cy, cz, cR, cw = (arr_("conf_cx", np.float32), arr_("conf_cy", np.float32),
                          arr_("conf_cz", np.float32), arr_("conf_R", np.float32), arr_("conf_w", np.float32))
    step_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    stretch_k = jnp.full((K,), f32(s.spring_stretch))
    squeeze_k = jnp.full((K,), f32(s.spring_squeeze))
    ang_k = jnp.full((K,), f32(s.spring_angular))
    excl_skip = int(s.exclusion_skip_neighbors)
    n_steps = int(s.mc_stop_steps_smooth)
    heat_w = float(s.subanchor_heatmap_dist_weight) if use_heat else 0.0
    motif_w = float(s.motif_weight) if use_orn else 0.0
    excl_w = float(s.exclusion_weight) if use_excl else 0.0
    dist_w, ang_w = f32(s.smooth_dist_weight), f32(s.smooth_angle_weight)
    symmetric = jnp.bool_(bool(getattr(s, "motifs_symmetric", True)))
    base_key = jax.random.PRNGKey(0)

    # --- init scores via the OLD bundle helpers (fed to BOTH kernels) ---
    bundle = mc_jax._build_smooth_kernel(n_steps, excl_skip, use_heat, use_orn, M)
    _kb, _kf, init_smooth, init_excl, init_heat, init_confine, init_anchor_orn, init_orn_score, old_kfmp = bundle

    def init_one(i):
        p1 = pos_k[i : i + 1]
        na = jnp.int32(int(np.asarray(n_active_k[i])))
        ss = init_smooth(p1, dtn_k[i], stretch_k[i], squeeze_k[i], ang_k[i], dist_w, ang_w, na)
        se = init_excl(p1, excl_r0_k[i], f32(excl_w), na) if use_excl else jnp.zeros((1,), f32)
        sh = init_heat(p1, heat_k[i], f32(heat_w)) if use_heat else jnp.zeros((1,), f32)
        sc = init_confine(p1, cx[i], cy[i], cz[i], cR[i], cw[i], na) if use_conf else jnp.zeros((1,), f32)
        if use_orn:
            ao = init_anchor_orn(p1, anchor_ar_k[i], is_L_k[i])
            so = init_orn_score(ao, nbr_idx_k[i], nbr_w_k[i], nbr_valid_k[i], f32(motif_w), symmetric)
        else:
            ao, so = jnp.zeros((1, A, 3), f32), jnp.zeros((1,), f32)
        return ss, se, sh, so, sc, ao

    inits = [init_one(i) for i in range(K)]
    ss_k, se_k, sh_k, so_k, sc_k = (jnp.concatenate([x[j] for x in inits]) for j in range(5))
    anchor_orn_k = jnp.concatenate([x[5] for x in inits])

    # --- OLD smooth kernel ---
    out_old = old_kfmp(
        pos_k, ss_k, se_k, sh_k, so_k, sc_k, anchor_orn_k, f32(s.max_temp_smooth),
        dtn_k, movable_k, heat_k, anchor_ar_k, b2a_k, nbr_idx_k, nbr_w_k, nbr_valid_k, is_L_k, step_k,
        f32(s.dt_temp_smooth), f32(s.jump_scale_smooth), f32(s.jump_coef_smooth),
        stretch_k, squeeze_k, ang_k, dist_w, ang_w, excl_r0_k, f32(excl_w), f32(heat_w), f32(motif_w), symmetric,
        cx, cy, cz, cR, cw, base_key,
        f32(s.mc_stop_improvement_smooth), jnp.int32(s.mc_stop_successes_smooth), f32(1e-6), n_active_k, n_movable_k,
    )
    pos_old = np.asarray(out_old[0])
    score_old = np.asarray(out_old[1] + out_old[2] + out_old[3] + out_old[4] + out_old[5])

    # --- NEW generic driver ---
    onesK = jnp.ones((K,), f32)
    term_params = (
        ChainP(dtn_k, stretch_k, squeeze_k, ang_k, onesK * dist_w, onesK * ang_w),
        ExclP(excl_r0_k, onesK * excl_w, (onesK * excl_skip).astype(jnp.int32)),
        HeatP(heat_k, onesK * heat_w),
        OrnP(b2a_k, anchor_ar_k, is_L_k, nbr_idx_k, nbr_w_k, nbr_valid_k, onesK * motif_w,
             jnp.full((K,), symmetric)),
        ConfP(cx, cy, cz, cR, cw),
    )
    kfmp = build_mc_kernel(
        [CHAIN, EXCLUDED_VOLUME, SUBANCHOR_HEAT, ORIENTATION, CONFINEMENT],
        n_steps, strict_accept=True, freeze_converged=False,
    )
    out_new = kfmp(
        pos_k, (ss_k, se_k, sh_k, so_k, sc_k), anchor_orn_k, f32(s.max_temp_smooth), term_params, movable_k, step_k,
        f32(s.dt_temp_smooth), f32(s.jump_scale_smooth), f32(s.jump_coef_smooth), base_key,
        f32(s.mc_stop_improvement_smooth), jnp.int32(s.mc_stop_successes_smooth), f32(1e-6), f32(np.inf),
        n_active_k, n_movable_k,
    )
    pos_new = np.asarray(out_new[0])
    sc_new = out_new[1]
    score_new = np.asarray(sc_new[0] + sc_new[1] + sc_new[2] + sc_new[3] + sc_new[4])

    pos_diff = float(np.max(np.abs(pos_old - pos_new)))
    score_diff = float(np.max(np.abs(score_old - score_new)))
    print(f"  K={K} B={B} A={A} M={M}  iters old={int(np.asarray(out_old[8]))} new={int(np.asarray(out_new[3]))}")
    print(f"  max |Δpos| = {pos_diff:.3e}   max |Δscore| = {score_diff:.3e}")
    exact = pos_diff == 0.0 and score_diff == 0.0
    print("PASS (byte-exact: generic driver == hand-written smooth kernel)" if exact else "FAILED")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
