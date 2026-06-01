"""Generic JAX driver (arcs recipe) == the hand-written arcs kernel, byte-exact.

Runs the OLD `_build_arcs_kernel().kernel_full_mp` and the NEW
`build_mc_kernel([ARC_SPRINGS, EXCLUDED_VOLUME, CONFINEMENT], ...)` on the SAME
prepped real IBs with the SAME base key, and asserts identical final positions +
scores.  Because the generic per-step body reproduces the old RNG draw order and
per-term accumulation, and the term jax fns are verbatim copies, the match should
be exact (not just ballpark) — a tight gate on the driver swap.

    JAX_PLATFORMS=cpu .venv/bin/python -u playground/refactor/validate_driver_arcs.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "playground/refactor")

from _common import load_region  # noqa: E402

from gnome3d import skeleton  # noqa: E402
from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_region  # noqa: E402
from gnome3d.mc import jax as mc_jax  # noqa: E402
from gnome3d.mc.jax_driver import build_mc_kernel  # noqa: E402
from gnome3d.mc.terms import ARC_SPRINGS, CONFINEMENT, EXCLUDED_VOLUME  # noqa: E402
from gnome3d.mc.terms.arc_springs import ArcP  # noqa: E402
from gnome3d.mc.terms.confinement import ConfP  # noqa: E402
from gnome3d.mc.terms.excluded_volume import ExclP  # noqa: E402
from gnome3d.pipeline import coarse as cb  # noqa: E402
from gnome3d.pipeline.coarse.stages import build_coarse_dag  # noqa: E402
from gnome3d.pipeline.executor import SerialExecutor  # noqa: E402


def main() -> int:
    s, bed, data, _ = load_region()
    s.mc_backend = "jax"
    s.jax_bucket_shapes = True
    s.mc_stop_steps = 2000  # smaller batch so the test converges quickly

    wbed = parse_region("chr1:1000000-30000000")
    wdata = ContactData.from_files(s, [wbed.chr], wbed)
    state = cb.build_state(s, wdata, [wbed.chr], wbed)
    spine, _ = build_coarse_dag(state, 0, fan_out=False)
    SerialExecutor().run(spine)
    seeds = [ib.seed for ib in skeleton.gather_all_ib_seeds(state, 0)]
    problems = [
        {"pos": sd.anchor_seed_pos, "exp_dist": sd.exp_dist, "step_size": sd.step_size_arcs}
        for sd in seeds
        if sd.anchor_seed_pos.shape[0] > 1
    ]
    print(f"arcs IBs: {len(problems)} (anchors {[p['pos'].shape[0] for p in problems]})")

    assert mc_jax._ensure_jax()
    jax = mc_jax._jax
    jnp = mc_jax._jnp

    B = max(mc_jax._bucket_for(int(p["pos"].shape[0])) for p in problems)
    preps = [mc_jax._prep_arcs_problem_np(p["pos"], p["exp_dist"], s, B) for p in problems]
    K = len(preps)

    def stack(key, dt):
        return jnp.asarray(np.array([pr[key] for pr in preps], dtype=dt))

    pos_k = jnp.asarray(np.stack([pr["pos"] for pr in preps], axis=0))
    exp_k = jnp.asarray(np.stack([pr["exp_mat"] for pr in preps], axis=0))
    n_active_k = stack("n_active", np.int32)
    excl_r0_k = stack("excl_r0", np.float32)
    cx, cy, cz, cR = stack("conf_cx", np.float32), stack("conf_cy", np.float32), stack("conf_cz", np.float32), stack("conf_R", np.float32)
    step_k = jnp.asarray(np.array([float(p["step_size"]) for p in problems], dtype=np.float32))

    excl_skip = int(s.exclusion_skip_neighbors)
    n_steps = int(s.mc_stop_steps)
    use_excl = bool(s.use_excluded_volume) and bool(s.exclusion_apply_to_arcs)
    use_conf = bool(s.use_confinement) and bool(s.confinement_apply_to_arcs)
    ew = float(s.exclusion_weight) if use_excl else 0.0
    cw = float(s.confinement_weight) if use_conf else 0.0
    stv, sqv = float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs)
    f32 = jnp.float32
    base_key = jax.random.PRNGKey(0)

    # ---- per-IB init scores (shared by both paths; verbatim term inits) ----
    def init_one(i):
        na = jnp.int32(int(np.asarray(n_active_k[i])))
        ss = ARC_SPRINGS.jax_init(pos_k[i], ArcP(exp_k[i], f32(stv), f32(sqv)), na)
        se = EXCLUDED_VOLUME.jax_init(pos_k[i], ExclP(excl_r0_k[i], f32(ew), jnp.int32(excl_skip)), na) if use_excl else f32(0.0)
        sc = CONFINEMENT.jax_init(pos_k[i], ConfP(cx[i], cy[i], cz[i], cR[i], f32(cw)), na) if use_conf else f32(0.0)
        return ss, se, sc

    inits = [init_one(i) for i in range(K)]
    ss_k = jnp.asarray([x[0] for x in inits])
    se_k = jnp.asarray([x[1] for x in inits])
    sc_k = jnp.asarray([x[2] for x in inits])

    # ---- OLD kernel ----
    bundle = mc_jax._build_arcs_kernel(n_steps, excl_skip)
    old_kfmp = bundle[4]
    out_old = old_kfmp(
        pos_k, ss_k, se_k, sc_k, f32(s.max_temp), exp_k, step_k,
        f32(s.dt_temp), f32(s.jump_scale), f32(s.jump_coef), f32(stv), f32(sqv),
        excl_r0_k, f32(ew), cx, cy, cz, cR, f32(cw), base_key,
        f32(s.mc_stop_improvement), jnp.int32(s.mc_stop_successes), f32(1e-5), f32(0.9999), n_active_k,
    )
    pos_old = np.asarray(out_old[0])
    score_old = np.asarray(out_old[1] + out_old[2] + out_old[3])

    # ---- NEW generic driver ----
    kfmp = build_mc_kernel([ARC_SPRINGS, EXCLUDED_VOLUME, CONFINEMENT], n_steps, strict_accept=False, freeze_converged=True)
    onesK = jnp.ones((K,), jnp.float32)
    term_params = (
        ArcP(exp_k, onesK * stv, onesK * sqv),
        ExclP(excl_r0_k, onesK * ew, (onesK * excl_skip).astype(jnp.int32)),
        ConfP(cx, cy, cz, cR, onesK * cw),
    )
    movable_k = jnp.broadcast_to(jnp.arange(B, dtype=jnp.int32), (K, B))
    anchor_dummy = jnp.zeros((K, 1, 3), jnp.float32)
    out_new = kfmp(
        pos_k, (ss_k, se_k, sc_k), anchor_dummy, f32(s.max_temp), term_params, movable_k, step_k,
        f32(s.dt_temp), f32(s.jump_scale), f32(s.jump_coef), base_key,
        f32(s.mc_stop_improvement), jnp.int32(s.mc_stop_successes), f32(1e-5), f32(0.9999), n_active_k, n_active_k,
    )
    pos_new = np.asarray(out_new[0])
    score_new = np.asarray(out_new[1][0] + out_new[1][1] + out_new[1][2])

    pos_diff = float(np.max(np.abs(pos_old - pos_new)))
    score_diff = float(np.max(np.abs(score_old - score_new)))
    iters_old = int(np.asarray(out_old[4]))
    iters_new = int(np.asarray(out_new[3]))
    print(f"  K={K} B={B}  iters old={iters_old} new={iters_new}")
    print(f"  max |Δpos| = {pos_diff:.3e}   max |Δscore| = {score_diff:.3e}")
    exact = pos_diff == 0.0 and score_diff == 0.0
    print("PASS (byte-exact: generic driver == hand-written arcs kernel)" if exact else "FAILED")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
