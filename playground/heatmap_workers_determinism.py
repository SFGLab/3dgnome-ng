import sys, tempfile, logging
from pathlib import Path
import numpy as np
sys.path.insert(0, "harness")
import integration as it
from gnome3d.settings import Settings
from gnome3d.io import parse_region
from gnome3d.data import ContactData
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse.stages import build_coarse_dag
from gnome3d.pipeline.executor import SerialExecutor
from gnome3d import skeleton, log

log.setup(logging.WARNING)
d = Path(tempfile.mkdtemp()); cfg = d / "c.ini"
it.write_config(cfg, fast=True, use_subanchor_heatmap=True, use_anchor_heatmap=True)
s = Settings(); s.load_ini(str(cfg))
bed = parse_region("chr1:1-60000000")
data = ContactData.from_files(s, [bed.chr], bed)
state = cb.build_state(s, data, [bed.chr], bed)
spine, _ = build_coarse_dag(state, 0, fan_out=False)
SerialExecutor().run(spine)

s.heatmap_workers = 1
seeds1 = skeleton.gather_all_ib_seeds(state, 0)
s.heatmap_workers = 4
seeds4 = skeleton.gather_all_ib_seeds(state, 0)

assert len(seeds1) == len(seeds4), (len(seeds1), len(seeds4))
nh = 0
for a, b in zip(seeds1, seeds4, strict=True):
    assert a.ib_id == b.ib_id, (a.ib_id, b.ib_id)
    assert a.wants_heat == b.wants_heat, a.ib_id
    h1, h2 = a.seed.subanchor_heat_raw, b.seed.subanchor_heat_raw
    if h1 is None:
        assert h2 is None
    else:
        assert np.array_equal(h1, h2), f"heat mismatch {a.ib_id}"
        nh += 1
    assert np.array_equal(a.seed.exp_dist, b.seed.exp_dist), f"exp_dist mismatch {a.ib_id}"
print(f"DETERMINISM OK (workers=4 thread pool, {len(seeds1)>1}): {len(seeds1)} IBs, {nh} with heat, parallel(4)==serial(1) byte-identical")
