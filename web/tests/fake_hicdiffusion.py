"""Stands in for HiCDiffusion's interpreter in tests. It is started as the interpreter. Its
first argument is the runner it would run. It writes a flat map and prints the GPU it was given,
or fails when FAKE_HICDIFFUSION_FAIL is set."""

from __future__ import annotations

import os
import sys

import numpy as np

runner, src, dst, model, encoder_decoder = sys.argv[1:6]
seq = np.load(src)["seq"]
print(f"fake HiCDiffusion, {len(seq)} bases, runner {os.path.basename(runner)}, GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}, PYTHONPATH {'set' if os.environ.get('PYTHONPATH') else 'unset'}")
if os.environ.get("FAKE_HICDIFFUSION_FAIL"):
    print("the model exploded", file=sys.stderr)
    sys.exit(4)
np.savez_compressed(dst, cond=np.full((256, 256), 1.5, dtype=np.float32))
