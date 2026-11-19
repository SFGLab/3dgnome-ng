"""The contact map HiCDiffusion predicts for one sequence, run inside HiCDiffusion's environment.

Started by gnome3d_web.hicdiffusion with HiCDiffusion's interpreter and its checkout as the
working directory. Its modules import. Reads an npz with `seq`, 2,097,152 bases coded A0 C1
T2 G3 N4, the model's own coding. Writes an npz with `cond`, the encoder decoder's map of 256 by
256 bins of 8,192 bp on the log of one plus the count. The map is deterministic. The diffusion
that refines it is not run. A job reads this map alone.

This file stays alone in its directory, see RUNNER in gnome3d_web/hicdiffusion.py.

    python runner.py <in.npz> <out.npz> <model.ckpt> <encoder_decoder.ckpt>
"""

import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as Fun

SIZE = 2_097_152


def main() -> None:
    sys.path.insert(0, os.getcwd())
    # The model builds its training metrics on load and each warns about its memory.
    warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")
    # The diffusion package decorates with a torch call that newer torch names differently.
    warnings.filterwarnings("ignore", category=FutureWarning, module="denoising_diffusion_pytorch")
    from hicdiffusion_model import HiCDiffusion

    src, dst, model_ckpt, ed_ckpt = sys.argv[1:5]
    t0 = time.time()
    seq = np.load(src)["seq"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "the CPU"
    print(f"HiCDiffusion on {name}", flush=True)
    model = HiCDiffusion.load_from_checkpoint(model_ckpt, encoder_decoder_model=ed_ckpt, map_location=dev)
    model.eval()
    ed = model.encoder_decoder.to(dev)
    with torch.no_grad():
        x = Fun.one_hot(torch.from_numpy(seq.astype(np.int64)), 5).to(torch.float).T.reshape(1, 5, SIZE).to(dev)
        cond = ed.reduce_layer(ed.decoder(ed.encoder(x))).view(256, 256).cpu().numpy()
    np.savez_compressed(dst, cond=cond)
    peak = f", {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB of GPU memory at the peak" if dev.type == "cuda" else ""
    print(f"HiCDiffusion predicted the map in {time.time() - t0:.0f} s{peak}", flush=True)


if __name__ == "__main__":
    main()
