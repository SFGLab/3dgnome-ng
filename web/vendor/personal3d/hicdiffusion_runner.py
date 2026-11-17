"""Run HiCDiffusion on many sequences, inside HiCDiffusion's own environment.

Started by personal3d.hicdiffusion with HiCDiffusion's interpreter and its checkout as the
working directory, so its modules import. Reads an npz of sequences coded A0 C1 T2 G3 N4, the
model's own coding, one row of 2,097,152 bases per sequence, and writes an npz of the two maps
the model makes per sequence, 256 by 256 at 8,192 bp, both on the log of one plus the count.

  cond, the encoder decoder's map, deterministic
  pred, the diffusion's refinement of it, one per seed

Every sequence is sampled with the same seeds, so the maps of two haplotypes of one window
differ by their sequences and not by the sampling.

    python hicdiffusion_runner.py <in.npz> <out.npz> <model.ckpt> <encoder_decoder.ckpt> <seed> [<seed> ...]
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fun

SIZE = 2_097_152


def main() -> None:
    # HiCDiffusion's modules sit in its checkout, the working directory, not beside this file.
    sys.path.insert(0, os.getcwd())
    import lightning.pytorch as pl
    from hicdiffusion_model import HiCDiffusion

    src, dst, model_ckpt, ed_ckpt = sys.argv[1:5]
    seeds = [int(s) for s in sys.argv[5:]] or [1996]
    seqs = np.load(src)["seqs"]
    model = HiCDiffusion.load_from_checkpoint(model_ckpt, encoder_decoder_model=ed_ckpt)
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    cond = np.zeros((len(seqs), 256, 256), dtype=np.float32)
    pred = np.zeros((len(seqs), len(seeds), 256, 256), dtype=np.float32)
    with torch.no_grad():
        for k, s in enumerate(seqs):
            x = Fun.one_hot(torch.from_numpy(s.astype(np.int64)), 5).to(torch.float).T.reshape(1, 5, SIZE).to(dev)
            ed = model.encoder_decoder
            y = ed.decoder(ed.encoder(x))
            cond[k] = ed.reduce_layer(y).view(256, 256).cpu().numpy()
            for m, seed in enumerate(seeds):
                pl.seed_everything(seed)
                pred[k, m] = model(x).view(256, 256).cpu().numpy()
            print(f"  sequence {k + 1} of {len(seqs)}", flush=True)
    np.savez_compressed(dst, cond=cond, pred=pred, seeds=np.array(seeds))


if __name__ == "__main__":
    main()
