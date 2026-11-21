"""The HiCDiffusion step of a job, the contact map predicted from the haplotype's sequence.

HiCDiffusion, Chiliński and Plewczynski 2024, runs in its own interpreter. Its package
pins clash with the engine's. Four variables name its parts.

HICDIFFUSION_DIR     the checkout
HICDIFFUSION_PYTHON  the command that starts its interpreter
HICDIFFUSION_MODELS  the directory with hicdiffusion.ckpt and hicdiffusion_encoder_decoder.ckpt
REFERENCE_FASTA      the hg38 FASTA, with its .fai index beside it

The Docker image holds the checkout and the interpreter and sets all four, the models and the
FASTA below /data. The step is offered when every part is in place.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from personal3d.hicdiffusion import BIN, Reference, encode, haplotype_window

VARIABLES = ("HICDIFFUSION_DIR", "HICDIFFUSION_PYTHON", "HICDIFFUSION_MODELS", "REFERENCE_FASTA")
CHECKPOINTS = ("hicdiffusion.ckpt", "hicdiffusion_encoder_decoder.ckpt")
# The runner has a directory of its own. Python puts a script's directory first on its path
# and beside the site's modules an import inside torch could find one of them.
RUNNER = Path(__file__).resolve().parent / "hicdiffusion_env" / "runner.py"
# The predicted map of a job, in the job's directory.
MAP_FILE = "hicdiffusion.bedpe"


def missing(env: Mapping[str, str] | None = None) -> list[str]:
    """What the step lacks on this server, empty when it can run.

    @param env the environment to read, the process's own by default
    @return one phrase per variable that is not set or names a part that is not there
    """
    env = os.environ if env is None else env
    out = [f"{k} is not set" for k in VARIABLES if not env.get(k)]
    if out:
        return out
    if not (Path(env["HICDIFFUSION_DIR"]) / "hicdiffusion_model.py").is_file():
        out.append("HICDIFFUSION_DIR has no hicdiffusion_model.py")
    if shutil.which(shlex.split(env["HICDIFFUSION_PYTHON"])[0]) is None:
        out.append("HICDIFFUSION_PYTHON does not start")
    out += [f"HICDIFFUSION_MODELS has no {c}" for c in CHECKPOINTS if not (Path(env["HICDIFFUSION_MODELS"]) / c).is_file()]
    fasta = Path(env["REFERENCE_FASTA"])
    if not fasta.is_file():
        out.append("REFERENCE_FASTA is not a file")
    elif not Path(f"{fasta}.fai").is_file():
        out.append("REFERENCE_FASTA has no .fai index")
    return out


def predict(chrom: str, start: int, svs: list, log: Path, env_extra: Mapping[str, str] | None = None) -> np.ndarray:  # type: ignore[type-arg]
    """The predicted map of the 2 Mb window from `start` on the haplotype.

    @param chrom the chromosome of the window
    @param start the reference position the window begins at, 0 based
    @param svs the haplotype's structural variants, applied to the reference sequence
    @param log the file that receives the model's output
    @param env_extra added to the model's environment, the GPU of a queue slot for example
    @return 256 by 256 bins of 8,192 bp on the log of one plus the count
    """
    window = haplotype_window(Reference(Path(os.environ["REFERENCE_FASTA"])), chrom, start, svs)
    models = Path(os.environ["HICDIFFUSION_MODELS"])
    # The site's own modules are on PYTHONPATH and are not for the model's interpreter.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | dict(env_extra or {})
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(src, seq=encode(window.seq))
        cmd = [*shlex.split(os.environ["HICDIFFUSION_PYTHON"]), str(RUNNER), str(src), str(dst), *(str(models / c) for c in CHECKPOINTS)]
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=os.environ["HICDIFFUSION_DIR"], env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"HiCDiffusion exited with {rc}")
        return np.load(dst)["cond"]


def write_map(path: Path, chrom: str, start: int, cond: np.ndarray) -> None:  # type: ignore[type-arg]
    """Write the predicted map as bin pairs, one row per pair of the upper half and the diagonal.

    The columns are those of a contact BEDPE. The seventh is the predicted count, the mean of the
    two halves of the prediction.

    @param start the position the window begins at, in the haplotype's coordinates
    @param cond the prediction on the log of one plus the count
    """
    counts = np.expm1(np.clip(cond, 0, None))
    counts = (counts + counts.T) / 2
    i, j = np.triu_indices(counts.shape[0])
    rows = pd.DataFrame({0: chrom, 1: start + i * BIN, 2: start + (i + 1) * BIN, 3: chrom, 4: start + j * BIN, 5: start + (j + 1) * BIN, 6: np.round(counts[i, j], 3)})
    rows.to_csv(path, sep="\t", header=False, index=False)


def read_map(path: Path) -> tuple[str, int, int, np.ndarray]:  # type: ignore[type-arg]
    """The map `write_map` wrote.

    @return the chromosome, the start of the window, the bin and the symmetric counts
    """
    rows = pd.read_csv(path, sep="\t", header=None)
    start, step = int(rows[1].min()), int(rows[2].iloc[0] - rows[1].iloc[0])
    n = int((rows[4].max() - start) // step) + 1
    i, j = ((rows[1] - start) // step).to_numpy(), ((rows[4] - start) // step).to_numpy()
    counts = np.zeros((n, n))
    counts[i, j] = rows[6].to_numpy()
    counts[j, i] = rows[6].to_numpy()
    return str(rows[0].iloc[0]), start, step, counts
