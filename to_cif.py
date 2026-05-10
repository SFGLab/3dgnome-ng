#!/usr/bin/env python
"""
Convert 3D coordinate files to mmCIF format for use with structure viewers.

Supported input formats
-----------------------
.smooth.txt   Original cudaMMC smooth output:  x y z _  (space-delimited, 1 header row)
.hcm          cudaMMC / 3dgnome-torch HCM:     genomic_pos x y z  (tab-delimited, no header)
.3d           3dgnome-torch default output:     chrom start end x y z level  (tab, 1 header)

Usage
-----
# Convert a single file (format auto-detected from extension):
python to_cif.py file.3d

# Convert all .3d files in a directory:
python to_cif.py -i output/ --ext 3d

# Original workflow: run cudaMMC smooth on .hcm files, then convert .smooth.txt:
python to_cif.py -i output/ -c /path/to/cudaMMC -s 5000
"""

import argparse
import os
import sys


# ── mmCIF boilerplate ─────────────────────────────────────────────────────────

_CIF_HEADER = """\
data_3dnome
#
_entry.id 3dgnome
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.296
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
"""


def _write_cif(path: str, coords):
    """Write a sequence of (x, y, z) tuples to a mmCIF file."""
    with open(path, "w") as fh:
        fh.write(_CIF_HEADER)
        for i, (x, y, z) in enumerate(coords, start=1):
            fh.write(
                f"ATOM {i} C CA . ALA A 1 {i} ? "
                f"{x:.4f} {y:.4f} {z:.4f} 1.00 99.99 A\n"
            )
    print(f"Saved {path}")


# ── Per-format readers ────────────────────────────────────────────────────────

def _read_smooth_txt(path: str):
    """x y z _ (space-delimited, 1 header row) — original cudaMMC smooth output."""
    coords = []
    with open(path) as fh:
        next(fh)  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return coords


def _read_hcm(path: str):
    """genomic_pos x y z (tab-delimited, no header) — cudaMMC / 3dgnome HCM."""
    coords = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return coords


def _read_3d(path: str):
    """chrom start end x y z level (tab-delimited, 1 header row) — 3dgnome .3d output."""
    coords = []
    with open(path) as fh:
        next(fh)  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            coords.append((float(parts[3]), float(parts[4]), float(parts[5])))
    return coords


def _auto_reader(path: str):
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("txt",):
        return _read_smooth_txt(path)
    if ext == "hcm":
        return _read_hcm(path)
    if ext == "3d":
        return _read_3d(path)
    raise ValueError(f"Unknown extension '.{ext}' for {path}. "
                     "Use --ext to specify: smooth_txt / hcm / 3d")


def _cif_path(src: str) -> str:
    base = src
    for suffix in (".smooth.txt", ".txt", ".hcm", ".3d"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base + ".cif"


# ── Conversion helpers ────────────────────────────────────────────────────────

def convert_file(src: str, ext_override: str = None):
    if ext_override:
        readers = {"smooth_txt": _read_smooth_txt, "hcm": _read_hcm, "3d": _read_3d}
        if ext_override not in readers:
            raise ValueError(f"Unknown --ext '{ext_override}'. Choose: smooth_txt / hcm / 3d")
        coords = readers[ext_override](src)
    else:
        coords = _auto_reader(src)
    _write_cif(_cif_path(src), coords)


def run_cudammc_smooth(cudammc_path: str, hcm_path: str, smooth_num: str):
    cmd = f"{cudammc_path} -a smooth -i {hcm_path} -r {smooth_num}"
    ret = os.system(cmd)
    if ret != 0:
        print(f"  Warning: cudaMMC returned non-zero exit code for {hcm_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert 3D coordinate files to mmCIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("files", nargs="*",
                   help="Input file(s) to convert (auto-detects format from extension).")
    p.add_argument("-i", "--indirectory",
                   help="Directory to scan for input files.")
    p.add_argument("--ext", default=None,
                   choices=["smooth_txt", "hcm", "3d"],
                   help="File format when extension is ambiguous.")
    p.add_argument("-c", "--cudaMMC-path", dest="cudammc_path", default=None,
                   help="Path to cudaMMC executable. When given, runs '-a smooth' on "
                        "every .hcm file in --indirectory before converting the "
                        "resulting .smooth.txt files.")
    p.add_argument("-s", "--smooth-num", dest="smooth_num", default=None,
                   help="Smoothing resolution passed to cudaMMC (e.g. 5000).")
    return p.parse_args()


def main():
    args = parse_args()

    targets = list(args.files)

    if args.indirectory:
        d = args.indirectory

        # Optional: run cudaMMC smooth on all .hcm files first
        if args.cudammc_path:
            if not args.smooth_num:
                print("Error: --smooth-num is required when --cudaMMC-path is given.",
                      file=sys.stderr)
                sys.exit(1)
            hcms = sorted(f for f in os.listdir(d) if f.endswith(".hcm"))
            for hcm in hcms:
                print(f"Running cudaMMC smooth on {hcm}...")
                run_cudammc_smooth(args.cudammc_path,
                                   os.path.join(d, hcm),
                                   args.smooth_num)

        # Collect files to convert
        ext_map = {
            "smooth_txt": ".smooth.txt",
            "hcm":        ".hcm",
            "3d":         ".3d",
            None:         None,
        }
        target_ext = ext_map.get(args.ext)

        if args.cudammc_path:
            # After smooth step, convert the resulting .smooth.txt files
            targets += [
                os.path.join(d, f)
                for f in sorted(os.listdir(d))
                if f.endswith(".smooth.txt")
            ]
        elif target_ext is not None:
            targets += [
                os.path.join(d, f)
                for f in sorted(os.listdir(d))
                if f.endswith(target_ext)
            ]
        else:
            # Auto: pick .3d first, then .hcm, then .smooth.txt
            for ext in (".3d", ".hcm", ".smooth.txt"):
                found = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(ext)]
                if found:
                    targets += found
                    break

    if not targets:
        print("No input files found. Pass file paths or use -i with a directory.",
              file=sys.stderr)
        sys.exit(1)

    for src in targets:
        print(f"Converting {src}...")
        try:
            convert_file(src, args.ext)
        except Exception as exc:
            print(f"  Error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
