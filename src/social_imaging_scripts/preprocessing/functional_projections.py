# Setup & imports
import sys, math, warnings, json
import numpy as np

import skimage
from skimage.feature import match_template
from skimage.transform import rescale, resize
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float32
from skimage.filters import gaussian
from skimage.transform import rotate as sk_rotate


import pandas as pd

from pathlib import Path
import numpy as np
import tifffile as tiff

def rotate_pages_uint8(stack_u8: np.ndarray, angle_deg: float) -> np.ndarray:
    from skimage.transform import rotate as sk_rotate
    rotated = np.stack(
        [sk_rotate(page, angle=angle_deg, resize=False, preserve_range=True, order=1, mode="constant", cval=0.0)
            for page in stack_u8],
        axis=0
    )
    return np.clip(rotated, 0, 255).astype(np.uint8)

def scale_to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = hi - lo if hi > lo else 1.0
    out = (arr - lo) / denom
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)

def collect_tiff_files(folder: Path, recursive: bool = False) -> list:
    """Collect all .tif/.tiff files in the given folder."""

    folder = Path(folder)
    suffixes = {".tif", ".tiff"}
    if recursive:
        return sorted(
            [
                p
                for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes
            ]
        )
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes])

def save_max_avg_projections(
    start_folder: Path,
    angle_deg: float = 140.0,
    out_dir: Path = None,
    animal_name: str = None,
    recursive: bool = False,
):

    files = collect_tiff_files(start_folder, recursive=recursive)
    print(f"Found {len(files)} TIFF stack(s).")

    # 1) For each file: read, compute max & avg projections (NO rotation yet)
    max_projs = []
    avg_projs = []
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp.name}")
        stack = tiff.imread(str(fp)).astype(np.float32)  # assume (Z, Y, X) identical across files
        max_projs.append(stack.max(axis=0))
        avg_projs.append(stack.mean(axis=0))

    # Combine projections into stacks (Z=N_files, Y, X)
    max_stack = np.stack(max_projs, axis=0)
    avg_stack = np.stack(avg_projs, axis=0)

    # 2) Compute 0th and 99.9th percentiles separately for each stack
    lo_max, hi_max = np.percentile(max_stack, [0.0, 99.9])
    lo_avg, hi_avg = np.percentile(avg_stack, [0.0, 99.9])
    print(f"Max stack percentiles: 0%={lo_max:.4g}, 99.9%={hi_max:.4g}")
    print(f"Avg stack percentiles: 0%={lo_avg:.4g}, 99.9%={hi_avg:.4g}")



    # 3) Scale stacks to uint8 (before rotation)
    max_u8 = scale_to_uint8(max_stack, lo_max, hi_max)
    avg_u8 = scale_to_uint8(avg_stack, lo_avg, hi_avg)

    # 4) Rotate each page at the very end (no resize -> no enlargement/cropping to same HxW)


    max_u8_rot = rotate_pages_uint8(max_u8, angle_deg)
    avg_u8_rot = rotate_pages_uint8(avg_u8, angle_deg)

    # 5) Save ImageJ-formatted multi-page TIFFs
    if out_dir is None:
        out_dir = start_folder / "projections"
        out_dir.mkdir(exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Use animal name if given, otherwise default
    prefix = f"{animal_name}_" if animal_name else ""

    max_path = out_dir / f"{prefix}max_projections.tif"
    avg_path = out_dir / f"{prefix}avg_projections.tif"

    # ImageJ expects axes "ZYX" for stacks
    tiff.imwrite(str(max_path), max_u8_rot, imagej=True, metadata={"axes": "ZYX"}, compression="deflate")
    tiff.imwrite(str(avg_path), avg_u8_rot, imagej=True, metadata={"axes": "ZYX"}, compression="deflate")

    print("Saved:")
    print(f"  {max_path}")
    print(f"  {avg_path}")
