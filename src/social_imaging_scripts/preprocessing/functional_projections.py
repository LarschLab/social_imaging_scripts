# Setup & imports
import sys, math, warnings, json
import numpy as np

from skimage.feature import match_template
from skimage.transform import rescale, resize
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float32
from skimage.filters import gaussian


from pathlib import Path
import tifffile as tiff

import pandas as pd


def normalize_to_unit_interval(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip arr to [lo, hi] and map to [0, 1] as float32."""
    denom = hi - lo if hi > lo else 1.0
    out = (arr - lo) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

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
    out_dir: Path = None,
    animal_name: str = None,
    recursive: bool = False,
):

    if out_dir is None:
        out_dir = Path(start_folder) / "projections"
    out_dir = Path(out_dir)

    files = collect_tiff_files(start_folder, recursive=recursive)
    excluded_suffixes = {"avg_projections", "max_projections"}
    filtered_files = []
    for path in files:
        # Skip anything already inside the output directory (including existing projections).
        try:
            if path.resolve().is_relative_to(out_dir.resolve()):
                continue
        except AttributeError:
            # Python <3.9 compatibility: fall back to manual check.
            if out_dir in path.resolve().parents:
                continue
        stem_lower = path.stem.lower()
        if any(stem_lower.endswith(suffix) for suffix in excluded_suffixes):
            continue
        filtered_files.append(path)

    files = filtered_files
    print(f"Found {len(files)} TIFF stack(s).")
    if not files:
        raise RuntimeError(
            f"No motion-corrected TIFF stacks found in {start_folder}; "
            "existing projection outputs are ignored automatically."
        )

    # 1) For each file: read, compute max & avg projections
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



    # 3) Normalize stacks to [0, 1]
    max_scaled = normalize_to_unit_interval(max_stack, lo_max, hi_max)
    avg_scaled = normalize_to_unit_interval(avg_stack, lo_avg, hi_avg)

    # 4) No additional rotation; the normalised stacks are kept as-is.
    max_norm = max_scaled
    avg_norm = avg_scaled

    # 5) Save ImageJ-formatted multi-page TIFFs
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use animal name if given, otherwise default
    prefix = f"{animal_name}_" if animal_name else ""

    max_path = out_dir / f"{prefix}max_projections.tif"
    avg_path = out_dir / f"{prefix}avg_projections.tif"

    # ImageJ expects axes "ZYX" for stacks
    tiff.imwrite(
        str(max_path),
        max_norm.astype(np.float32, copy=False),
        imagej=True,
        metadata={"axes": "ZYX"},
        compression="deflate",
    )
    tiff.imwrite(
        str(avg_path),
        avg_norm.astype(np.float32, copy=False),
        imagej=True,
        metadata={"axes": "ZYX"},
        compression="deflate",
    )

    metadata_payload = {
        "animal": animal_name,
        "source_files": [str(path) for path in files],
        "percentiles": {
            "max": {"lo": float(lo_max), "hi": float(hi_max)},
            "avg": {"lo": float(lo_avg), "hi": float(hi_avg)},
        },
        "dtype": "float32",
    }
    metadata_path = out_dir / f"{prefix}projections_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2))

    print("Saved:")
    print(f"  {max_path}")
    print(f"  {avg_path}")
    print(f"  {metadata_path}")
