"""Motion correction via Suite2p, adapted from legacy pipeline."""

from __future__ import annotations

import json
import logging
import shutil
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import tifffile
import suite2p

from ...metadata.models import AnimalMetadata

logger = logging.getLogger(__name__)


def load_global_ops(path: Path) -> Dict:
    """Load the shared Suite2p ops dictionary from ``.npy``."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Suite2p ops file not found: {path}")
    ops = np.load(path, allow_pickle=True).item()
    if not isinstance(ops, dict):
        raise TypeError("Suite2p ops file did not contain a dictionary")
    return ops


def run_suite2p_one_plane(
    plane_tiff: Path,
    ops_template: Dict,
    save_path: Path,
    fps: float,
    fast_disk: Optional[Path] = None,
) -> None:
    """Run Suite2p on a single plane TIFF using provided ops template."""

    import copy

    ops = copy.deepcopy(ops_template)
    ops["input_format"] = "tif"
    ops["fs"] = fps
    ops["tiff_list"] = [plane_tiff.name]
    ops["data_path"] = [str(plane_tiff.parent)]
    ops["save_path0"] = str(save_path)
    ops["save_path"] = str(save_path)
    ops["nplanes"] = 1
    ops["planeID"] = 0
    ops["keep_movie_raw"] = False
    ops["save_tiff"] = True

    if fast_disk:
        ops["fast_disk"] = str(fast_disk)
    else:
        ops.setdefault("fast_disk", [])

    with tifffile.TiffFile(plane_tiff) as tif:
        n_frames = len(tif.pages)

    batch_size = ops.get("batch_size")
    if batch_size is None:
        batch_size = min(400, n_frames)
    else:
        batch_size = max(1, min(int(batch_size), 400, n_frames))
    ops["batch_size"] = batch_size

    suite2p_tmp = save_path / 'suite2p'
    if suite2p_tmp.exists():
        shutil.rmtree(suite2p_tmp)

    suite2p.run_s2p(ops=ops)

def move_suite2p_outputs(
    animal_id: str,
    plane_idx: int,
    suite2p_folder: Path,
    motion_output: Path,
    segmentation_output: Path,
    dest_motion: Path,
) -> Dict[str, Path]:
    """Collate Suite2p outputs and clean project layout."""

    reg_folder = suite2p_folder / "suite2p" / "plane0" / "reg_tif"
    if not reg_folder.exists():
        raise FileNotFoundError(f"Suite2p reg_tif folder missing: {reg_folder}")

    def _chunk_index(path: Path) -> int:
        match = re.search(r"file(\d+)", path.name)
        if not match:
            raise ValueError(f"Unexpected Suite2p chunk name: {path.name}")
        return int(match.group(1))

    tiff_files = sorted(reg_folder.glob("*.tif"), key=_chunk_index)
    if not tiff_files:
        raise FileNotFoundError(f"No registered TIFF chunks found in {reg_folder}")

    motion_output.mkdir(parents=True, exist_ok=True)
    if dest_motion.exists():
        dest_motion.unlink()

    with tifffile.TiffWriter(dest_motion, bigtiff=True) as writer:
        for chunk_path in tiff_files:
            with tifffile.TiffFile(chunk_path) as tif:
                for page in tif.pages:
                    writer.write(page.asarray(), contiguous=True)

    for chunk_path in tiff_files:
        chunk_path.unlink()

    plane_folder = suite2p_folder / "suite2p" / "plane0"
    if not plane_folder.exists():
        raise FileNotFoundError(f"Suite2p plane folder missing: {plane_folder}")

    if segmentation_output.exists():
        shutil.rmtree(segmentation_output)
    segmentation_output.mkdir(parents=True, exist_ok=True)

    for seg_file in plane_folder.glob("*.npy"):
        new_name = f"{animal_id}_plane{plane_idx}_{seg_file.name}"
        dest_file = segmentation_output / new_name
        shutil.move(str(seg_file), str(dest_file))

    outputs: Dict[str, Path] = {
        "motion_tiff": dest_motion,
        "segmentation_folder": segmentation_output,
    }

    shutil.rmtree(suite2p_folder / "suite2p", ignore_errors=True)

    return outputs


def run_motion_correction(
    *,
    animal: AnimalMetadata,
    plane_idx: int,
    plane_tiff: Path,
    ops_template: Dict,
    fps: float,
    output_root: Path,
    fast_disk: Optional[Path] = None,
    reprocess: bool = False,
    session_id: Optional[str] = None,
    motion_output_subdir: str | Path = Path("02_motionCorrected"),
    suite2p_output_subdir: str | Path = Path("03_suite2p"),
    plane_folder_template: str = "plane{plane_index}",
    segmentation_folder_template: str = "plane{plane_index}",
    motion_filename_template: str = "{animal_id}_plane{plane_index}_mcorrected.tif",
    metadata_filename: str = "motion_metadata.json",
) -> Dict[str, Path]:
    """Run Suite2p on one plane and organize outputs under output_root.

    Returns a mapping with keys like "motion_tiff", "segmentation_folder", and
    "metadata".
    """

    output_root = Path(output_root)
    motion_output_subdir = Path(motion_output_subdir)
    suite2p_output_subdir = Path(suite2p_output_subdir)

    context = {
        "animal_id": animal.animal_id,
        "plane_index": plane_idx,
    }
    if session_id is not None:
        context["session_id"] = session_id

    plane_folder = plane_folder_template.format(**context)
    segmentation_folder = segmentation_folder_template.format(**context)

    motion_output = output_root / motion_output_subdir / plane_folder
    segmentation_output = output_root / suite2p_output_subdir / segmentation_folder
    metadata_name = metadata_filename.format(**context)
    metadata_path = motion_output / metadata_name

    if metadata_path.exists() and not reprocess:
        logger.info("Skipping plane %d (metadata exists at %s)", plane_idx, metadata_path)
        return {"metadata": metadata_path}

    run_suite2p_one_plane(
        plane_tiff=plane_tiff,
        ops_template=ops_template,
        save_path=output_root,
        fps=fps,
        fast_disk=fast_disk,
    )

    dest_motion = motion_output / motion_filename_template.format(**context)
    outputs = move_suite2p_outputs(
        animal_id=animal.animal_id,
        plane_idx=plane_idx,
        suite2p_folder=output_root,
        motion_output=motion_output,
        segmentation_output=segmentation_output,
        dest_motion=dest_motion,
    )

    metadata = {
        "animal_id": animal.animal_id,
        "plane_idx": plane_idx,
        "plane_tiff": str(plane_tiff),
        "fps": fps,
        "fast_disk": str(fast_disk) if fast_disk else None,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    outputs["metadata"] = metadata_path
    return outputs
