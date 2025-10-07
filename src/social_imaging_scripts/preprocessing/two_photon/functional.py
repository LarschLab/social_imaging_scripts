"""Functional two-photon preprocessing driven by repository metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from ...metadata.models import TwoPhotonPreprocessing
from . import utils

logger = logging.getLogger(__name__)


def _select_blocks(raw_dir: Path, settings: TwoPhotonPreprocessing) -> Iterable[Path]:
    try:
        return utils.discover_tiff_blocks(raw_dir, settings.blocks)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Functional raw directory '{raw_dir}' not found"
        ) from exc


def _load_resonant_stack(
    block_paths: Iterable[Path],
    n_planes: int,
    frames_per_plane: int,
    flyback_frames: int,
    remove_first_frame: bool,
) -> np.ndarray:
    arrays = []
    for path in block_paths:
        logger.info("Loading block %s", path.name)
        block_stack = utils.load_tiff_stack(path)
        if block_stack.ndim > 3:
            block_stack = block_stack.reshape(-1, block_stack.shape[-2], block_stack.shape[-1])
        arrays.append(block_stack)
    stack = np.concatenate(arrays, axis=0)
    logger.info("Concatenated stack shape: %s", stack.shape)

    stack = utils.correct_negative_values(stack)

    reshaped = utils.drop_flyback_and_reshape(
        stack,
        n_planes=n_planes,
        frames_per_plane=frames_per_plane,
        flyback_frames=flyback_frames,
        remove_first_frame=remove_first_frame,
    )
    return reshaped


def _average_planes(reshaped: np.ndarray) -> np.ndarray:
    """Average across the frame axis per plane and return ``(volumes, planes, H, W)``."""

    avg = reshaped.astype(np.float32, copy=False).mean(axis=2)
    return np.round(avg).astype(np.uint16)


def run(
    *,
    animal_id: str,
    session_id: str,
    raw_dir: Path,
    output_root: Path,
    settings: TwoPhotonPreprocessing,
    planes_subdir: str | Path = Path("01_individualPlanes"),
    plane_filename_template: str = "{animal_id}_plane{plane_index}.tif",
    metadata_filename: str = "{animal_id}_preprocessing_metadata.json",
) -> Dict[str, Path]:
    """Preprocess a functional stack described by *settings*."""

    raw_dir = Path(raw_dir)
    output_root = Path(output_root)
    planes_subdir = Path(planes_subdir)
    output_planes = output_root / planes_subdir
    output_planes.mkdir(parents=True, exist_ok=True)

    block_paths = list(_select_blocks(raw_dir, settings))

    frames_per_plane = settings.frames_per_plane or 1
    if frames_per_plane <= 0:
        raise ValueError("frames_per_plane must be a positive integer")

    if settings.mode.lower() != "resonant":
        stack = [utils.load_tiff_stack(path) for path in block_paths]
        concatenated = np.concatenate(stack, axis=0)
        corrected = utils.correct_negative_values(concatenated)
        stack_path = output_root / f"{animal_id}_functional_stack.tif"
        utils.save_tiff_stack(stack_path, corrected)

        metadata = {
            "session_id": session_id,
            "animal_id": animal_id,
            "mode": settings.mode,
            "frames_per_plane": frames_per_plane,
            "flyback_frames": settings.flyback_frames,
            "blocks": settings.blocks,
            "output_stack": str(stack_path),
        }
        metadata_path = output_root / metadata_filename.format(
            animal_id=animal_id, session_id=session_id
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return {"stack": stack_path, "metadata": metadata_path}

    if settings.n_planes is None:
        raise ValueError("Resonant two-photon functional sessions require n_planes")

    reshaped = _load_resonant_stack(
        block_paths,
        n_planes=settings.n_planes,
        frames_per_plane=frames_per_plane,
        flyback_frames=settings.flyback_frames,
        remove_first_frame=settings.remove_first_frame,
    )

    volumes, planes, frames, height, width = reshaped.shape
    logger.info(
        "Reshaped stack into %d volumes x %d planes x %d frames (%d x %d)",
        volumes,
        planes,
        frames,
        height,
        width,
    )

    averaged = _average_planes(reshaped)

    plane_paths: Dict[str, Path] = {}
    for plane_idx in range(planes):
        plane_stack = averaged[:, plane_idx, :, :]
        plane_filename = plane_filename_template.format(
            animal_id=animal_id,
            session_id=session_id,
            plane_index=plane_idx,
        )
        plane_path = output_planes / plane_filename
        utils.save_tiff_stack(plane_path, plane_stack)
        plane_paths[f"plane_{plane_idx}"] = plane_path

    metadata = {
        "session_id": session_id,
        "animal_id": animal_id,
        "mode": settings.mode,
        "n_planes": settings.n_planes,
        "frames_per_plane": frames_per_plane,
        "flyback_frames": settings.flyback_frames,
        "remove_first_frame": settings.remove_first_frame,
        "blocks": settings.blocks,
        "output_dir": str(output_planes),
        "volumes": volumes,
    }
    metadata_path = output_planes / metadata_filename.format(
        animal_id=animal_id, session_id=session_id
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))

    result = {"metadata": metadata_path}
    result.update(plane_paths)
    return result
