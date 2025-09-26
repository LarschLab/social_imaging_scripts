"""Preprocessing helpers for two-photon anatomy stacks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from ...metadata.models import TwoPhotonPreprocessing
from . import utils

logger = logging.getLogger(__name__)


def run(
    *,
    animal_id: str,
    session_id: str,
    raw_dir: Path,
    output_root: Path,
    settings: TwoPhotonPreprocessing | None = None,
) -> Dict[str, Path]:
    """Preprocess an anatomy stack.

    The anatomy acquisition is typically a single multi-page TIFF. We simply
    concatenate all TIFF files in ``raw_dir`` (sorted lexicographically), apply
    negative-value correction, and save the result to ``output_root``. When
    *settings* is provided it is echoed in the metadata for traceability, but
    currently no additional processing is performed.
    """

    raw_dir = Path(raw_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tiff_paths = sorted(raw_dir.glob("*.tif"))
    if not tiff_paths:
        raise FileNotFoundError(f"No anatomy TIFF files found in {raw_dir}")

    stacks = [utils.load_tiff_stack(path) for path in tiff_paths]
    concatenated = np.concatenate(stacks, axis=0)
    corrected = utils.correct_negative_values(concatenated)

    stack_path = output_root / f"{animal_id}_anatomy_stack.tif"
    utils.save_tiff_stack(stack_path, corrected)

    metadata = {
        "session_id": session_id,
        "animal_id": animal_id,
        "mode": settings.mode if settings else "linear",
        "n_planes": settings.n_planes if settings else None,
        "frames_per_plane": settings.frames_per_plane if settings else None,
        "flyback_frames": settings.flyback_frames if settings else None,
        "remove_first_frame": settings.remove_first_frame if settings else None,
        "blocks": settings.blocks if settings else None,
        "output_stack": str(stack_path),
    }
    metadata_path = output_root / f"{animal_id}_anatomy_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {"stack": stack_path, "metadata": metadata_path}
