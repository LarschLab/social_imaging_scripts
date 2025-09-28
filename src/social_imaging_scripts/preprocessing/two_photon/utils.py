"""Shared helpers for two-photon preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import logging
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def discover_tiff_blocks(raw_dir: Path, blocks: Optional[Iterable[int]] = None) -> List[Path]:
    """Return TIFF files in *raw_dir* filtered by the optional *blocks* list."""

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    candidates = sorted(raw_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No TIFF files found in {raw_dir}")

    def block_id(path: Path) -> Optional[int]:
        name = path.stem
        suffix = name.split("_")[-1]
        return int(suffix) if suffix.isdigit() else None

    selected: List[Path] = []
    block_filter = set(blocks) if blocks is not None else None

    for path in candidates:
        bid = block_id(path)
        if block_filter is not None and bid not in block_filter:
            continue
        selected.append(path)

    if not selected:
        raise ValueError(
            f"No TIFF files match requested blocks {sorted(block_filter or [])} in {raw_dir}"
        )

    selected.sort(key=lambda p: (block_id(p) is None, block_id(p) if block_id(p) is not None else p.name))
    return selected


def load_tiff_stack(path: Path) -> np.ndarray:
    """Load a multi-page TIFF as ``(frames, height, width)`` array."""

    with tifffile.TiffFile(path) as tif:
        # Some microscope exports contain inconsistent metadata about the
        # intended stack shape (e.g., ImageJ tags claiming fewer slices).
        # Reading page-by-page avoids reshape warnings while yielding the
        # actual acquired frames.
        frames = [page.asarray() for page in tif.pages]

    stack = np.stack(frames, axis=0)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    return np.asarray(stack)


def correct_negative_values(stack: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
    """Shift negative pixel values into the positive ``uint16`` range.

    Processing happens in slabs along the frame axis to keep memory bounded.
    """

    stack = np.asarray(stack)
    min_val = int(stack.min())
    if min_val >= 0 and stack.dtype == np.uint16:
        return stack

    offset = -min_val if min_val < 0 else 0
    if offset == 0 and stack.dtype == np.uint16:
        return stack

    chunk_size = max(1, min(chunk_size, stack.shape[0]))
    result = np.empty(stack.shape, dtype=np.uint16)

    for start in range(0, stack.shape[0], chunk_size):
        stop = min(start + chunk_size, stack.shape[0])
        chunk = stack[start:stop].astype(np.int32, copy=False)
        if offset:
            chunk += offset
        np.clip(chunk, 0, np.iinfo(np.uint16).max, out=chunk)
        result[start:stop] = chunk.astype(np.uint16, copy=False)

    return result


def save_tiff_stack(path: Path, data: np.ndarray) -> None:
    """Persist ``data`` as a TIFF stack (uint16)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, data.astype(np.uint16, copy=False), photometric="minisblack")


def drop_flyback_and_reshape(
    stack: np.ndarray,
    *,
    n_planes: int,
    frames_per_plane: int,
    flyback_frames: int,
    remove_first_frame: bool,
) -> np.ndarray:
    """Return stack reshaped to ``(volumes, planes, frames, H, W)``."""

    frames_per_volume = n_planes * frames_per_plane + flyback_frames
    total_frames = stack.shape[0]
    usable_frames = (total_frames // frames_per_volume) * frames_per_volume
    if usable_frames == 0:
        raise ValueError(
            "Stack does not contain a full volume given the provided parameters"
        )

    if usable_frames != total_frames:
        logger.warning(
            "Dropping %d trailing frames that do not form a complete volume",
            total_frames - usable_frames,
        )
        stack = stack[:usable_frames]

    volumes = usable_frames // frames_per_volume
    h, w = stack.shape[1:]
    stack = stack.reshape(volumes, frames_per_volume, h, w)

    if flyback_frames:
        stack = stack[:, : n_planes * frames_per_plane, :, :]

    stack = stack.reshape(volumes, n_planes, frames_per_plane, h, w)

    if remove_first_frame:
        if frames_per_plane <= 1:
            logger.warning(
                "Cannot remove first frame: frames_per_plane=%d", frames_per_plane
            )
        else:
            stack = stack[:, :, 1:, :, :]

    return stack
