"""Confocal stack preprocessing helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile

from ..metadata.models import AnatomySession, AnimalMetadata


@dataclass
class ConfocalPreprocessOutputs:
    session_id: str
    metadata_path: Path
    channel_paths: Dict[str, Path]
    voxel_size_um: Tuple[float, float, float]
    flip_horizontal: bool
    reused: bool


def _sanitize_channel_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_")
    return safe.lower() or "channel"


def _load_confocal_stack(path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """Load confocal LSM stack as (Z, C, Y, X) float32 and extract voxel size."""

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        data = series.asarray().astype(np.float32, copy=False)
        metadata = tf.lsm_metadata or {}

    if data.ndim != 4:
        raise ValueError(f"Unexpected confocal data shape {data.shape} for {path}")
    axes = getattr(series, "axes", "")
    if axes != "ZCYX":
        # attempt to reshape if axes differ
        raise ValueError(f"Unsupported axes {axes!r} for confocal stack {path}")

    vx = float(metadata.get("VoxelSizeX", 1.0)) * 1e6
    vy = float(metadata.get("VoxelSizeY", 1.0)) * 1e6
    vz = float(metadata.get("VoxelSizeZ", 1.0)) * 1e6

    return data, {"voxel_size_x_um": vx, "voxel_size_y_um": vy, "voxel_size_z_um": vz}


def _resolve_channel_names(session: AnatomySession) -> List[str]:
    channels = getattr(session.session_data, "channels", None)
    if not channels:
        return []
    names: List[str] = []
    for channel in channels:
        if hasattr(channel, "model_dump"):
            payload = channel.model_dump()
        else:
            try:
                payload = dict(channel)
            except Exception:
                payload = {}
        label = (
            getattr(channel, "name", None)
            or getattr(channel, "marker", None)
            or payload.get("name")
            or payload.get("marker")
            or f"channel{getattr(channel, 'channel_id', payload.get('channel_id', ''))}"
        )
        names.append(_sanitize_channel_name(str(label)))
    return names


def run(
    *,
    animal: AnimalMetadata,
    session: AnatomySession,
    cfg_root: Path,
    channel_template: str,
    metadata_filename: str,
    flip_horizontal: bool,
    reprocess: bool = False,
    raw_path_override: Optional[Path] = None,
) -> ConfocalPreprocessOutputs:
    """Split a confocal LSM stack into per-channel TIFF volumes."""

    session_id = session.session_id
    output_dir = cfg_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / metadata_filename.format(session_id=session_id, animal_id=animal.animal_id)

    if metadata_path.exists() and not reprocess:
        payload = json.loads(metadata_path.read_text())
        channel_paths = {name: Path(path) for name, path in payload.get("channels", {}).items()}
        voxel = payload.get("voxel_size_um") or [1.0, 1.0, 1.0]
        flip = bool(payload.get("flip_horizontal", False))
        return ConfocalPreprocessOutputs(
            session_id=session_id,
            metadata_path=metadata_path,
            channel_paths=channel_paths,
            voxel_size_um=(float(voxel[0]), float(voxel[1]), float(voxel[2])),
            flip_horizontal=flip,
            reused=True,
        )

    raw_path = raw_path_override or Path(session.session_data.raw_path)
    if not raw_path.is_absolute():
        base = Path(animal.root_dir) if getattr(animal, "root_dir", None) else Path(".")
        raw_path = (base / raw_path).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Confocal stack not found at {raw_path}")

    stack, meta = _load_confocal_stack(raw_path)
    voxel = (
        float(meta["voxel_size_x_um"]),
        float(meta["voxel_size_y_um"]),
        float(meta["voxel_size_z_um"]),
    )

    if flip_horizontal:
        stack = np.flip(stack, axis=-1)

    channel_names = _resolve_channel_names(session)
    if not channel_names:
        channel_names = [f"channel{idx}" for idx in range(stack.shape[1])]
    if len(channel_names) != stack.shape[1]:
        raise ValueError(
            f"Channel metadata mismatch for {session_id}: expected {stack.shape[1]} entries, "
            f"found {len(channel_names)}"
        )

    channel_paths: Dict[str, Path] = {}
    for idx, name in enumerate(channel_names):
        channel_data = stack[:, idx, :, :]
        channel_filename = channel_template.format(
            animal_id=animal.animal_id,
            session_id=session_id,
            channel=name,
        )
        channel_path = output_dir / channel_filename
        tifffile.imwrite(channel_path, channel_data.astype(np.float32, copy=False))
        channel_paths[name] = channel_path

    metadata = {
        "animal_id": animal.animal_id,
        "session_id": session_id,
        "raw_path": str(raw_path),
        "flip_horizontal": flip_horizontal,
        "voxel_size_um": list(voxel),
        "channels": {name: str(path) for name, path in channel_paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return ConfocalPreprocessOutputs(
        session_id=session_id,
        metadata_path=metadata_path,
        channel_paths=channel_paths,
        voxel_size_um=voxel,
        flip_horizontal=flip_horizontal,
        reused=False,
    )
