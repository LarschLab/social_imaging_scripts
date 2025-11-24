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
    pixels_xyz: Tuple[int, int, int]
    flip_horizontal: bool
    flip_z: bool
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
    flip_z: bool,
    rotation_deg: float | None = None,
    rotation_offset_deg: float = 0.0,
    rotation_offset_signed: bool = False,
    apply_rotation: bool = True,
    apply_flips: bool = True,
    gui_translation_px: Optional[Tuple[float, float]] = None,
    gui_source: Optional[str] = None,
    gui_display_channel: Optional[str] = None,
    reprocess: bool = False,
    raw_path_override: Optional[Path] = None,
    plane_spacing_um: Optional[float] = None,
) -> ConfocalPreprocessOutputs:
    """Split a confocal LSM stack into per-channel TIFF volumes."""

    session_id = session.session_id
    output_dir = cfg_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / metadata_filename.format(session_id=session_id, animal_id=animal.animal_id)

    if metadata_path.exists() and not reprocess:
        payload = json.loads(metadata_path.read_text())
        channel_paths = {name: Path(path) for name, path in payload.get("channels", {}).items()}
        voxel = payload.get("voxel_size_um")
        plane_spacing = payload.get("plane_spacing_um")
        if voxel is None or plane_spacing is None:
            raise ValueError(f"Confocal metadata missing voxel_size_um/plane_spacing_um: {metadata_path}")
        flip = bool(payload.get("flip_horizontal", False))
        flip_axial = bool(payload.get("flip_z", False))
        pixels = payload.get("pixels_xyz")
        if pixels is None:
            example = next(iter(channel_paths.values()), None)
            if example and example.exists():
                example_shape = tifffile.imread(example).shape
                pixels = [int(example_shape[0]), int(example_shape[1]), int(example_shape[2])]
            else:
                raise ValueError(f"Confocal metadata missing pixels_xyz and no channel file to infer: {metadata_path}")
        return ConfocalPreprocessOutputs(
            session_id=session_id,
            metadata_path=metadata_path,
            channel_paths=channel_paths,
            voxel_size_um=(float(voxel[0]), float(voxel[1]), float(voxel[2])),
            pixels_xyz=(int(pixels[0]), int(pixels[1]), int(pixels[2])),
            flip_horizontal=flip,
            flip_z=flip_axial,
            reused=True,
        )

    if plane_spacing_um is None:
        plane_spacing_um = getattr(session.session_data, "plane_spacing", None)
    if plane_spacing_um is None:
        raise ValueError("plane_spacing must be provided for confocal preprocessing")

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
        float(plane_spacing_um),
    )
    if np.allclose([voxel[0], voxel[1]], (1.0, 1.0)):
        import logging
        logging.getLogger(__name__).warning(
            "Confocal pixel size reported as (1.0, 1.0) µm for %s; header likely missing correct spacing",
            raw_path.name,
        )

    # Apply flips first (before rotation) if requested
    flips_applied = False
    if apply_flips:
        if flip_horizontal:
            stack = np.flip(stack, axis=-1)
            flips_applied = True
        if flip_z:
            stack = np.flip(stack, axis=0)
            flips_applied = True

    # Apply rotation (about Z) if provided
    rot_applied = False
    rot_value_raw = float(rotation_deg) if rotation_deg is not None else None
    if rot_value_raw is not None:
        offset = float(rotation_offset_deg)
        if rotation_offset_signed:
            import math as _math
            offset = _math.copysign(offset, rot_value_raw if abs(rot_value_raw) > 1e-12 else 1.0)
        rot_value_applied = rot_value_raw + offset
        rot_value_applied = ((rot_value_applied + 180.0) % 360.0) - 180.0
        if abs(rot_value_applied - 180.0) < 1e-6:
            rot_value_applied = -180.0
    else:
        rot_value_applied = None
    if apply_rotation and rot_value_applied is not None and abs(rot_value_applied) > 1e-9:
        try:
            import SimpleITK as sitk  # local import to avoid hard dependency at import time

            theta = float(rot_value_applied) * np.pi / 180.0
            z_sz, n_channels, y_sz, x_sz = stack.shape
            center_phys = (
                0.5 * (x_sz - 1) * voxel[0],
                0.5 * (y_sz - 1) * voxel[1],
                0.5 * (z_sz - 1) * voxel[2],
            )

            def _rotate_volume(vol_zyx: np.ndarray) -> np.ndarray:
                img = sitk.GetImageFromArray(vol_zyx.astype(np.float32, copy=False))
                img.SetSpacing((float(voxel[0]), float(voxel[1]), float(voxel[2])))
                tx = sitk.Euler3DTransform()
                tx.SetCenter(center_phys)
                tx.SetRotation(0.0, 0.0, theta)
                
                # Calculate output size to fit rotated image without cropping
                # For rotation about Z-axis, compute bounding box in XY plane
                cos_theta = abs(np.cos(theta))
                sin_theta = abs(np.sin(theta))
                new_x = int(np.ceil(x_sz * cos_theta + y_sz * sin_theta))
                new_y = int(np.ceil(x_sz * sin_theta + y_sz * cos_theta))
                new_size = (new_x, new_y, z_sz)
                
                # Adjust origin to center the rotated content
                new_origin = (
                    center_phys[0] - 0.5 * (new_x - 1) * voxel[0],
                    center_phys[1] - 0.5 * (new_y - 1) * voxel[1],
                    center_phys[2] - 0.5 * (z_sz - 1) * voxel[2],
                )
                
                out = sitk.Resample(
                    img,
                    new_size,  # expanded size to fit rotated content
                    tx,
                    sitk.sitkLinear,
                    new_origin,
                    img.GetSpacing(),
                    img.GetDirection(),
                    0.0,
                    img.GetPixelID(),
                )
                return sitk.GetArrayFromImage(out).astype(np.float32, copy=False)

            # Rotate first channel to get output shape
            first_rotated = _rotate_volume(stack[:, 0, :, :])
            new_shape = (first_rotated.shape[0], stack.shape[1], first_rotated.shape[1], first_rotated.shape[2])
            rotated = np.empty(new_shape, dtype=stack.dtype)
            rotated[:, 0, :, :] = first_rotated
            
            # Rotate remaining channels
            for c in range(1, stack.shape[1]):
                rotated[:, c, :, :] = _rotate_volume(stack[:, c, :, :])
            stack = rotated
            rot_applied = True
        except Exception:
            rot_applied = False

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
        # ImageJ-compatible scaling metadata: spacing along Z is plane_spacing_um; XY from header
        res = 1e4 / float(voxel[0]) if voxel[0] > 0 else None
        tifffile.imwrite(
            channel_path,
            channel_data.astype(np.float32, copy=False),
            imagej=True,
            metadata={
                "axes": "ZYX",
                "spacing": float(plane_spacing_um),
                "unit": "um",
            },
            resolution=(res, res) if res is not None else None,
            resolutionunit="CENTIMETER" if res is not None else None,
        )
        channel_paths[name] = channel_path

    metadata = {
        "animal_id": animal.animal_id,
        "session_id": session_id,
        "raw_path": str(raw_path),
        "flip_horizontal": flip_horizontal,
        "flip_z": flip_z,
        "apply_rotation": bool(apply_rotation),
        "apply_flips": bool(apply_flips),
        "gui_transform_applied": bool(rot_applied or flips_applied),
        # Record both raw GUI rotation and applied rotation (with offset)
        "gui_rotation_deg_raw": float(rot_value_raw) if rot_value_raw is not None else None,
        "gui_rotation_offset_deg": float(rotation_offset_deg),
        "gui_rotation_offset_signed": bool(rotation_offset_signed),
        "gui_rotation_deg_applied": float(rot_value_applied) if rot_value_applied is not None else None,
        # Maintain legacy key pointing to applied value for compatibility
        "gui_rotation_deg": float(rot_value_applied) if rot_value_applied is not None else None,
        "gui_translation_px": list(gui_translation_px) if gui_translation_px is not None else None,
        "gui_source": gui_source or None,
        "gui_display_channel": gui_display_channel or None,
        "voxel_size_um": list(voxel),
        "plane_spacing_um": float(plane_spacing_um),
        # pixels_xyz reflects the actual output shape (may be expanded after rotation)
        # Stack shape is (z, channels, y, x), stored as [z, y, x] for legacy compatibility
        "pixels_xyz": [int(stack.shape[0]), int(stack.shape[2]), int(stack.shape[3])],
        "channels": {name: str(path) for name, path in channel_paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return ConfocalPreprocessOutputs(
        session_id=session_id,
        metadata_path=metadata_path,
        channel_paths=channel_paths,
        voxel_size_um=voxel,
        # Stack shape is (z, channels, y, x), stored as (z, y, x)
        pixels_xyz=(int(stack.shape[0]), int(stack.shape[2]), int(stack.shape[3])),
        flip_horizontal=flip_horizontal,
        flip_z=flip_z,
        reused=False,
    )
