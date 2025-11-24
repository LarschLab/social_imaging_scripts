"""Transform Suite2p functional ROIs into reference-brain coordinates."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import ants  # type: ignore
import numpy as np
import pandas as pd
import tifffile
from skimage.transform import AffineTransform, warp

from ..metadata.config import FunctionalRoiRegistrationQcConfig
from ..registration import align_substack

logger = logging.getLogger(__name__)


class FunctionalRoiRegistrationError(RuntimeError):
    """Raised when ROI registration cannot proceed for the requested animal/session."""


@dataclass
class FunctionalRoiRegistrationResult:
    """Summary of ROI transformation outputs for one functional session."""

    animal_id: str
    session_id: str
    output_csv: Path
    total_planes: int
    processed_planes: int
    total_rois: int
    skipped_planes: Mapping[int, str] = field(default_factory=dict)
    qc_path: Optional[Path] = None  # reference QC
    anatomy_qc_path: Optional[Path] = None
    native_qc_path: Optional[Path] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "animal_id": self.animal_id,
            "session_id": self.session_id,
            "output_csv": str(self.output_csv),
            "total_planes": int(self.total_planes),
            "processed_planes": int(self.processed_planes),
            "total_rois": int(self.total_rois),
            "skipped_planes": dict(self.skipped_planes),
            "qc_path": str(self.qc_path) if self.qc_path else None,
            "native_qc_path": str(self.native_qc_path) if self.native_qc_path else None,
            "anatomy_qc_path": str(self.anatomy_qc_path) if self.anatomy_qc_path else None,
        }


def load_suite2p_rois(
    plane_dir: Path,
    *,
    animal_id: str,
    plane_index: int,
    extra_candidates: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return ROI centre coordinates ``(y, x)`` from a Suite2p ``stat.npy`` file."""

    candidates = [
        plane_dir / f"{animal_id}_plane{plane_index}_stat.npy",
        plane_dir / "stat.npy",
    ]
    if extra_candidates:
        candidates.extend(plane_dir / name for name in extra_candidates)

    stat_path = next((path for path in candidates if path.exists()), None)
    if stat_path is None:
        raise FileNotFoundError(f"No Suite2p stat.npy found in {plane_dir}")

    stat = np.load(stat_path, allow_pickle=True)
    if stat.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    try:
        rois = np.array([entry["med"] for entry in stat], dtype=np.float32)
    except Exception as exc:  # pragma: no cover - unexpected suite2p format
        raise FunctionalRoiRegistrationError(
            f"Failed to extract ROI medians from {stat_path}"
        ) from exc

    if rois.ndim != 2 or rois.shape[1] != 2:
        raise FunctionalRoiRegistrationError(
            f"Unexpected ROI coordinate shape {rois.shape} in {stat_path}"
        )

    return rois


def _load_suite2p_mean_image(plane_dir: Path) -> Optional[np.ndarray]:
    """Return Suite2p mean image for a plane if available."""

    ops_path = plane_dir / "ops.npy"
    if ops_path.exists():
        try:
            ops = np.load(ops_path, allow_pickle=True).item()
            if isinstance(ops, dict):
                for key in ("meanImg", "meanImg_chan0", "meanImgE"):
                    mean_img = ops.get(key)
                    if mean_img is not None:
                        arr = np.asarray(mean_img, dtype=np.float32)
                        if arr.ndim == 2 and arr.size > 0:
                            return arr
        except Exception:
            logger.debug("Failed to load %s", ops_path, exc_info=True)
    mean_path = plane_dir / "meanImg.npy"
    if mean_path.exists():
        try:
            arr = np.load(mean_path)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2 and arr.size > 0:
                return arr
        except Exception:
            logger.debug("Failed to load %s", mean_path, exc_info=True)
    return None


def _apply_functional_to_anatomy_transform(
    roi_yx: np.ndarray,
    *,
    scale: float,
    translation_y: float,
    translation_x: float,
    z_index: int,
) -> np.ndarray:
    scaled = roi_yx * float(scale)
    shifted = scaled + np.array([translation_y, translation_x], dtype=np.float32)
    z_coords = np.full((scaled.shape[0], 1), float(z_index), dtype=np.float32)
    return np.hstack([z_coords, shifted.astype(np.float32, copy=False)])


def _resolve_stack_shape(path: Path) -> tuple[int, int, int]:
    """Return stack shape as ``(z, y, x)`` without fully loading the volume."""

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            shape = series.shape
            if len(shape) == 2:
                return (1, int(shape[0]), int(shape[1]))
            if len(shape) == 3:
                return (int(shape[0]), int(shape[1]), int(shape[2]))
            raise ValueError(f"Unsupported TIFF stack shape {shape} for {path}")
    if suffix == ".nrrd":
        # nrrd.read_header requires importing lazily to avoid optional dependency cost.
        import nrrd  # type: ignore

        header = nrrd.read_header(str(path))
        sizes = header.get("sizes")
        if sizes is None:
            raise ValueError(f"NRRD header missing 'sizes' for {path}")
        if len(sizes) != 3:
            raise ValueError(f"Expected 3D NRRD; got {sizes} for {path}")
        # NRRD stores as (X, Y, Z); convert to (Z, Y, X).
        return (int(sizes[2]), int(sizes[1]), int(sizes[0]))
    raise ValueError(f"Unsupported anatomy stack format: {path.suffix}")


def _apply_anatomy_to_reference_transform(
    roi_anat_zyx: np.ndarray,
    *,
    transform_sequence: Sequence[tuple[Path, bool]],
    anatomy_spacing_zyx: np.ndarray,
    anatomy_origin_zyx: np.ndarray,
    reference_spacing_zyx: np.ndarray,
    reference_origin_zyx: np.ndarray,
) -> np.ndarray:
    if roi_anat_zyx.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    anat_phys_zyx = align_substack.indexZYX_to_physZYX(
        roi_anat_zyx.astype(np.float64, copy=False),
        anatomy_spacing_zyx,
        anatomy_origin_zyx,
    )
    anat_phys_xyz = anat_phys_zyx[:, [2, 1, 0]]
    df = pd.DataFrame(anat_phys_xyz, columns=["x", "y", "z"])

    paths = [str(path) for path, _ in transform_sequence]
    invert = [bool(flag) for _, flag in transform_sequence]

    if not paths:
        raise FunctionalRoiRegistrationError("No transforms provided for ROI mapping")

    df_out = ants.apply_transforms_to_points(
        3,
        df,
        transformlist=paths,
        whichtoinvert=invert,
    )
    ref_phys_zyx = df_out[["z", "y", "x"]].to_numpy(dtype=np.float64, copy=False)
    ref_idx = align_substack.physZYX_to_indexZYX(
        ref_phys_zyx,
        reference_spacing_zyx,
        reference_origin_zyx,
    )
    return ref_idx.astype(np.float32, copy=False)


def _load_fireants_transforms(metadata_path: Path) -> dict[str, Path]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"FireANTs metadata not found: {metadata_path}")

    data = json.loads(metadata_path.read_text())
    outputs = data.get("outputs") or {}

    result: dict[str, Path] = {}
    for key in ("affine_transform", "greedy_transform", "greedy_inverse_transform"):
        value = outputs.get(key)
        if value:
            path = Path(value)
            result[key] = path if path.is_absolute() else metadata_path.parent / path
    return result


def build_transform_sequence(
    metadata_path: Path,
    *,
    use_inverse_warp: bool = False,
    invert_forward_warp: bool = False,
    invert_inverse_warp: bool = False,
    invert_affine: bool = True,
) -> list[tuple[Path, bool]]:
    """Return the ordered transform sequence to map anatomy → reference."""

    transforms = _load_fireants_transforms(metadata_path)
    affine_path = transforms.get("affine_transform")
    if affine_path is None:
        raise FunctionalRoiRegistrationError(
            f"Affine transform missing in {metadata_path}"
        )

    sequence: list[tuple[Path, bool]] = []
    if use_inverse_warp:
        inverse_path = transforms.get("greedy_inverse_transform")
        if inverse_path is None:
            raise FunctionalRoiRegistrationError(
                f"Inverse warp not found in {metadata_path}"
            )
        sequence.append((inverse_path, invert_inverse_warp))
    else:
        warp_path = transforms.get("greedy_transform")
        if warp_path is not None:
            sequence.append((warp_path, invert_forward_warp))
    sequence.append((affine_path, invert_affine))

    validated: list[tuple[Path, bool]] = []
    for path, invert in sequence:
        path = Path(path)
        if not path.exists():
            raise FunctionalRoiRegistrationError(f"Transform file not found: {path}")
        validated.append((path, invert))
    return validated


def transform_rois_to_reference(
    *,
    animal_id: str,
    session_id: str,
    registration_df: pd.DataFrame,
    suite2p_root: Path,
    plane_folder_template: str,
    anatomy_stack_path: Path,
    fireants_metadata_path: Path,
    reference_brain_path: Path,
    output_csv: Path,
    transform_sequence: Optional[Sequence[tuple[Path, bool]]] = None,
    plane_column: Optional[str] = "moving_plane",
    extra_stat_filenames: Optional[Sequence[str]] = None,
    flip_anatomy_x: bool = True,
    flip_anatomy_z: bool = False,
    overwrite: bool = False,
    reference_qc_settings: Optional[FunctionalRoiRegistrationQcConfig] = None,
    reference_qc_output_path: Optional[Path] = None,
    anatomy_qc_settings: Optional[FunctionalRoiRegistrationQcConfig] = None,
    anatomy_qc_output_path: Optional[Path] = None,
    native_qc_settings: Optional[FunctionalRoiRegistrationQcConfig] = None,
    native_qc_output_path: Optional[Path] = None,
    native_projection_path: Optional[Path] = None,
) -> FunctionalRoiRegistrationResult:
    """Transform Suite2p ROIs from functional space into reference coordinates."""

    output_csv = Path(output_csv)
    if output_csv.exists() and not overwrite:
        raise FunctionalRoiRegistrationError(
            f"Output already exists at {output_csv} (set overwrite=True to recompute)"
        )

    if registration_df.empty:
        raise FunctionalRoiRegistrationError("Registration summary dataframe is empty")

    transforms = transform_sequence
    if transforms is None:
        transforms = build_transform_sequence(fireants_metadata_path)
    else:
        validated_sequence: list[tuple[Path, bool]] = []
        for path, invert in transforms:
            candidate = Path(path)
            if not candidate.exists():
                raise FunctionalRoiRegistrationError(f"Transform file not found: {candidate}")
            validated_sequence.append((candidate, bool(invert)))
        transforms = validated_sequence

    def _load_anatomy_spacing() -> tuple[np.ndarray, np.ndarray]:
        meta_candidates = list(anatomy_stack_path.parent.glob("*_anatomy_metadata.json"))
        for meta_path in meta_candidates:
            try:
                data = json.loads(meta_path.read_text())
                px = data.get("pixel_size_xy_um")
                z = data.get("plane_spacing_um")
                if px and z is not None:
                    spacing = np.array([float(z), float(px[0]), float(px[1])], dtype=float)
                    origin = np.zeros(3, dtype=float)
                    return spacing, origin
            except Exception:
                continue
        return align_substack.get_spacing_origin_ZYX(anatomy_stack_path)

    anatomy_spacing, anatomy_origin = _load_anatomy_spacing()
    reference_spacing, reference_origin = align_substack.get_spacing_origin_ZYX(
        reference_brain_path
    )

    anatomy_shape = _resolve_stack_shape(anatomy_stack_path)
    width = anatomy_shape[2]
    depth = anatomy_shape[0]

    suite2p_root = Path(suite2p_root)
    if not suite2p_root.exists():
        raise FunctionalRoiRegistrationError(f"Suite2p output not found: {suite2p_root}")

    rows: list[dict[str, float | int | str]] = []
    skipped: dict[int, str] = {}
    processed_planes = 0
    total_rois = 0
    plane_order: list[int] = []
    plane_to_ref_values: dict[int, list[float]] = {}
    plane_details: dict[int, dict[str, Any]] = {}
    native_rois: dict[int, np.ndarray] = {}
    plane_to_anatomy_z: dict[int, int] = {}

    for idx, row in registration_df.iterrows():
        plane_idx = None
        if plane_column and plane_column in row:
            try:
                plane_idx = int(row[plane_column])
            except (TypeError, ValueError):
                plane_idx = None
        if plane_idx is None or plane_idx < 0:
            plane_idx = int(idx)

        plane_dirname = plane_folder_template.format(plane_index=plane_idx)
        plane_dir = suite2p_root / plane_dirname
        if not plane_dir.exists():
            skipped[plane_idx] = f"suite2p plane directory missing: {plane_dir}"
            continue

        success = bool(row.get("success", True))
        if not success:
            skipped[plane_idx] = "registration marked as failed"
            continue

        scale = float(row.get("scale_moving_to_fixed", 0.0))
        if not np.isfinite(scale) or scale <= 0.0:
            skipped[plane_idx] = f"invalid scale {scale}"
            continue

        z_value = float(row.get("z_index", np.nan))
        if not np.isfinite(z_value):
            skipped[plane_idx] = "missing z index"
            continue
        z_index = int(round(z_value))
        translation_y = float(row.get("y_px", np.nan))
        translation_x = float(row.get("x_px", np.nan))
        if not np.isfinite(translation_y) or not np.isfinite(translation_x):
            skipped[plane_idx] = "missing translation values"
            continue

        try:
            roi_yx = load_suite2p_rois(
                plane_dir,
                animal_id=animal_id,
                plane_index=plane_idx,
                extra_candidates=extra_stat_filenames,
            )
        except FileNotFoundError as exc:
            skipped[plane_idx] = str(exc)
            continue

        if roi_yx.size == 0:
            skipped[plane_idx] = "no ROIs detected"
            continue

        roi_anat = _apply_functional_to_anatomy_transform(
            roi_yx,
            scale=scale,
            translation_y=translation_y,
            translation_x=translation_x,
            z_index=z_index,
        )
        if flip_anatomy_x:
            roi_anat[:, 2] = (width - 1) - roi_anat[:, 2]
        if flip_anatomy_z:
            roi_anat[:, 0] = (depth - 1) - roi_anat[:, 0]
            z_index = depth - 1 - z_index

        roi_ref = _apply_anatomy_to_reference_transform(
            roi_anat,
            transform_sequence=transforms,
            anatomy_spacing_zyx=np.asarray(anatomy_spacing, dtype=np.float64),
            anatomy_origin_zyx=np.asarray(anatomy_origin, dtype=np.float64),
            reference_spacing_zyx=np.asarray(reference_spacing, dtype=np.float64),
            reference_origin_zyx=np.asarray(reference_origin, dtype=np.float64),
        )

        processed_planes += 1
        total_rois += roi_ref.shape[0]
        if plane_idx not in plane_to_ref_values:
            plane_order.append(plane_idx)
        plane_to_ref_values.setdefault(plane_idx, []).extend(
            roi_ref[:, 0].astype(float).tolist()
        )
        plane_to_anatomy_z[plane_idx] = z_index
        native_rois[plane_idx] = roi_yx
        mean_img = _load_suite2p_mean_image(plane_dir)
        if mean_img is not None:
            plane_details[plane_idx] = {
                "mean_image": mean_img,
                "scale": scale,
                "translation": (translation_y, translation_x),
                "z_index": z_index,
            }

        for roi_id, (anat_xyz, ref_xyz) in enumerate(zip(roi_anat, roi_ref)):
            z_anat, y_anat, x_anat = anat_xyz
            z_ref, y_ref, x_ref = ref_xyz
            rows.append(
                {
                    "animal": animal_id,
                    "session": session_id,
                    "plane": plane_idx,
                    "roi_id": roi_id,
                    "z_anat": float(z_anat),
                    "y_anat": float(y_anat),
                    "x_anat": float(x_anat),
                    "z_ref": float(z_ref),
                    "y_ref": float(y_ref),
                    "x_ref": float(x_ref),
                }
            )

    if not rows:
        raise FunctionalRoiRegistrationError(
            f"No ROIs transformed for {animal_id}::{session_id}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)

    qc_path: Optional[Path] = None
    native_qc_path: Optional[Path] = None

    if native_qc_settings and native_qc_settings.enabled:
        formatted_name = native_qc_settings.filename_template.format(
            animal_id=animal_id,
            session_id=session_id,
        )
        native_target = Path(native_qc_output_path) if native_qc_output_path is not None else None
        if native_target is None:
            qc_dir = output_csv.parent / native_qc_settings.output_subdir
            qc_dir.mkdir(parents=True, exist_ok=True)
            native_target = qc_dir / formatted_name
        else:
            native_target.parent.mkdir(parents=True, exist_ok=True)

        projection_path = native_projection_path if native_projection_path and Path(native_projection_path).exists() else None
        if projection_path is None:
            logger.warning("Native QC skipped: projection stack missing")
        else:
            try:
                native_qc_path = _generate_native_qc_overlay(
                    projections_path=projection_path,
                    native_rois=native_rois,
                    percentiles=tuple(native_qc_settings.percentiles),
                    point_size=native_qc_settings.point_size,
                    alpha=native_qc_settings.alpha,
                    output_path=native_target,
                    figsize=tuple(native_qc_settings.figsize),
                    dpi=native_qc_settings.dpi,
                    max_columns=native_qc_settings.max_columns,
                )
            except Exception:
                logger.exception("Failed to generate native ROI QC", exc_info=True)
                native_qc_path = None

    if anatomy_qc_settings and anatomy_qc_settings.enabled:
        formatted_name = anatomy_qc_settings.filename_template.format(
            animal_id=animal_id,
            session_id=session_id,
        )
        target_path = Path(anatomy_qc_output_path) if anatomy_qc_output_path is not None else None
        if target_path is None:
            qc_dir = output_csv.parent / anatomy_qc_settings.output_subdir
            qc_dir.mkdir(parents=True, exist_ok=True)
            target_path = qc_dir / formatted_name
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        plane_order_unique = list(dict.fromkeys(plane_order))
        anatomy_qc_path = _generate_anatomy_qc_overlay(
            df_out=df_out,
            anatomy_stack_path=anatomy_stack_path,
            config=anatomy_qc_settings,
            output_path=target_path,
            plane_order=plane_order_unique,
            plane_to_anatomy_z=plane_to_anatomy_z,
            plane_details=plane_details,
            flip_anatomy_x=flip_anatomy_x,
            flip_anatomy_z=flip_anatomy_z,
            anatomy_shape=anatomy_shape,
        )

    if reference_qc_settings and reference_qc_settings.enabled:
        formatted_name = reference_qc_settings.filename_template.format(
            animal_id=animal_id,
            session_id=session_id,
        )
        target_path = Path(reference_qc_output_path) if reference_qc_output_path is not None else None
        if target_path is None:
            qc_dir = output_csv.parent / reference_qc_settings.output_subdir
            qc_dir.mkdir(parents=True, exist_ok=True)
            target_path = qc_dir / formatted_name
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)

        plane_order_unique = list(dict.fromkeys(plane_order))

        reference_qc_path = _generate_reference_qc_overlay(
            df_out=df_out,
            reference_path=reference_brain_path,
            qc_settings=reference_qc_settings,
            output_path=target_path,
            plane_order=plane_order_unique,
            plane_to_ref_values=plane_to_ref_values,
            plane_details=plane_details,
            transform_sequence=transforms,
            anatomy_shape=anatomy_shape,
            anatomy_spacing=anatomy_spacing,
            anatomy_origin=anatomy_origin,
            flip_anatomy_x=flip_anatomy_x,
            flip_anatomy_z=flip_anatomy_z,
        )
        if reference_qc_path is not None:
            logger.info(
                "functional_roi_registration_qc_reference",
                extra={
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "qc_path": str(reference_qc_path),
                },
            )

    logger.info(
        "functional_roi_registration_completed",
        extra={
            "animal_id": animal_id,
            "session_id": session_id,
            "planes_total": int(len(registration_df)),
            "planes_processed": processed_planes,
            "rois_total": total_rois,
            "output_csv": str(output_csv),
        },
    )

    return FunctionalRoiRegistrationResult(
        animal_id=animal_id,
        session_id=session_id,
        output_csv=output_csv,
        total_planes=len(registration_df),
        processed_planes=processed_planes,
        total_rois=total_rois,
        skipped_planes=skipped,
        qc_path=reference_qc_path,
        native_qc_path=native_qc_path,
        anatomy_qc_path=anatomy_qc_path,
    )


def _generate_reference_qc_overlay(
    *,
    df_out: pd.DataFrame,
    reference_path: Path,
    qc_settings: FunctionalRoiRegistrationQcConfig,
    output_path: Path,
    plane_order: Sequence[int],
    plane_to_ref_values: Mapping[int, Sequence[float]],
    plane_details: Mapping[int, dict[str, Any]],
    transform_sequence: Sequence[tuple[Path, bool]],
    anatomy_shape: tuple[int, int, int],
    anatomy_spacing: np.ndarray,
    anatomy_origin: np.ndarray,
    flip_anatomy_x: bool,
    flip_anatomy_z: bool,
) -> Optional[Path]:
    if df_out.empty:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
    except Exception:
        pass

    import matplotlib.pyplot as plt

    try:
        reference_stack = align_substack.read_stack_float32(reference_path)
    except Exception:  # pragma: no cover - IO failure
        logger.exception("Failed to load reference stack for ROI QC", exc_info=True)
        return None

    if reference_stack.ndim != 3 or reference_stack.size == 0:
        return None

    ref_depth = reference_stack.shape[0]

    plane_to_ref_plane: dict[int, int] = {}
    for plane_idx, values in plane_to_ref_values.items():
        if not values:
            continue
        median = float(np.nanmedian(values))
        if not np.isfinite(median):
            continue
        target = int(round(median))
        target = int(np.clip(target, 0, ref_depth - 1))
        plane_to_ref_plane[int(plane_idx)] = target

    plane_overlays_ref: dict[int, np.ndarray] = {}
    ref_image_ants = None
    if plane_details and transform_sequence:
        try:
            ref_image_ants = ants.from_numpy(reference_stack.astype(np.float32, copy=False))
            ref_spacing, ref_origin = align_substack.get_spacing_origin_ZYX(reference_path)
            ref_image_ants.set_spacing(tuple(float(s) for s in ref_spacing[::-1]))
            ref_image_ants.set_origin(tuple(float(o) for o in ref_origin[::-1]))
        except Exception:
            logger.exception("Failed to prepare reference ANTs image for QC overlay", exc_info=True)
            ref_image_ants = None
    if ref_image_ants is not None:
        for plane_idx, details in plane_details.items():
            ref_plane = plane_to_ref_plane.get(int(plane_idx))
            if ref_plane is None:
                continue
            mean_image = details.get("mean_image")
            if mean_image is None:
                continue
            try:
                overlay = _warp_functional_plane_to_reference(
                    mean_image=mean_image,
                    scale=float(details.get("scale", 1.0)),
                    translation_y=float(details.get("translation", (0.0, 0.0))[0]),
                    translation_x=float(details.get("translation", (0.0, 0.0))[1]),
                    z_index=float(details.get("z_index", 0.0)),
                    anatomy_shape=anatomy_shape,
                    anatomy_spacing=anatomy_spacing,
                    anatomy_origin=anatomy_origin,
                    flip_x=flip_anatomy_x,
                    flip_z=flip_anatomy_z,
                    transform_sequence=transform_sequence,
                    ref_image=ref_image_ants,
                    ref_plane_index=ref_plane,
                )
                if overlay is not None:
                    plane_overlays_ref[int(plane_idx)] = overlay
            except Exception:
                logger.exception(
                    "Failed to warp functional plane %s for QC",
                    plane_idx,
                    exc_info=True,
                )

    planes_to_display = [
        int(plane) for plane in plane_order if plane in plane_to_ref_plane
    ]
    if not planes_to_display:
        return None

    max_planes = max(1, int(qc_settings.max_planes)) if qc_settings.max_planes else len(planes_to_display)
    planes_to_display = planes_to_display[:max_planes]

    ncols = max(1, min(qc_settings.max_columns, len(planes_to_display)))
    nrows = math.ceil(len(planes_to_display) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(qc_settings.figsize))
    axes = np.atleast_1d(axes).flatten()

    for ax in axes[len(planes_to_display) :]:
        ax.axis("off")

    cmap = plt.get_cmap("tab20")
    percentiles = tuple(qc_settings.percentiles)
    z_tol = float(qc_settings.z_tolerance)

    flip_z = bool(qc_settings.flip_z)
    for ax, plane in zip(axes, planes_to_display):
        z_plane = plane_to_ref_plane[plane]
        display_z = (ref_depth - 1 - z_plane) if flip_z else z_plane
        slice_img = reference_stack[display_z].astype(np.float32, copy=False)
        low, high = np.percentile(slice_img, percentiles)
        if high <= low:
            high = low + 1e-3
        norm = np.clip((slice_img - low) / (high - low), 0.0, 1.0)
        ax.imshow(norm, cmap="gray", interpolation="nearest")

        plane_rows = df_out[df_out["plane"] == plane]
        z_ref_values = plane_rows["z_ref"]
        if flip_z:
            z_ref_values = (ref_depth - 1) - z_ref_values
        subset = plane_rows[np.abs(z_ref_values - float(display_z)) <= z_tol]
        if subset.empty and not plane_rows.empty:
            subset = plane_rows
        overlay_img = plane_overlays_ref.get(plane)
        if overlay_img is not None and overlay_img.size > 0:
            overlay_img = np.asarray(overlay_img, dtype=np.float32)
            if overlay_img.shape != norm.shape:
                min_h = min(norm.shape[0], overlay_img.shape[0])
                min_w = min(norm.shape[1], overlay_img.shape[1])
                overlay_img = overlay_img[:min_h, :min_w]
                background = norm[:min_h, :min_w]
            else:
                background = norm
            o_low, o_high = np.percentile(overlay_img, (1, 99))
            if o_high <= o_low:
                o_high = o_low + 1e-3
            overlay_norm = np.clip((overlay_img - o_low) / (o_high - o_low), 0.0, 1.0)
            overlay_alpha = max(0.2, min(0.7, qc_settings.alpha))
            ax.imshow(
                overlay_norm,
                cmap="magma",
                alpha=overlay_alpha,
                interpolation="bilinear",
            )
        else:
            background = norm

        if not subset.empty:
            if hasattr(cmap, "colors"):
                color = cmap.colors[plane % len(cmap.colors)]
            else:
                denom = getattr(cmap, "N", 256) or 256
                color = cmap((plane % denom) / float(denom))
            ax.scatter(
                subset["x_ref"],
                subset["y_ref"],
                color=color,
                s=qc_settings.point_size,
                alpha=qc_settings.alpha,
                edgecolors="none",
            )

        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
        ax.set_title(f"plane {plane} → ref z {display_z} (n={int(len(subset))})")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=qc_settings.dpi)
    plt.close(fig)

    return output_path


def _generate_anatomy_qc_overlay(
    *,
    df_out: pd.DataFrame,
    anatomy_stack_path: Path,
    config: FunctionalRoiRegistrationQcConfig,
    output_path: Path,
    plane_order: Sequence[int],
    plane_to_anatomy_z: Mapping[int, int],
    plane_details: Mapping[int, dict[str, Any]],
    flip_anatomy_x: bool,
    flip_anatomy_z: bool,
    anatomy_shape: tuple[int, int, int],
) -> Optional[Path]:
    if df_out.empty:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    try:
        anatomy_stack = align_substack.read_stack_float32(anatomy_stack_path)
    except Exception:
        logger.exception("Failed to load anatomy stack for ROI QC", exc_info=True)
        return None
    if anatomy_stack.ndim != 3 or anatomy_stack.size == 0:
        return None

    planes = [p for p in plane_order if p in plane_to_anatomy_z]
    if not planes:
        return None
    max_planes = max(1, int(config.max_planes)) if config.max_planes else len(planes)
    planes = planes[:max_planes]

    ncols = max(1, min(config.max_columns, len(planes)))
    nrows = math.ceil(len(planes) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(config.figsize))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[len(planes) :]:
        ax.axis("off")

    percentiles = tuple(config.percentiles)
    for ax, plane_idx in zip(axes, planes):
        z_plane = plane_to_anatomy_z[plane_idx]
        z_plane = int(np.clip(z_plane, 0, anatomy_stack.shape[0] - 1))
        slice_img = anatomy_stack[z_plane].astype(np.float32, copy=False)
        low, high = np.percentile(slice_img, percentiles)
        if high <= low:
            high = low + 1e-3
        norm = np.clip((slice_img - low) / (high - low), 0.0, 1.0)
        ax.imshow(norm, cmap="gray", interpolation="nearest")

        overlay = _warp_functional_plane_to_anatomy(
            details=plane_details.get(plane_idx),
            anatomy_shape=anatomy_shape,
            flip_x=flip_anatomy_x,
            flip_z=flip_anatomy_z,
        )
        if overlay is not None:
            o_low, o_high = np.percentile(overlay, (1, 99))
            if o_high <= o_low:
                o_high = o_low + 1e-3
            overlay_norm = np.clip((overlay - o_low) / (o_high - o_low), 0.0, 1.0)
            ax.imshow(overlay_norm, cmap="magma", alpha=max(0.2, min(0.7, config.alpha)), interpolation="bilinear")

        plane_rows = df_out[df_out["plane"] == plane_idx]
        subset = plane_rows[np.abs(plane_rows["z_anat"] - float(z_plane)) <= float(config.z_tolerance or 0.0)]
        if subset.empty and not plane_rows.empty:
            subset = plane_rows
        if not subset.empty:
            ax.scatter(
                subset["x_anat"],
                subset["y_anat"],
                s=config.point_size,
                alpha=config.alpha,
                color="lime",
                edgecolors="none",
            )
        ax.set_xlim(0, norm.shape[1])
        ax.set_ylim(norm.shape[0], 0)
        ax.set_title(f"plane {plane_idx} → anat z {z_plane} (n={int(len(subset))})")
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=config.dpi)
    plt.close(fig)
    return output_path


def _warp_functional_plane_to_reference(
    *,
    mean_image: np.ndarray,
    scale: float,
    translation_y: float,
    translation_x: float,
    z_index: float,
    anatomy_shape: tuple[int, int, int],
    anatomy_spacing: np.ndarray,
    anatomy_origin: np.ndarray,
    flip_x: bool,
    flip_z: bool,
    transform_sequence: Sequence[tuple[Path, bool]],
    ref_image,
    ref_plane_index: int,
) -> Optional[np.ndarray]:
    if mean_image is None or mean_image.size == 0:
        return None
    if scale <= 0.0:
        return None

    try:
        forward = AffineTransform(
            scale=(float(scale), float(scale)),
            translation=(float(translation_x), float(translation_y)),
        )
        warped_plane = warp(
            mean_image.astype(np.float32, copy=False),
            inverse_map=forward.inverse,
            output_shape=(anatomy_shape[1], anatomy_shape[2]),
            order=1,
            preserve_range=True,
        ).astype(np.float32, copy=False)
    except Exception:
        logger.exception("Failed to warp functional plane via affine", exc_info=True)
        return None

    if flip_x:
        warped_plane = warped_plane[:, ::-1]

    depth = anatomy_shape[0]
    z_idx = int(round(z_index))
    z_idx = int(np.clip(z_idx, 0, depth - 1))
    if flip_z:
        z_idx = (depth - 1) - z_idx

    volume = np.zeros(anatomy_shape, dtype=np.float32)
    volume[z_idx] = warped_plane

    ants_image = ants.from_numpy(volume)
    ants_image.set_spacing(tuple(float(s) for s in anatomy_spacing[::-1]))
    ants_image.set_origin(tuple(float(o) for o in anatomy_origin[::-1]))

    transformlist = [str(path) for path, _ in transform_sequence]
    whichtoinvert = [bool(flag) for _, flag in transform_sequence]

    try:
        warped = ants.apply_transforms(
            fixed=ref_image,
            moving=ants_image,
            transformlist=transformlist,
            whichtoinvert=whichtoinvert,
        )
    except Exception:
        logger.exception("Failed to apply transforms to functional plane", exc_info=True)
        return None

    warped_np = warped.numpy()
    if warped_np.ndim != 3:
        return None
    ref_plane_index = int(np.clip(ref_plane_index, 0, warped_np.shape[0] - 1))
    return warped_np[ref_plane_index]


def _warp_functional_plane_to_anatomy(
    *,
    details: Optional[dict[str, Any]],
    anatomy_shape: tuple[int, int, int],
    flip_x: bool,
    flip_z: bool,
) -> Optional[np.ndarray]:
    if not details:
        return None
    mean_image = details.get("mean_image")
    if mean_image is None:
        return None
    scale = float(details.get("scale", 1.0))
    translation = details.get("translation", (0.0, 0.0))
    z_index = float(details.get("z_index", 0.0))
    if scale <= 0.0:
        return None

    try:
        forward = AffineTransform(
            scale=(scale, scale),
            translation=(float(translation[1]), float(translation[0])),
        )
        warped_plane = warp(
            mean_image.astype(np.float32, copy=False),
            inverse_map=forward.inverse,
            output_shape=(anatomy_shape[1], anatomy_shape[2]),
            order=1,
            preserve_range=True,
        ).astype(np.float32, copy=False)
    except Exception:
        logger.exception("Failed to warp functional plane to anatomy", exc_info=True)
        return None

    if flip_x:
        warped_plane = warped_plane[:, ::-1]
    z_idx = int(round(z_index))
    z_idx = int(np.clip(z_idx, 0, anatomy_shape[0] - 1))
    if flip_z:
        z_idx = (anatomy_shape[0] - 1) - z_idx
    return warped_plane if warped_plane.size > 0 else None

def _generate_native_qc_overlay(
    *,
    projections_path: Path,
    native_rois: Mapping[int, np.ndarray],
    percentiles: tuple[float, float],
    point_size: float,
    alpha: float,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    max_columns: int,
) -> Optional[Path]:
    projections_path = Path(projections_path)
    if not projections_path.exists():
        return None

    try:
        stack = tifffile.imread(str(projections_path))
    except Exception:
        logger.exception("Failed to load projection stack for native QC", exc_info=True)
        return None

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3 or stack.size == 0:
        return None

    plane_indices = sorted(native_rois.keys())
    if not plane_indices:
        return None

    ncols = max(1, min(max_columns or 1, len(plane_indices)))
    nrows = math.ceil(len(plane_indices) / ncols)

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for ax in axes[len(plane_indices) :]:
        ax.axis("off")

    for ax, plane_idx in zip(axes, plane_indices):
        if plane_idx >= stack.shape[0]:
            ax.axis("off")
            continue
        img = stack[plane_idx].astype(np.float32, copy=False)
        lo, hi = np.percentile(img, percentiles)
        if hi <= lo:
            hi = lo + 1e-3
        norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        ax.imshow(norm, cmap="gray", interpolation="nearest")

        rois = native_rois.get(plane_idx)
        if rois is not None and rois.size > 0:
            ax.scatter(
                rois[:, 1],
                rois[:, 0],
                s=point_size,
                alpha=alpha,
                c="lime",
                edgecolors="none",
            )
        ax.set_xlim(0, norm.shape[1])
        ax.set_ylim(norm.shape[0], 0)
        ax.set_title(f"plane {plane_idx} (n={int(len(rois)) if rois is not None else 0})")
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path
