"""FireANTs registration of confocal stacks to two-photon anatomy."""

from __future__ import annotations

import copy
import logging
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import tifffile

from .fireants_pipeline import (
    FireANTsRegistrationConfig,
    _import_fireants,
    _winsorize_image,
    _write_qc_figure,
)


logger = logging.getLogger(__name__)


def _to_sitk_image(array: np.ndarray, spacing: Tuple[float, float, float]) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    image.SetSpacing(tuple(spacing))
    return image


def _margin_to_voxels(value: float, size: int) -> int:
    """Convert fractional or absolute margin specifications into voxels."""

    if value <= 0:
        return 0
    if value < 1.0:
        return max(0, int(round(value * size)))
    return max(0, int(round(value)))


def _build_central_mask(
    shape: tuple[int, int, int],
    margin_xy: float,
    margin_z: float,
) -> np.ndarray:
    """Binary mask that keeps the central region of a volume."""

    z, y, x = shape
    margin_z_vox = min(_margin_to_voxels(margin_z, z), z // 2)
    margin_y_vox = min(_margin_to_voxels(margin_xy, y), y // 2)
    margin_x_vox = min(_margin_to_voxels(margin_xy, x), x // 2)

    if margin_z_vox <= 0 and margin_y_vox <= 0 and margin_x_vox <= 0:
        return np.ones(shape, dtype=np.float32)

    mask = np.zeros(shape, dtype=np.float32)
    z0 = margin_z_vox
    z1 = max(z - margin_z_vox, z0 + 1)
    y0 = margin_y_vox
    y1 = max(y - margin_y_vox, y0 + 1)
    x0 = margin_x_vox
    x1 = max(x - margin_x_vox, x0 + 1)
    mask[z0:z1, y0:y1, x0:x1] = 1.0
    return mask


def _compute_extent_crop_slices(
    moving_shape: tuple[int, int, int],
    moving_spacing: Tuple[float, float, float],
    fixed_shape: tuple[int, int, int],
    fixed_spacing: Tuple[float, float, float],
    padding_um: float,
) -> tuple[tuple[slice, slice, slice], dict[str, list[int]]]:
    """Determine symmetric crop slices so confocal XY extent matches anatomy."""

    def _axis_bounds(size: int, spacing: float, target_extent: float) -> tuple[int, int]:
        current_extent = size * spacing
        if current_extent <= target_extent or spacing <= 0:
            return 0, size
        trim_um = max(0.0, (current_extent - target_extent) / 2.0)
        trim_vox = min(size // 2, int(round(trim_um / spacing)))
        return trim_vox, size - trim_vox

    target_x_extent = fixed_shape[2] * fixed_spacing[0] + 2.0 * max(0.0, padding_um)
    target_y_extent = fixed_shape[1] * fixed_spacing[1] + 2.0 * max(0.0, padding_um)

    x0, x1 = _axis_bounds(moving_shape[2], moving_spacing[0], target_x_extent)
    y0, y1 = _axis_bounds(moving_shape[1], moving_spacing[1], target_y_extent)

    crop_info = {"y_vox": [int(y0), int(y1)], "x_vox": [int(x0), int(x1)]}
    return (slice(None), slice(y0, y1), slice(x0, x1)), crop_info


def register_confocal_to_anatomy(
    *,
    animal_id: str,
    confocal_session_id: str,
    anatomy_session_id: str,
    moving_channel_path: Path,
    fixed_stack_path: Path,
    additional_channels: Dict[str, Path],
    output_root: Path,
    config: FireANTsRegistrationConfig,
    voxel_spacing_um: Tuple[float, float, float],
    fixed_spacing_um: Tuple[float, float, float],
    warped_channel_template: str,
    metadata_filename: str,
    transforms_subdir: Path,
    qc_subdir: Path,
    reference_channel_name: str,
    mask_margin_xy: float,
    mask_margin_z: float,
    histogram_match: bool,
    histogram_levels: int,
    histogram_match_points: int,
    histogram_threshold_at_mean: bool,
    crop_to_extent: bool,
    crop_padding_um: float,
) -> Dict[str, object]:
    """Register a confocal channel to two-photon anatomy and warp additional channels."""

    cfg = copy.deepcopy(config)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"

    (
        fireants_version,
        FAImage,
        BatchedImages,
        MomentsRegistration,
        AffineRegistration,
        GreedyRegistration,
    ) = _import_fireants()

    moving_channel_path = Path(moving_channel_path)
    fixed_stack_path = Path(fixed_stack_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not moving_channel_path.exists():
        raise FileNotFoundError(moving_channel_path)
    if not fixed_stack_path.exists():
       raise FileNotFoundError(fixed_stack_path)

    moving_array = tifffile.imread(moving_channel_path).astype(np.float32, copy=False)
    fixed_array = tifffile.imread(fixed_stack_path).astype(np.float32, copy=False)

    crop_slices = (slice(None), slice(None), slice(None))
    crop_info: dict[str, list[int]] | None = None
    if crop_to_extent:
        crop_slices, crop_info = _compute_extent_crop_slices(
            moving_array.shape,
            voxel_spacing_um,
            fixed_array.shape,
            fixed_spacing_um,
            crop_padding_um,
        )
        logger.info(
            "Confocal cropping enabled; y:%s x:%s padding %.2f µm",
            crop_info["y_vox"] if crop_info else None,
            crop_info["x_vox"] if crop_info else None,
            crop_padding_um,
        )
        moving_array = moving_array[crop_slices]

    moving_mask = _build_central_mask(moving_array.shape, mask_margin_xy, mask_margin_z)
    fixed_mask = _build_central_mask(fixed_array.shape, mask_margin_xy, mask_margin_z)
    if np.any(moving_mask != 1.0) or np.any(fixed_mask != 1.0):
        logger.info(
            "Applying central masks (mask_margin_xy=%.3f, mask_margin_z=%.3f)",
            mask_margin_xy,
            mask_margin_z,
        )
    moving_array *= moving_mask
    fixed_array *= fixed_mask

    spacing = tuple(float(s) for s in voxel_spacing_um)
    moving_image = _to_sitk_image(moving_array, spacing)
    fixed_image = _to_sitk_image(fixed_array, fixed_spacing_um)

    if histogram_match:
        logger.info(
            "Histogram matching enabled (levels=%d, match_points=%d, threshold_at_mean=%s)",
            histogram_levels,
            histogram_match_points,
            histogram_threshold_at_mean,
        )
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(int(histogram_levels))
        matcher.SetNumberOfMatchPoints(int(histogram_match_points))
        if histogram_threshold_at_mean:
            matcher.ThresholdAtMeanIntensityOn()
        else:
            matcher.ThresholdAtMeanIntensityOff()
        moving_image = matcher.Execute(moving_image, fixed_image)

    moving_image, moving_winsor = _winsorize_image(moving_image, cfg.winsorize)
    fixed_image, fixed_winsor = _winsorize_image(fixed_image, cfg.winsorize)

    moving_fa = FAImage(moving_image, device=cfg.device, dtype=torch.float32)
    fixed_fa = FAImage(fixed_image, device=cfg.device, dtype=torch.float32)
    moving_batch = BatchedImages(moving_fa)
    fixed_batch = BatchedImages(fixed_fa)

    moments = MomentsRegistration(
        scale=cfg.moments_scale,
        fixed_images=fixed_batch,
        moving_images=moving_batch,
    )
    moments.optimize()
    init_affine = moments.get_affine_init().detach()

    affine = AffineRegistration(
        list(cfg.affine.scales),
        list(cfg.affine.iterations),
        fixed_batch,
        moving_batch,
        optimizer=cfg.affine.optimizer,
        optimizer_lr=cfg.affine.optimizer_lr,
        cc_kernel_size=cfg.affine.cc_kernel_size,
        tolerance=cfg.affine.tolerance,
        max_tolerance_iters=cfg.affine.max_tolerance_iters,
        loss_type=cfg.affine.loss_type,
        init_rigid=init_affine,
        **cfg.affine.extra_args,
    )
    affine.optimize()
    final_tensor = affine.evaluate(fixed_batch, moving_batch)

    greedy = None
    if cfg.greedy.enabled:
        greedy = GreedyRegistration(
            list(cfg.greedy.scales),
            list(cfg.greedy.iterations),
            fixed_batch,
            moving_batch,
            optimizer=cfg.greedy.optimizer,
            optimizer_lr=cfg.greedy.optimizer_lr,
            cc_kernel_size=cfg.greedy.cc_kernel_size,
            loss_type=cfg.greedy.loss_type,
            smooth_grad_sigma=cfg.greedy.smooth_grad_sigma,
            smooth_warp_sigma=cfg.greedy.smooth_warp_sigma,
            deformation_type=cfg.greedy.deformation_type,
            optimizer_params=cfg.greedy.optimizer_params,
            loss_params=cfg.greedy.loss_params,
            init_affine=affine.get_affine_matrix().detach(),
            **cfg.greedy.extra_args,
        )
        greedy.optimize()
        final_tensor = greedy.evaluate(fixed_batch, moving_batch)

    transforms_dir = output_root / transforms_subdir
    transforms_dir.mkdir(exist_ok=True)

    affine_transform_path = transforms_dir / f"{animal_id}_{confocal_session_id}_affine.mat"
    affine.save_as_ants_transforms(str(affine_transform_path))

    greedy_transform_path = None
    greedy_inverse_path = None
    inverse_summary = None

    if greedy is not None:
        greedy_transform_path = transforms_dir / f"{animal_id}_{confocal_session_id}_greedy_warp.nii.gz"
        greedy_inverse_path = transforms_dir / f"{animal_id}_{confocal_session_id}_greedy_inverse_warp.nii.gz"
        greedy.save_as_ants_transforms(str(greedy_transform_path))
        greedy.save_as_ants_transforms(str(greedy_inverse_path), save_inverse=True)
        try:
            inverse_tensor = greedy.evaluate_inverse(fixed_batch, moving_batch)
            diff = inverse_tensor - moving_batch()
            inverse_summary = {
                "mse": float(torch.mean(diff**2).item()),
                "max_abs": float(torch.max(torch.abs(diff)).item()),
            }
        except Exception:  # pragma: no cover - diagnostic only
            inverse_summary = None

    warped_gcamp = final_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

    warped_channels: Dict[str, Path] = {}
    gcamp_output = output_root / warped_channel_template.format(
        animal_id=animal_id,
        confocal_session_id=confocal_session_id,
        anatomy_session_id=anatomy_session_id,
        channel=reference_channel_name,
    )
    tifffile.imwrite(gcamp_output, warped_gcamp)
    warped_channels[reference_channel_name] = gcamp_output

    def _warp_additional_channel(name: str, channel_path: Path) -> Path:
        arr = tifffile.imread(channel_path).astype(np.float32, copy=False)
        arr = arr[crop_slices]
        if arr.shape != moving_mask.shape:
            raise ValueError(
                f"Additional channel {name} shape {arr.shape} does not match reference {moving_mask.shape}"
            )
        arr = arr * moving_mask
        img = _to_sitk_image(arr, spacing)
        img, _ = _winsorize_image(img, cfg.winsorize)
        channel_fa = FAImage(img, device=cfg.device, dtype=torch.float32)
        channel_batch = BatchedImages(channel_fa)
        if greedy is not None:
            warped = greedy.evaluate(fixed_batch, channel_batch)
        else:
            warped = affine.evaluate(fixed_batch, channel_batch)
        warped_arr = warped.squeeze().detach().cpu().numpy().astype(np.float32)
        output_path = output_root / warped_channel_template.format(
            animal_id=animal_id,
            confocal_session_id=confocal_session_id,
            anatomy_session_id=anatomy_session_id,
            channel=name,
        )
        tifffile.imwrite(output_path, warped_arr)
        return output_path

    for name, path in additional_channels.items():
        warped_channels[name] = _warp_additional_channel(name, path)

    qc_dir = output_root / qc_subdir
    qc_dir.mkdir(exist_ok=True)
    qc_path = qc_dir / f"{animal_id}_{confocal_session_id}_qc.png"
    _write_qc_figure(
        reference=fixed_fa.array.squeeze().detach().cpu().numpy(),
        moving=moving_fa.array.squeeze().detach().cpu().numpy(),
        warped=warped_gcamp,
        output_path=qc_path,
        percentiles=(cfg.winsorize.lower_percentile, cfg.winsorize.upper_percentile),
        middle_plane=cfg.qc_middle_plane,
        figsize=cfg.qc_figsize,
    )

    metadata = {
        "animal_id": animal_id,
        "confocal_session_id": confocal_session_id,
        "anatomy_session_id": anatomy_session_id,
        "fireants_version": fireants_version,
        "config": asdict(cfg),
        "voxel_spacing_um": list(voxel_spacing_um),
        "fixed_spacing_um": list(fixed_spacing_um),
        "winsorize": {
            "moving": moving_winsor,
            "fixed": fixed_winsor,
        },
        "mask": {
            "margin_xy": float(mask_margin_xy),
            "margin_z": float(mask_margin_z),
        },
        "histogram_match": {
            "enabled": bool(histogram_match),
            "levels": int(histogram_levels),
            "match_points": int(histogram_match_points),
            "threshold_at_mean": bool(histogram_threshold_at_mean),
        },
        "cropping": {
            "enabled": bool(crop_to_extent),
            "y_vox": (crop_info["y_vox"] if crop_info else None),
            "x_vox": (crop_info["x_vox"] if crop_info else None),
            "padding_um": float(crop_padding_um),
        },
        "outputs": {
            "gcamp": str(gcamp_output),
            "warped_channels": {name: str(path) for name, path in warped_channels.items()},
            "affine_transform": str(affine_transform_path),
            "greedy_transform": str(greedy_transform_path) if greedy_transform_path else None,
            "greedy_inverse_transform": str(greedy_inverse_path) if greedy_inverse_path else None,
            "qc": str(qc_path),
        },
    }
    if inverse_summary is not None:
        metadata["inverse_qc"] = inverse_summary

    metadata_path = output_root / metadata_filename.format(
        animal_id=animal_id,
        confocal_session_id=confocal_session_id,
        anatomy_session_id=anatomy_session_id,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "warped_channels": warped_channels,
        "gcamp_warp": gcamp_output,
        "affine_transform": affine_transform_path,
        "greedy_transform": greedy_transform_path,
        "greedy_inverse_transform": greedy_inverse_path,
        "metadata": metadata_path,
        "qc": qc_path,
        "inverse_qc": inverse_summary,
    }
