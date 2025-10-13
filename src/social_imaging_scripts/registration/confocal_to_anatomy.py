"""FireANTs registration of confocal stacks to two-photon anatomy."""

from __future__ import annotations

import copy
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


def _to_sitk_image(array: np.ndarray, spacing: Tuple[float, float, float]) -> sitk.Image:
    image = sitk.GetImageFromArray(array.astype(np.float32, copy=False))
    image.SetSpacing(tuple(spacing))
    return image


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

    spacing = tuple(float(s) for s in voxel_spacing_um)
    moving_image = _to_sitk_image(moving_array, spacing)
    fixed_image = _to_sitk_image(fixed_array, fixed_spacing_um)

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
