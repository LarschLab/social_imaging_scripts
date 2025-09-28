"""FireANTs-based registration of two-photon anatomy stacks."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import SimpleITK as sitk

try:
    from fireants.io.image import Image as FAImage, BatchedImages
    from fireants.registration.affine import AffineRegistration
except ImportError as exc:  # pragma: no cover - import error surfaced to caller
    raise ImportError(
        "FireANTs core modules are unavailable. Install `fireants` inside the antspy-win environment."
    ) from exc

try:  # SyN is optional because it requires fused ops
    from fireants.registration.syn import SyNRegistration  # type: ignore
    _SYN_AVAILABLE = True
    _SYN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - missing fused ops
    SyNRegistration = None  # type: ignore
    _SYN_AVAILABLE = False
    _SYN_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FireANTsConfig:
    """Configuration for FireANTs registration pipelines."""

    device: str = field(default_factory=_default_device)
    voxel_spacing_um: Optional[tuple[float, float, float]] = None
    intensity_percentiles: tuple[float, float] = (0.5, 99.5)

    affine_scales: tuple[int, ...] = (4, 2, 1)
    affine_iterations: tuple[int, ...] = (1000, 500, 250)
    affine_loss: str = "cc"
    affine_optimizer: str = "Adam"
    affine_optimizer_lr: float = 0.1
    affine_cc_kernel_size: int = 5
    affine_tolerance: float = 1e-6
    affine_max_tolerance_iters: int = 10

    run_syn: bool = True
    syn_scales: tuple[int, ...] = (4, 2, 1)
    syn_iterations: tuple[int, ...] = (200, 100, 50)
    syn_optimizer: str = "Adam"
    syn_optimizer_lr: float = 0.05
    syn_cc_kernel_size: int = 5
    syn_smooth_grad_sigma: float = 1.0
    syn_smooth_warp_sigma: float = 0.5

    qc_middle_plane: Optional[int] = None
    qc_figsize: tuple[int, int] = (10, 6)

    extra_args: Dict[str, Any] = field(default_factory=dict)


def _percentile_normalise(
    volume: np.ndarray,
    percentiles: tuple[float, float],
) -> np.ndarray:
    lo, hi = np.percentile(volume, list(percentiles))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (volume - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def _to_uint16(volume: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    scaled = _percentile_normalise(volume, percentiles)
    return (scaled * np.iinfo(np.uint16).max).astype(np.uint16, copy=False)


def _apply_spacing(image: sitk.Image, spacing_um: Optional[tuple[float, float, float]]) -> sitk.Image:
    if spacing_um is None:
        return image
    spacing_xyz = tuple(float(x) for x in spacing_um[::-1])  # SITK expects (x, y, z)
    image.SetSpacing(spacing_xyz)
    return image


def _load_fireants_image(path: Path, config: FireANTsConfig, *, allow_spacing_override: bool) -> FAImage:
    itk_image = sitk.ReadImage(str(path))
    if allow_spacing_override and config.voxel_spacing_um is not None:
        itk_image = _apply_spacing(itk_image, config.voxel_spacing_um)
    dtype = torch.float32
    image = FAImage(itk_image, device=config.device, dtype=dtype)
    return image


def _generate_qc(
    *,
    reference: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    output_path: Path,
    percentiles: tuple[float, float],
    middle_plane: Optional[int],
    figsize: tuple[int, int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    z = reference.shape[0]
    index = middle_plane if middle_plane is not None else z // 2
    index = int(np.clip(index, 0, z - 1))

    ref_slice = _percentile_normalise(reference[index], percentiles)
    mov_slice = _percentile_normalise(moving[index], percentiles)
    warped_slice = _percentile_normalise(warped[index], percentiles)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("FireANTs registration")

    axes[0, 0].imshow(ref_slice, cmap="gray")
    axes[0, 0].set_title("Reference")
    axes[0, 1].imshow(mov_slice, cmap="gray")
    axes[0, 1].set_title("Moving (pre)")

    axes[1, 0].imshow(ref_slice, cmap="gray")
    axes[1, 0].set_title("Reference")
    axes[1, 1].imshow(warped_slice, cmap="gray")
    axes[1, 1].set_title("Moving (post)")

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def register_two_photon_anatomy(
    *,
    animal_id: str,
    session_id: str,
    stack_path: Path,
    output_root: Path,
    reference_brain_path: Path,
    config: Optional[FireANTsConfig] = None,
) -> Dict[str, Path]:
    """Register a preprocessed two-photon stack to the reference brain."""

    cfg = config or FireANTsConfig()
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        cfg.device = "cpu"
    stack_path = Path(stack_path)
    reference_brain_path = Path(reference_brain_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not stack_path.exists():
        raise FileNotFoundError(f"Anatomy stack not found: {stack_path}")
    if not reference_brain_path.exists():
        raise FileNotFoundError(f"Reference brain not found: {reference_brain_path}")

    logger.info(
        "Starting FireANTs registration | animal=%s session=%s stack=%s reference=%s",
        animal_id,
        session_id,
        stack_path,
        reference_brain_path,
    )

    reference_img = _load_fireants_image(reference_brain_path, cfg, allow_spacing_override=False)
    moving_img = _load_fireants_image(stack_path, cfg, allow_spacing_override=True)

    fixed_batch = BatchedImages(reference_img)
    moving_batch = BatchedImages(moving_img)

    affine = AffineRegistration(
        list(cfg.affine_scales),
        list(cfg.affine_iterations),
        fixed_batch,
        moving_batch,
        loss_type=cfg.affine_loss,
        optimizer=cfg.affine_optimizer,
        optimizer_lr=cfg.affine_optimizer_lr,
        cc_kernel_size=cfg.affine_cc_kernel_size,
        tolerance=cfg.affine_tolerance,
        max_tolerance_iters=cfg.affine_max_tolerance_iters,
    )
    affine.optimize()
    affine_warped_tensor = affine.evaluate(fixed_batch, moving_batch)

    affine_transform_path = output_root / f"{animal_id}_fireants_affine.mat"
    affine.save_as_ants_transforms(str(affine_transform_path))

    final_tensor = affine_warped_tensor
    syn_transform_path: Optional[Path] = None
    syn_inverse_path: Optional[Path] = None

    if cfg.run_syn:
        if not _SYN_AVAILABLE:
            logger.warning("SyN registration skipped: %s", _SYN_IMPORT_ERROR)
        else:
            init_affine = affine.get_affine_matrix(homogenous=True)
            syn = SyNRegistration(
                list(cfg.syn_scales),
                list(cfg.syn_iterations),
                fixed_batch,
                moving_batch,
                optimizer=cfg.syn_optimizer,
                optimizer_lr=cfg.syn_optimizer_lr,
                cc_kernel_size=cfg.syn_cc_kernel_size,
                smooth_grad_sigma=cfg.syn_smooth_grad_sigma,
                smooth_warp_sigma=cfg.syn_smooth_warp_sigma,
                init_affine=init_affine,
            )
            syn.optimize()
            final_tensor = syn.evaluate(fixed_batch, moving_batch)

            syn_transform_path = output_root / f"{animal_id}_fireants_syn_warp.nii.gz"
            syn.save_as_ants_transforms(str(syn_transform_path))

            syn_inverse_path = output_root / f"{animal_id}_fireants_syn_inverse_warp.nii.gz"
            syn.save_as_ants_transforms(str(syn_inverse_path), save_inverse=True)

    final_volume = final_tensor.squeeze().detach().cpu().numpy()
    warped_path = output_root / f"{animal_id}_anatomy_warped_fireants.tif"
    tifffile.imwrite(warped_path, _to_uint16(final_volume, cfg.intensity_percentiles))

    qc_dir = output_root / "qc"
    qc_path = qc_dir / f"{animal_id}_fireants_qc.png"
    reference_np = reference_img.array.squeeze().detach().cpu().numpy()
    moving_np = moving_img.array.squeeze().detach().cpu().numpy()
    _generate_qc(
        reference=reference_np,
        moving=moving_np,
        warped=final_volume,
        output_path=qc_path,
        percentiles=cfg.intensity_percentiles,
        middle_plane=cfg.qc_middle_plane,
        figsize=cfg.qc_figsize,
    )

    provenance = {
        "animal_id": animal_id,
        "session_id": session_id,
        "input_stack": str(stack_path),
        "reference_brain": str(reference_brain_path),
        "device": cfg.device,
        "config": asdict(cfg),
        "syn_available": _SYN_AVAILABLE,
        "syn_import_error": str(_SYN_IMPORT_ERROR) if _SYN_IMPORT_ERROR else None,
    }
    metadata_path = output_root / "fireants_registration_metadata.json"
    metadata_path.write_text(json.dumps(provenance, indent=2))

    results: Dict[str, Path] = {
        "warped_stack": warped_path,
        "affine_transform": affine_transform_path,
        "qc_figure": qc_path,
        "provenance": metadata_path,
    }
    if syn_transform_path is not None:
        results["syn_transform"] = syn_transform_path
    if syn_inverse_path is not None:
        results["syn_inverse_transform"] = syn_inverse_path

    logger.info("FireANTs registration finished | outputs: %s", results)
    return results


__all__ = ["FireANTsConfig", "register_two_photon_anatomy"]
