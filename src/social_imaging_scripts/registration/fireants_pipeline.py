"""FireANTs-based registration pipeline utilities.

This module encapsulates the exact FireANTs workflow we validated in
``exampleNotebooks/RohitSettings.ipynb`` (Rohit Jena, 2025). The notebook
established a three-stage recipe—moment alignment, affine refinement, and
greedy diffeomorphic warping—along with specific hyper-parameters that provide
robust alignment for zebrafish two-photon anatomy stacks. The defaults below
mirror those values so that the automated pipeline reproduces the same
behaviour and runtime characteristics you would observe when running the
notebook manually.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tifffile
import torch
import pandas as pd

from social_imaging_scripts.metadata.config import resolve_raw_path
from social_imaging_scripts.metadata.models import AnimalMetadata, AnatomySession

logger = logging.getLogger(__name__)


@dataclass
class WinsorizeSettings:
    """Intensity clipping + normalization prior to registration.

    Rohit's workflow clips both moving and reference stacks to the 1st/99th
    percentile before rescaling into ``[0, 1]``. This stabilises the
    cross-correlation loss (especially `fusedcc`) and keeps the optimiser from
    chasing extremely bright nuclei. The ``enabled`` switch allows providers to
    skip the step when pre-normalised volumes are supplied.
    """

    enabled: bool = True
    lower_percentile: float = 1.0
    upper_percentile: float = 99.0


@dataclass
class AffineSettings:
    """Configuration for the FireANTs affine registration stage.

    Defaults reproduce the multi-resolution schedule from Rohit's notebook:
    coarse-to-fine scales ``[12, 8, 4, 2, 1]`` with per-level iteration counts
    ``[200, 200, 200, 100, 50]``. The `fusedcc` loss and kernel size ``31`` are
    essential for contrast-invariant matching across anatomy stacks, while the
    relatively small Adam learning rate ``3e-3`` prevents overshooting once the
    rigid moment initialisation is applied.
    """

    scales: tuple[int, ...] = (12, 8, 4, 2, 1)
    iterations: tuple[int, ...] = (200, 200, 200, 100, 50)
    optimizer: str = "Adam"
    optimizer_lr: float = 3e-3
    cc_kernel_size: int = 31
    tolerance: float = 1e-6
    max_tolerance_iters: int = 10
    loss_type: str = "fusedcc"
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GreedySettings:
    """Configuration for the FireANTs greedy (non-linear) registration stage.

    Again, these numbers originate from the RohitSettings notebook: the greedy
    solver runs through the same multiscale pyramid (``[12, 8, 4, 2, 1]``) but
    stops at 25 iterations on the finest level to limit runtime. The fused
    cross-correlation loss with kernel ``31`` proved more stable than MI for
    this dataset; the relatively aggressive Adam learning rate ``0.5`` is
    compensated by the heavy Gaussian smoothing (``smooth_grad_sigma=5``,
    ``smooth_warp_sigma=3``) which regularises each incremental warp update.
    """

    enabled: bool = True
    scales: tuple[int, ...] = (12, 8, 4, 2, 1)
    iterations: tuple[int, ...] = (200, 200, 200, 100, 25)
    optimizer: str = "Adam"
    optimizer_lr: float = 0.5
    cc_kernel_size: int = 31
    loss_type: str = "fusedcc"
    smooth_grad_sigma: float = 5.0
    smooth_warp_sigma: float = 3.0
    deformation_type: str = "compositive"
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    loss_params: Dict[str, Any] = field(default_factory=dict)
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FireANTsRegistrationConfig:
    """Top-level registration configuration with embedded defaults.

    ``moments_scale`` controls the isotropic down-sampling applied inside
    ``MomentsRegistration``. Setting this to ``4`` matches Rohit's notebook and
    gives a quick-yet-robust rigid initialisation before the affine stage takes
    over.
    """

    device: str = "cuda"
    winsorize: WinsorizeSettings = field(default_factory=WinsorizeSettings)
    moments_scale: int = 4
    affine: AffineSettings = field(default_factory=AffineSettings)
    greedy: GreedySettings = field(default_factory=GreedySettings)
    qc_middle_plane: Optional[int] = None
    qc_figsize: tuple[int, int] = (10, 6)

    @classmethod
    def with_overrides(
        cls, overrides: Optional[Dict[str, Any]] = None
    ) -> "FireANTsRegistrationConfig":
        cfg = cls()
        if overrides:
            _apply_overrides(cfg, overrides)
        return cfg


def _apply_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    """Recursively apply dictionary overrides to (nested) dataclasses."""

    for key, value in overrides.items():
        if not hasattr(target, key):
            raise AttributeError(f"Unknown configuration field '{key}'")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(target, key, value)


def _import_fireants():
    fireants_version = "unknown"
    try:
        from importlib import metadata as _md

        fireants_version = _md.version("fireants")
    except Exception:  # pragma: no cover - best-effort
        pass

    try:
        from fireants.io.image import Image as FAImage  # type: ignore
        from fireants.io.image import BatchedImages  # type: ignore
        from fireants.registration.affine import AffineRegistration  # type: ignore
        from fireants.registration.greedy import GreedyRegistration  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fireants (with GPU extensions) is required for registration"
        ) from exc
    try:
        from fireants.registration.moments import MomentsRegistration  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fireants moments registration not available"  # pragma: no cover
        ) from exc
    return (
        fireants_version,
        FAImage,
        BatchedImages,
        MomentsRegistration,
        AffineRegistration,
        GreedyRegistration,
    )


def _winsorize_image(
    image: sitk.Image, settings: WinsorizeSettings
) -> tuple[sitk.Image, Dict[str, float]]:
    if not settings.enabled:
        return image, {}
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    lower, upper = np.percentile(
        array, [settings.lower_percentile, settings.upper_percentile]
    )
    if upper <= lower:
        upper = lower + 1.0
    np.clip(array, lower, upper, out=array)
    array = (array - lower) / (upper - lower + 1e-8)
    winsorized = sitk.GetImageFromArray(array.astype(np.float32))
    winsorized.CopyInformation(image)
    # Mirror Rohit's notebook by printing the clipping bounds – handy when
    # troubleshooting intensity scaling without diving into the metadata file.
    print(
        "Winsorize stats:",
        {
            "lower_percentile": settings.lower_percentile,
            "upper_percentile": settings.upper_percentile,
            "lower_value": float(lower),
            "upper_value": float(upper),
        },
    )
    return winsorized, {"lower": float(lower), "upper": float(upper)}


def _compute_pixel_size_from_tiff(path: Path) -> Optional[Tuple[float, float]]:
    """Return lateral pixel size in micrometres using TIFF resolution tags."""

    with tifffile.TiffFile(path) as tif:
        tags = tif.pages[0].tags
        xres = tags.get("XResolution")
        yres = tags.get("YResolution")
        unit = tags.get("ResolutionUnit")

    if not xres or not yres or not unit:
        return None

    def _single(res_value) -> Optional[float]:
        if isinstance(res_value, tuple) and len(res_value) == 2:
            num, den = res_value
        else:
            return None
        if den == 0:
            return None
        return num / den

    px_per_unit_x = _single(xres.value)
    px_per_unit_y = _single(yres.value)
    if not px_per_unit_x or not px_per_unit_y:
        return None

    unit_val = unit.value if hasattr(unit, "value") else unit
    if unit_val == 2:  # inch
        factor = 25400.0  # microns per inch
    elif unit_val == 3:  # centimeter
        factor = 10000.0  # microns per cm
    else:
        return None

    size_x = factor / px_per_unit_x
    size_y = factor / px_per_unit_y
    return float(size_x), float(size_y)


def _lookup_z_step_um(animal: AnimalMetadata, session: AnatomySession) -> Optional[float]:
    """Read step_size_um_anatomy from the microscope metadata CSV if present."""

    try:
        metadata_dir = resolve_raw_path(Path(animal.root_dir) / "01_raw/2p/metadata")
    except FileNotFoundError:
        return None

    for csv_path in sorted(Path(metadata_dir).glob("*_metadata.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "parameter" not in df.columns or "value" not in df.columns:
            continue
        mask = df["parameter"].str.fullmatch("step_size_um_anatomy", case=False, na=False)
        if mask.any():
            raw_value = df.loc[mask, "value"].dropna().iloc[0]
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                continue
    return None


def _prepare_voxel_spacing(
    animal: Optional[AnimalMetadata], session: Optional[AnatomySession]
) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """Ensure session carries voxel spacing metadata; return (xy, z) in microns."""

    if animal is None or session is None:
        return None, None

    pixel_size = session.session_data.pixel_size_xy_um
    z_step = session.session_data.z_step_um

    raw_stack_path = resolve_raw_path(Path(animal.root_dir) / session.session_data.raw_path)

    if pixel_size is None:
        pixel_size = _compute_pixel_size_from_tiff(raw_stack_path)
        if pixel_size:
            session.session_data.pixel_size_xy_um = pixel_size

    if z_step is None:
        z_step = _lookup_z_step_um(animal, session)
        if z_step is not None:
            session.session_data.z_step_um = z_step

    return pixel_size, z_step


def register_two_photon_anatomy(
    *,
    animal_id: str,
    session_id: str,
    stack_path: Path,
    reference_path: Path,
    output_root: Path,
    config: Optional[FireANTsRegistrationConfig] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run FireANTs affine + greedy registration and persist outputs."""

    cfg = config or FireANTsRegistrationConfig()
    if overrides:
        _apply_overrides(cfg, overrides)

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        cfg.device = "cpu"

    (
        fireants_version,
        FAImage,
        BatchedImages,
        MomentsRegistration,
        AffineRegistration,
        GreedyRegistration,
    ) = _import_fireants()

    stack_path = Path(stack_path)
    reference_path = Path(reference_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not stack_path.exists():
        raise FileNotFoundError(stack_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    pixel_size_um, z_step_um = _prepare_voxel_spacing(animal_metadata, session_metadata)

    spacing_info = {
        "pixel_size_xy_um": list(pixel_size_um) if pixel_size_um else None,
        "z_step_um": z_step_um,
    }

    logger.info(
        "fireants_registration_start",
        extra={
            "animal_id": animal_id,
            "session_id": session_id,
            "stack_path": str(stack_path),
            "reference_path": str(reference_path),
            "config": asdict(cfg),
            "voxel_spacing_um": spacing_info,
        },
    )

    moving_image = sitk.ReadImage(str(stack_path))
    fixed_image = sitk.ReadImage(str(reference_path))

    spacing = list(moving_image.GetSpacing())
    if pixel_size_um:
        spacing[0] = pixel_size_um[0] / 1000.0
        spacing[1] = pixel_size_um[1] / 1000.0
    if z_step_um is not None and len(spacing) >= 3:
        spacing[2] = z_step_um / 1000.0
    moving_image.SetSpacing(tuple(spacing))

    moving_image, winsorize_stats_moving = _winsorize_image(moving_image, cfg.winsorize)
    fixed_image, winsorize_stats_fixed = _winsorize_image(fixed_image, cfg.winsorize)
    winsorized = cfg.winsorize.enabled and (winsorize_stats_moving or winsorize_stats_fixed)

    moving_fa = FAImage(moving_image, device=cfg.device, dtype=torch.float32)
    fixed_fa = FAImage(fixed_image, device=cfg.device, dtype=torch.float32)
    fixed_batch = BatchedImages(fixed_fa)
    moving_batch = BatchedImages(moving_fa)

    # Stage 1: rigid moment alignment (Rohit's notebook used scale=4). This
    # provides a fast coarse initialisation that greatly accelerates the affine
    # optimiser.
    moments = MomentsRegistration(
        scale=cfg.moments_scale, fixed_images=fixed_batch, moving_images=moving_batch
    )
    moments.optimize()
    init_affine = moments.get_affine_init().detach()
    # Emit the 3×4 affine (again matching the notebook output) so operators can
    # quickly verify the rigid offset.
    print("Moments init affine:\n", init_affine.cpu().numpy())

    # Stage 2: affine refinement with fused cross-correlation as configured
    # above. We feed in the rigid result via ``init_rigid`` to replicate the
    # interactive workflow exactly.
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
    affine_warp = affine.evaluate(fixed_batch, moving_batch)

    transforms_dir = output_root / "transforms"
    transforms_dir.mkdir(exist_ok=True)
    affine_transform_path = transforms_dir / f"{animal_id}_fireants_affine.mat"
    affine.save_as_ants_transforms(str(affine_transform_path))

    final_tensor = affine_warp
    greedy_transform_path: Optional[Path] = None
    greedy_inverse_path: Optional[Path] = None
    inverse_summary: Optional[Dict[str, float]] = None

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
        greedy_transform_path = transforms_dir / f"{animal_id}_fireants_greedy_warp.nii.gz"
        greedy_inverse_path = transforms_dir / f"{animal_id}_fireants_greedy_inverse_warp.nii.gz"
        greedy.save_as_ants_transforms(str(greedy_transform_path))
        greedy.save_as_ants_transforms(str(greedy_inverse_path), save_inverse=True)

        # After saving the forward warp we ask FireANTs for an inverse solution.
        # There is no public parameter to shorten the inverse solve, so we keep
        # the default behaviour but record simple residual metrics for QC.
        try:
            inverse_tensor = greedy.evaluate_inverse(fixed_batch, moving_batch)
            moving_tensor = moving_batch()
            diff = inverse_tensor - moving_tensor
            inverse_mse = float(torch.mean(diff**2).item())
            inverse_max = float(torch.max(torch.abs(diff)).item())
            inverse_summary = {
                "mse": inverse_mse,
                "max_abs": inverse_max,
            }
        except Exception:  # pragma: no cover - diagnostic only
            inverse_summary = None

    warped_volume = final_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    warped_stack_path = output_root / f"{animal_id}_anatomy_warped_fireants.tif"
    tifffile.imwrite(warped_stack_path, warped_volume)

    qc_dir = output_root / "qc"
    qc_dir.mkdir(exist_ok=True)
    qc_path = qc_dir / f"{animal_id}_fireants_qc.png"
    _write_qc_figure(
        reference=fixed_fa.array.squeeze().detach().cpu().numpy(),
        moving=moving_fa.array.squeeze().detach().cpu().numpy(),
        warped=warped_volume,
        output_path=qc_path,
        percentiles=(cfg.winsorize.lower_percentile, cfg.winsorize.upper_percentile),
        middle_plane=cfg.qc_middle_plane,
        figsize=cfg.qc_figsize,
    )

    if cfg.greedy.enabled:
        if inverse_summary is not None:
            print("Inverse QC:", inverse_summary)
            logger.info(
                "fireants_inverse_qc",
                extra={
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "metrics": inverse_summary,
                },
            )

    metadata = {
        "animal_id": animal_id,
        "session_id": session_id,
        "stack_path": str(stack_path),
        "reference_path": str(reference_path),
        "fireants_version": fireants_version,
        "config": asdict(cfg),
        "voxel_spacing_um": spacing_info,
        "winsorize": {
            "applied": winsorized,
            "moving": winsorize_stats_moving,
            "fixed": winsorize_stats_fixed,
        },
        "outputs": {
            "warped_stack": str(warped_stack_path),
            "affine_transform": str(affine_transform_path),
            "greedy_transform": str(greedy_transform_path) if greedy_transform_path else None,
            "greedy_inverse_transform": str(greedy_inverse_path) if greedy_inverse_path else None,
            "qc": str(qc_path),
        },
    }
    if cfg.greedy.enabled:
        metadata["inverse_qc"] = inverse_summary
    metadata_path = output_root / "fireants_registration_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(
        "fireants_registration_complete",
        extra={
            "animal_id": animal_id,
            "session_id": session_id,
            "outputs": metadata["outputs"],
            "voxel_spacing_um": spacing_info,
        },
    )

    results: Dict[str, Any] = {
        "warped_stack": warped_stack_path,
        "affine_transform": affine_transform_path,
        "metadata": metadata_path,
        "qc": qc_path,
    }
    if greedy_transform_path:
        results["greedy_transform"] = greedy_transform_path
    if greedy_inverse_path:
        results["greedy_inverse_transform"] = greedy_inverse_path
    results["voxel_spacing_um"] = spacing_info
    return results


def _write_qc_figure(
    *,
    reference: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    output_path: Path,
    percentiles: tuple[float, float],
    middle_plane: Optional[int],
    figsize: tuple[int, int],
) -> None:
    z_max = reference.shape[0]
    index = middle_plane if middle_plane is not None else z_max // 2
    index = int(np.clip(index, 0, z_max - 1))

    def _norm(slice_2d: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(slice_2d, percentiles)
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((slice_2d - lo) / (hi - lo), 0.0, 1.0)
        return out

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(_norm(reference[index]), cmap="gray")
    axes[0].set_title("Reference")
    axes[1].imshow(_norm(moving[index]), cmap="gray")
    axes[1].set_title("Moving (pre)")
    axes[2].imshow(_norm(warped[index]), cmap="gray")
    axes[2].set_title("Warpped (post)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
