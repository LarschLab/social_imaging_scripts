"""FireANTs registration of confocal stacks to two-photon anatomy."""

from __future__ import annotations

import copy
import logging
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING

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
from .prematch import (
    XYMIPPrematchResult,
    XYMIPPrematchSettings,
    run_xy_mip_prematch,
)

if TYPE_CHECKING:
    from ..pipeline.processing_log import (
        build_processing_log_path,
        load_processing_log,
    )


logger = logging.getLogger(__name__)


def _load_manual_prematch_from_log(
    animal_id: str,
    session_id: str,
    output_base_dir: Path,
    processing_log_config,
) -> Optional[Dict[str, float]]:
    """Load manual prematch parameters from processing log if available.
    
    Args:
        animal_id: Animal identifier
        session_id: Confocal session identifier
        output_base_dir: Base output directory (e.g., /mnt/f/johannes/pipelineOut)
        processing_log_config: ProcessingLogConfig from pipeline config
    
    Returns:
        Dictionary with translation_x_px, translation_y_px, rotation_deg if found, else None
    """
    try:
        # Import here to avoid circular dependency
        from ..pipeline.processing_log import (
            build_processing_log_path,
            load_processing_log,
        )
        
        log_path = build_processing_log_path(
            processing_log_config,
            animal_id,
            base_dir=output_base_dir,
        )
        
        if not log_path.exists():
            return None
        
        log = load_processing_log(log_path)
        stage = log.stages.get("confocal_to_anatomy_registration")
        
        if not stage:
            return None
        
        manual_prematch = stage.parameters.get("manual_prematch", {})
        session_data = manual_prematch.get(session_id)
        
        if session_data:
            logger.info(
                "Loaded manual prematch from processing log: x=%.1f px, y=%.1f px, rot=%.1f°",
                session_data.get("translation_x_px", 0.0),
                session_data.get("translation_y_px", 0.0),
                session_data.get("rotation_deg", 0.0),
            )
        
        return session_data
    
    except Exception as e:
        logger.warning("Could not load manual prematch from processing log: %s", e)
        return None


def _apply_rotation_to_volume_legacy(volume: np.ndarray, rotation_deg: float, order: int = 1) -> np.ndarray:
    """Apply scipy rotation to volume (LEGACY approach for comparison).
    
    This pre-applies rotation before FireANTs, which breaks the transform chain
    but was the old behavior for comparison purposes.
    """
    from scipy.ndimage import rotate
    
    if abs(rotation_deg) < 0.1:
        return volume
    
    # Rotate each Z-slice around its center
    rotated = np.zeros_like(volume)
    for z in range(volume.shape[0]):
        rotated[z] = rotate(
            volume[z],
            angle=rotation_deg,
            reshape=False,
            order=order,
            mode='constant',
            cval=0.0,
        )
    
    return rotated


def _build_fov_overlap_mask(
    moving_shape: tuple[int, int, int],
    fixed_shape: tuple[int, int, int],
    moving_spacing: tuple[float, float, float],
    fixed_spacing: tuple[float, float, float],
    rotation_deg: float = 0.0,
    translation_um: tuple[float, float, float] = (0.0, 0.0, 0.0),
    taper_um: float = 50.0,
    margin_xy: float = 0.0,
    margin_z: float = 0.0,
) -> np.ndarray:
    """Build mask that restricts registration to anatomy FOV in confocal space.
    
    This creates a soft-edged mask that:
    1. Has weight=1.0 where anatomy FOV overlaps with confocal
    2. Tapers to 0.0 over taper_um distance at boundaries
    3. Accounts for rotation and translation from prematch
    4. Prevents peripheral confocal regions from affecting registration
    
    Args:
        moving_shape: Confocal shape (Z, Y, X) in voxels
        fixed_shape: Anatomy shape (Z, Y, X) in voxels
        moving_spacing: Confocal spacing (x, y, z) in micrometers
        fixed_spacing: Anatomy spacing (x, y, z) in micrometers
        rotation_deg: Expected rotation from prematch
        translation_um: Expected translation (x, y, z) from prematch
        taper_um: Width of soft taper zone at boundaries in micrometers
        margin_xy: Additional margin to add/subtract in XY (fraction or voxels)
        margin_z: Additional margin to add/subtract in Z (fraction or voxels)
    
    Returns:
        Mask array with same shape as moving volume, values in [0, 1]
    """
    import math
    
    mz, my, mx = moving_shape
    fz, fy, fx = fixed_shape
    
    # Fixed FOV extent in physical coordinates (micrometers)
    fixed_extent_x = fx * fixed_spacing[0]
    fixed_extent_y = fy * fixed_spacing[1]
    fixed_extent_z = fz * fixed_spacing[2]
    
    # Apply margins to fixed extent
    margin_x_vox = _margin_to_voxels(margin_xy, fx)
    margin_y_vox = _margin_to_voxels(margin_xy, fy)
    margin_z_vox = _margin_to_voxels(margin_z, fz)
    
    fixed_extent_x -= 2 * margin_x_vox * fixed_spacing[0]
    fixed_extent_y -= 2 * margin_y_vox * fixed_spacing[1]
    fixed_extent_z -= 2 * margin_z_vox * fixed_spacing[2]
    
    # Fixed FOV center in its own coordinates
    fixed_center_x = (fx - 1) / 2.0 * fixed_spacing[0]
    fixed_center_y = (fy - 1) / 2.0 * fixed_spacing[1]
    fixed_center_z = (fz - 1) / 2.0 * fixed_spacing[2]
    
    # Moving FOV center
    moving_center_x = (mx - 1) / 2.0 * moving_spacing[0]
    moving_center_y = (my - 1) / 2.0 * moving_spacing[1]
    moving_center_z = (mz - 1) / 2.0 * moving_spacing[2]
    
    # Create coordinate grids for moving volume (in micrometers)
    x_coords = np.arange(mx, dtype=np.float32) * moving_spacing[0]
    y_coords = np.arange(my, dtype=np.float32) * moving_spacing[1]
    z_coords = np.arange(mz, dtype=np.float32) * moving_spacing[2]
    
    # Center coordinates on moving center
    x_coords -= moving_center_x
    y_coords -= moving_center_y
    z_coords -= moving_center_z
    
    # Apply FORWARD transformation to see where moving voxels land in fixed frame
    # Forward: fixed = R(moving) + T
    theta = math.radians(rotation_deg)  # Forward rotation (not inverse)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    # Build 3D coordinate grids
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    
    # Rotate XY coordinates (Z unchanged)
    xx_rot = xx * cos_t - yy * sin_t
    yy_rot = xx * sin_t + yy * cos_t
    
    # Apply forward translation
    xx_rot += translation_um[0]
    yy_rot += translation_um[1]
    zz += translation_um[2]
    
    # Now xx_rot, yy_rot, zz are in "anatomy reference frame"
    # Check if they fall within anatomy FOV extent
    
    # Distance from center of anatomy FOV
    half_x = fixed_extent_x / 2.0
    half_y = fixed_extent_y / 2.0
    half_z = fixed_extent_z / 2.0
    
    # Compute distance outside boundary for each axis
    dist_outside_x = np.maximum(0, np.abs(xx_rot) - half_x)
    dist_outside_y = np.maximum(0, np.abs(yy_rot) - half_y)
    dist_outside_z = np.maximum(0, np.abs(zz) - half_z)
    
    # Euclidean distance to nearest anatomy boundary
    dist_outside = np.sqrt(dist_outside_x**2 + dist_outside_y**2 + dist_outside_z**2)
    
    # Build soft-edged mask using cosine taper
    mask = np.ones_like(dist_outside)
    if taper_um > 0:
        taper_region = (dist_outside > 0) & (dist_outside < taper_um)
        # Cosine taper: 1.0 at boundary, 0.0 at taper_um
        taper_vals = 0.5 * (1.0 + np.cos(np.pi * dist_outside[taper_region] / taper_um))
        mask[taper_region] = taper_vals
        mask[dist_outside >= taper_um] = 0.0
    else:
        mask[dist_outside > 0] = 0.0
    
    return mask.astype(np.float32)


def _manual_prematch_to_result(
    manual_prematch: Dict[str, float],
    moving_spacing_um: Tuple[float, float, float],
) -> XYMIPPrematchResult:
    """Convert manual prematch pixel values to XYMIPPrematchResult.
    
    Args:
        manual_prematch: Dict with translation_x_px, translation_y_px, rotation_deg
        moving_spacing_um: Voxel spacing (x, y, z) in micrometers for the moving (confocal) stack
    
    Returns:
        XYMIPPrematchResult with translation in micrometers
    """
    x_px = float(manual_prematch.get("translation_x_px", 0.0))
    y_px = float(manual_prematch.get("translation_y_px", 0.0))
    rotation_deg = float(manual_prematch.get("rotation_deg", 0.0))
    
    # Convert pixel translations to micrometers
    # Note: manual GUI saves shifts in anatomy pixel space, but we need to convert to moving voxel space
    # The translation_vox is in (x, y, z) order, matching moving stack coordinates
    translation_vox = np.array([x_px, y_px, 0.0], dtype=np.float64)
    
    # Convert to physical units using moving stack spacing
    translation_um = np.array([
        translation_vox[0] * moving_spacing_um[0],
        translation_vox[1] * moving_spacing_um[1],
        0.0,
    ], dtype=np.float64)
    
    # Create a result that looks like it came from automated prematch
    result = XYMIPPrematchResult(
        rotation_deg=rotation_deg,
        translation_vox=translation_vox,
        translation_um=translation_um,
        score=1.0,  # Manual prematch is assumed perfect
        delta_pixels=np.array([y_px, x_px], dtype=np.float64),
        matched_centre_pixels=np.array([0.0, 0.0], dtype=np.float64),
        peak_index=(0, 0),
        downsample_scale=1.0,
        resample_factors=(1.0, 1.0),
        angle_records=[{"angle_deg": rotation_deg, "score": 1.0}],
        applied=True,
    )
    
    return result


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
    soft_edges: bool,
) -> np.ndarray:
    """Binary mask that keeps the central region of a volume."""

    z, y, x = shape
    margin_z_vox = min(_margin_to_voxels(margin_z, z), z // 2)
    margin_y_vox = min(_margin_to_voxels(margin_xy, y), y // 2)
    margin_x_vox = min(_margin_to_voxels(margin_xy, x), x // 2)

    if margin_z_vox <= 0 and margin_y_vox <= 0 and margin_x_vox <= 0:
        return np.ones(shape, dtype=np.float32)

    def _axis_window(length: int, margin: int) -> np.ndarray:
        if margin <= 0:
            return np.ones(length, dtype=np.float32)
        window = np.ones(length, dtype=np.float32)
        if soft_edges:
            idx = np.arange(margin, dtype=np.float32) / max(margin, 1)
            taper = 0.5 * (1 - np.cos(np.pi * idx))
            window[:margin] = taper[::-1]
            window[-margin:] = taper
        else:
            window[:margin] = 0.0
            window[-margin:] = 0.0
        return window

    z_window = _axis_window(z, margin_z_vox)
    y_window = _axis_window(y, margin_y_vox)
    x_window = _axis_window(x, margin_x_vox)
    mask = z_window[:, None, None] * y_window[None, :, None] * x_window[None, None, :]
    return mask.astype(np.float32)


def _compute_centroid(volume: np.ndarray) -> np.ndarray:
    total = float(volume.sum(dtype=np.float64))
    if total <= 0.0:
        return np.array([(dim - 1) / 2.0 for dim in volume.shape], dtype=np.float64)
    coords = np.indices(volume.shape, dtype=np.float64)
    return np.array(
        [(coords[i] * volume).sum(dtype=np.float64) / total for i in range(volume.ndim)],
        dtype=np.float64,
    )


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


def _build_prematch_affine(
    prematch: XYMIPPrematchResult,
    moving_shape: tuple[int, int, int],
    spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Construct a 4x4 affine matrix from the prematch rotation/translation."""

    theta = math.radians(float(prematch.rotation_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation = np.eye(3, dtype=np.float64)
    rotation[0, 0] = cos_t
    rotation[0, 1] = -sin_t
    rotation[1, 0] = sin_t
    rotation[1, 1] = cos_t

    center = np.array(
        [
            0.5 * (moving_shape[2] - 1) * spacing[0],
            0.5 * (moving_shape[1] - 1) * spacing[1],
            0.5 * (moving_shape[0] - 1) * spacing[2],
        ],
        dtype=np.float64,
    )
    translation = prematch.translation_um.astype(np.float64, copy=False)
    total_translation = center - rotation @ center + translation

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation
    affine[:3, 3] = total_translation
    return affine


def _apply_prematch_to_affine(
    affine_tensor: torch.Tensor,
    prematch_matrix: np.ndarray,
) -> torch.Tensor:
    """Left-multiply the existing affine initialisation with the prematch matrix."""

    if prematch_matrix.shape != (4, 4):
        raise ValueError("prematch_matrix must be 4x4 in homogeneous coordinates")

    arr = affine_tensor.detach().cpu().numpy()
    squeeze = arr[0] if arr.ndim == 3 else arr

    if squeeze.shape == (3, 4):
        base = np.eye(4, dtype=np.float64)
        base[:3, :3] = squeeze[:, :3]
        base[:3, 3] = squeeze[:, 3]
        combined = base @ prematch_matrix
        updated = combined[:3, :]
        out = torch.from_numpy(updated).to(device=affine_tensor.device, dtype=affine_tensor.dtype)
        return out.unsqueeze(0)

    if squeeze.shape == (4, 4):
        combined = squeeze @ prematch_matrix
        out = torch.from_numpy(combined).to(device=affine_tensor.device, dtype=affine_tensor.dtype)
        if arr.ndim == 3:
            return out.unsqueeze(0)
        return out

    logger.warning("Unexpected affine init shape %s; skipping prematch seed", tuple(squeeze.shape))
    return affine_tensor


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
    prematch_settings: Optional[XYMIPPrematchSettings] = None,
    warped_channel_template: str,
    metadata_filename: str,
    transforms_subdir: Path,
    qc_subdir: Path,
    reference_channel_name: str,
    mask_margin_xy: float,
    mask_margin_z: float,
    mask_soft_edges: bool,
    histogram_match: bool,
    histogram_levels: int,
    histogram_match_points: int,
    histogram_threshold_at_mean: bool,
    initial_translation_mode: str,
    crop_to_extent: bool,
    crop_padding_um: float,
    output_base_dir: Optional[Path] = None,
    processing_log_config = None,
    use_legacy_rotation_cropping: bool = False,
) -> Dict[str, object]:
    """Register a confocal channel to two-photon anatomy and warp additional channels.
    
    Args:
        output_base_dir: Base output directory for loading manual prematch from processing log
        processing_log_config: ProcessingLogConfig from pipeline config for loading manual prematch
        use_legacy_rotation_cropping: If True, use old approach (pre-rotate with scipy, then crop).
                                       If False (default), seed prematch into FireANTs without pre-rotation.
        ... (other existing parameters)
    """

    cfg = copy.deepcopy(config)
    prematch_settings = prematch_settings or XYMIPPrematchSettings()

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
    original_shape = moving_array.shape
    spacing = tuple(float(s) for s in voxel_spacing_um)

    # ============================================================================
    # STEP 1: Calculate prematch on FULL (uncropped) images
    # ============================================================================
    prematch_result: Optional[XYMIPPrematchResult] = None
    manual_prematch_data = None
    
    # Check for manual prematch from processing log first
    if output_base_dir is not None and processing_log_config is not None:
        manual_prematch_data = _load_manual_prematch_from_log(
            animal_id,
            confocal_session_id,
            output_base_dir,
            processing_log_config,
        )
        if manual_prematch_data:
            logger.info("Using manual prematch from processing log (overrides automated prematch)")
            prematch_result = _manual_prematch_to_result(
                manual_prematch_data,
                spacing,
            )
    
    # Fall back to automated prematch if no manual prematch found
    if prematch_result is None and prematch_settings.enabled:
        try:
            prematch_result = run_xy_mip_prematch(
                moving_array,
                fixed_array,
                spacing,
                fixed_spacing_um,
                prematch_settings,
            )
            if prematch_result is None:
                logger.warning("Prematch did not return a result; falling back to FireANTs moments init")
            elif prematch_result.applied:
                logger.info(
                    (
                        "Automated prematch seed accepted: θ=%.2f°, score=%.3f, translation=(%.1f, %.1f, %.1f) µm"
                    ),
                    prematch_result.rotation_deg,
                    prematch_result.score,
                    prematch_result.translation_um[0],
                    prematch_result.translation_um[1],
                    prematch_result.translation_um[2],
                )
            else:
                logger.info(
                    "Prematch score %.3f below threshold %.3f; ignoring prematch seed",
                    prematch_result.score,
                    prematch_settings.min_score,
                )
        except Exception:
            logger.exception("Prematch heuristic failed; continuing without seed")
            prematch_result = None

    # ============================================================================
    # STEP 2: Legacy cropping (for comparison) OR modern approach
    # ============================================================================
    crop_slices = (slice(None), slice(None), slice(None))
    crop_info: dict[str, list[int]] | None = None
    
    if use_legacy_rotation_cropping and crop_to_extent:
        # LEGACY PATH: Hard crop to anatomy extent (original working behavior)
        # This was the approach before today's refactoring
        crop_slices, crop_info = _compute_extent_crop_slices(
            moving_array.shape,
            voxel_spacing_um,
            fixed_array.shape,
            fixed_spacing_um,
            crop_padding_um,
        )
        logger.info(
            "[LEGACY MODE] Cropping confocal: y:%s x:%s padding %.2f µm",
            crop_info["y_vox"] if crop_info else None,
            crop_info["x_vox"] if crop_info else None,
            crop_padding_um,
        )
        moving_array = moving_array[crop_slices]
        logger.info("[LEGACY MODE] Using original hard-crop approach (no pre-rotation)")
    
    # Modern approach: No cropping
    # FOV masking code below (currently disabled)
    if False and crop_to_extent and prematch_result is not None and prematch_result.applied:
        # Use FOV-aware mask instead of hard cropping
        logger.info("=" * 80)
        logger.info("FOV OVERLAP MASKING (rotation-aware)")
        logger.info("=" * 80)
        logger.info("  Moving (confocal) shape: %s", moving_array.shape)
        logger.info("  Moving spacing (µm): %.3f × %.3f × %.3f", spacing[0], spacing[1], spacing[2])
        logger.info("  Fixed (anatomy) shape: %s", fixed_array.shape)
        logger.info("  Fixed spacing (µm): %.3f × %.3f × %.3f", 
                    fixed_spacing_um[0], fixed_spacing_um[1], fixed_spacing_um[2])
        logger.info("  Prematch rotation: %.2f°", prematch_result.rotation_deg)
        logger.info("  Prematch translation: (%.1f, %.1f) µm", 
                    prematch_result.translation_um[0], prematch_result.translation_um[1])
        logger.info("  Taper width: 50.0 µm")
        logger.info("  Margin XY: %.3f, Margin Z: %.3f", mask_margin_xy, mask_margin_z)
        
        fov_mask = _build_fov_overlap_mask(
            moving_array.shape,
            fixed_array.shape,
            spacing,
            fixed_spacing_um,
            rotation_deg=prematch_result.rotation_deg,
            translation_um=tuple(prematch_result.translation_um),
            taper_um=50.0,  # 50 µm soft edge
            margin_xy=mask_margin_xy,
            margin_z=mask_margin_z,
        )
        
        # Report mask statistics
        mask_zero = np.sum(fov_mask == 0.0)
        mask_one = np.sum(fov_mask == 1.0)
        mask_taper = np.sum((fov_mask > 0.0) & (fov_mask < 1.0))
        mask_total = fov_mask.size
        logger.info("  Mask statistics:")
        logger.info("    - Fully masked (0.0): %d voxels (%.1f%%)", 
                    mask_zero, 100.0 * mask_zero / mask_total)
        logger.info("    - Fully included (1.0): %d voxels (%.1f%%)", 
                    mask_one, 100.0 * mask_one / mask_total)
        logger.info("    - Tapered (0.0-1.0): %d voxels (%.1f%%)", 
                    mask_taper, 100.0 * mask_taper / mask_total)
        logger.info("  Applying FOV overlap mask to moving volume")
        logger.info("=" * 80)
        
        moving_array = moving_array * fov_mask
        fov_mask_applied = True
    elif False and crop_to_extent:
        # No prematch available, use simple central crop (DISABLED - too aggressive)
        logger.warning("crop_to_extent=True but no prematch available; using simple central mask")
        simple_mask = _build_central_mask(
            moving_array.shape, 
            mask_margin_xy * 2,  # More aggressive without prematch
            mask_margin_z * 2, 
            mask_soft_edges
        )
        moving_array = moving_array * simple_mask
        fov_mask_applied = True
    else:
        fov_mask_applied = False

    # ============================================================================
    # STEP 3: Apply central masks to both volumes
    # ============================================================================
    # Only apply central mask to moving if FOV mask wasn't already applied
    if not fov_mask_applied:
        moving_mask = _build_central_mask(moving_array.shape, mask_margin_xy, mask_margin_z, mask_soft_edges)
        if np.any(moving_mask != 1.0):
            logger.info(
                "Applying central mask to moving volume (mask_margin_xy=%.3f, mask_margin_z=%.3f)",
                mask_margin_xy,
                mask_margin_z,
            )
            moving_array *= moving_mask
    else:
        logger.info("Skipping additional central mask for moving volume (FOV mask already applied)")
        moving_mask = np.ones_like(moving_array)
    
    # Always apply central mask to fixed volume
    fixed_mask = _build_central_mask(fixed_array.shape, mask_margin_xy, mask_margin_z, mask_soft_edges)
    if np.any(fixed_mask != 1.0):
        logger.info(
            "Applying central mask to fixed volume (mask_margin_xy=%.3f, mask_margin_z=%.3f)",
            mask_margin_xy,
            mask_margin_z,
        )
    fixed_array *= fixed_mask

    moving_image = _to_sitk_image(moving_array, spacing)
    fixed_image = _to_sitk_image(fixed_array, fixed_spacing_um)
    dim = moving_image.GetDimension()

    # No more "crop" mode since we use FOV masking instead of hard cropping
    translation_mode = (initial_translation_mode or "none").lower()
    translation_vec = np.zeros(3, dtype=np.float64)
    if translation_mode == "centroid":
        moving_centroid = _compute_centroid(moving_array)
        fixed_centroid = _compute_centroid(fixed_array)
        delta_vox = np.array(
            [
                moving_centroid[2] - fixed_centroid[2],
                moving_centroid[1] - fixed_centroid[1],
                moving_centroid[0] - fixed_centroid[0],
            ],
            dtype=np.float64,
        )
        translation_vec = -delta_vox * np.array(
            [spacing[0], spacing[1], float(fixed_spacing_um[2])],
            dtype=np.float64,
        )
    else:
        translation_mode = "none"

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

    # ============================================================================
    # STEP 5: Apply prematch to FireANTs initialization
    # Both legacy and modern modes seed the FULL prematch (rotation + translation)
    # ============================================================================
    prematch_affine_matrix: Optional[np.ndarray] = None
    if prematch_result is not None and prematch_result.applied:
        try:
            logger.info(
                "Seeding FireANTs with full prematch (rotation + translation): "
                "θ=%.2f°, Δx=%.3f µm, Δy=%.3f µm",
                prematch_result.rotation_deg,
                prematch_result.translation_um[0],
                prematch_result.translation_um[1],
            )
            prematch_affine_matrix = _build_prematch_affine(prematch_result, moving_array.shape, spacing)
            init_affine = _apply_prematch_to_affine(init_affine, prematch_affine_matrix)
        except Exception:
            logger.exception("Failed to apply prematch affine seed; continuing without it")
            prematch_affine_matrix = None

    if translation_mode != "none" and np.linalg.norm(translation_vec) > 1e-6:
        logger.info(
            "Seeding affine translation (mode=%s): Δx=%.3f µm, Δy=%.3f µm, Δz=%.3f µm",
            translation_mode,
            translation_vec[0],
            translation_vec[1],
            translation_vec[2],
        )
        tr = torch.tensor(translation_vec, device=init_affine.device, dtype=init_affine.dtype)
        if init_affine.shape[-2:] == (dim, dim + 1):
            init_affine = init_affine.clone()
            init_affine[:, 0, -1] += tr[0]
            init_affine[:, 1, -1] += tr[1]
            if dim > 2:
                init_affine[:, 2, -1] += tr[2]
        elif init_affine.shape[-2:] == (dim + 1, dim + 1):
            init_affine = init_affine.clone()
            init_affine[:, 0, -1] += tr[0]
            init_affine[:, 1, -1] += tr[1]
            init_affine[:, 2, -1] += tr[2]
        else:
            logger.warning(
                "Unexpected affine init shape %s; skipping translation seed.",
                tuple(init_affine.shape),
            )

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
        
        # Legacy mode: apply same cropping
        if use_legacy_rotation_cropping and crop_info is not None:
            arr = arr[crop_slices]
        
        # Check shape matches reference
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
            "soft_edges": bool(mask_soft_edges),
        },
        "histogram_match": {
            "enabled": bool(histogram_match),
            "levels": int(histogram_levels),
            "match_points": int(histogram_match_points),
            "threshold_at_mean": bool(histogram_threshold_at_mean),
        },
        "initial_translation": {
            "mode": translation_mode,
            "vector_um": [float(translation_vec[0]), float(translation_vec[1]), float(translation_vec[2])],
        },
        "fov_masking": {
            "enabled": bool(crop_to_extent),
            "note": "FOV overlap mask applied instead of hard cropping to preserve dimensions",
        },
        "prematch": {
            "enabled": bool(prematch_settings.enabled),
            "settings": asdict(prematch_settings),
            "result": prematch_result.to_metadata() if prematch_result else None,
            "affine_matrix": prematch_affine_matrix.tolist() if prematch_affine_matrix is not None else None,
            "note": "Full prematch (rotation + translation) seeded into FireANTs; rotation not pre-applied",
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
        "prematch": prematch_result.to_metadata() if prematch_result else None,
    }
