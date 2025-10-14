"""XY maximum-intensity projection prematching for confocal→anatomy alignment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import match_template
from skimage.transform import resize, rotate


@dataclass
class XYMIPPrematchSettings:
    """Runtime knobs for the XY-MIP prematch heuristic."""

    enabled: bool = False
    mip_percentiles: tuple[float, float] = (2.0, 99.5)
    gaussian_sigma_px: float = 1.0
    downsample_max_dim: int = 512
    rotation_coarse_step_deg: float = 22.5
    rotation_fine_step_deg: float = 1.5
    rotation_fine_window_deg: float = 6.0
    evaluate_opposite_flip: bool = True
    min_score: float = 0.15

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "XYMIPPrematchSettings":
        kwargs = dict(data)
        percentiles = kwargs.get("mip_percentiles")
        if percentiles is not None:
            kwargs["mip_percentiles"] = tuple(float(p) for p in percentiles)
        return cls(**kwargs)


@dataclass
class XYMIPPrematchResult:
    """Summary of the prematch outcome."""

    rotation_deg: float
    translation_vox: np.ndarray  # (x, y, z) in moving voxels
    translation_um: np.ndarray  # (x, y, z)
    score: float
    delta_pixels: np.ndarray  # (y, x) offset of matched centre in downsampled grid
    matched_centre_pixels: np.ndarray  # (y, x) centre in downsampled grid
    peak_index: tuple[int, int]
    downsample_scale: float
    resample_factors: tuple[float, float]  # fixed→moving scaling (y, x)
    angle_records: list[dict[str, float]]
    applied: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "rotation_deg": float(self.rotation_deg),
            "translation_um": [float(v) for v in self.translation_um],
            "translation_vox": [float(v) for v in self.translation_vox],
            "score": float(self.score),
            "delta_pixels": [float(v) for v in self.delta_pixels],
            "matched_centre_pixels": [float(v) for v in self.matched_centre_pixels],
            "peak_index": [int(self.peak_index[0]), int(self.peak_index[1])],
            "downsample_scale": float(self.downsample_scale),
            "resample_factors": [float(v) for v in self.resample_factors],
            "evaluated_angles": self.angle_records,
            "applied": bool(self.applied),
        }


def run_xy_mip_prematch(
    moving: np.ndarray,
    fixed: np.ndarray,
    moving_spacing_um: Sequence[float],
    fixed_spacing_um: Sequence[float],
    settings: XYMIPPrematchSettings,
) -> XYMIPPrematchResult | None:
    if not settings.enabled:
        return None

    if moving.ndim != 3 or fixed.ndim != 3:
        raise ValueError("Prematch expects 3D volumes (Z, Y, X).")

    moving_spacing_x, moving_spacing_y = float(moving_spacing_um[0]), float(moving_spacing_um[1])
    fixed_spacing_x, fixed_spacing_y = float(fixed_spacing_um[0]), float(fixed_spacing_um[1])
    moving_spacing_z = float(moving_spacing_um[2]) if len(moving_spacing_um) > 2 else float(moving_spacing_um[0])

    moving_mip = np.max(moving.astype(np.float32, copy=False), axis=0)
    fixed_mip = np.max(fixed.astype(np.float32, copy=False), axis=0)

    moving_mip = _clip_and_normalise(moving_mip, settings.mip_percentiles)
    fixed_mip = _clip_and_normalise(fixed_mip, settings.mip_percentiles)

    resample_factor_y = fixed_spacing_y / moving_spacing_y if moving_spacing_y > 0 else 1.0
    resample_factor_x = fixed_spacing_x / moving_spacing_x if moving_spacing_x > 0 else 1.0
    fixed_mip = _resize_image(
        fixed_mip,
        scale_y=resample_factor_y,
        scale_x=resample_factor_x,
    )

    if settings.gaussian_sigma_px > 0:
        moving_mip = gaussian_filter(moving_mip, settings.gaussian_sigma_px)
        fixed_mip = gaussian_filter(fixed_mip, settings.gaussian_sigma_px)

    downsample_scale = _compute_downsample_scale(moving_mip.shape, settings.downsample_max_dim)
    if downsample_scale < 1.0:
        moving_ds = _resize_by_scale(moving_mip, downsample_scale)
        fixed_ds = _resize_by_scale(fixed_mip, downsample_scale)
    else:
        moving_ds = moving_mip
        fixed_ds = fixed_mip

    if (
        fixed_ds.shape[0] >= moving_ds.shape[0]
        or fixed_ds.shape[1] >= moving_ds.shape[1]
    ):
        raise ValueError(
            "Fixed XY MIP is not smaller than the moving MIP after resampling; "
            "prematch requires the moving stack to have a larger XY extent."
        )

    best = _evaluate_angles(moving_ds, fixed_ds, settings)
    if best is None:
        return None

    delta_y_ds, delta_x_ds = best["delta_pixels"]
    delta_x_vox = delta_x_ds / downsample_scale
    delta_y_vox = delta_y_ds / downsample_scale

    translation_vox = np.array([-delta_x_vox, -delta_y_vox, 0.0], dtype=np.float64)
    translation_um = np.array(
        [
            translation_vox[0] * moving_spacing_x,
            translation_vox[1] * moving_spacing_y,
            translation_vox[2] * moving_spacing_z,
        ],
        dtype=np.float64,
    )

    result = XYMIPPrematchResult(
        rotation_deg=best["angle_deg"],
        translation_vox=translation_vox,
        translation_um=translation_um,
        score=best["score"],
        delta_pixels=np.array([delta_y_ds, delta_x_ds], dtype=np.float64),
        matched_centre_pixels=np.array(best["matched_centre"], dtype=np.float64),
        peak_index=best["peak_index"],
        downsample_scale=downsample_scale,
        resample_factors=(resample_factor_y, resample_factor_x),
        angle_records=best["angle_records"],
        applied=best["score"] >= settings.min_score,
    )
    return result


def _clip_and_normalise(image: np.ndarray, percentiles: Sequence[float]) -> np.ndarray:
    lower, upper = float(percentiles[0]), float(percentiles[1])
    lower = max(0.0, min(100.0, lower))
    upper = max(lower + 1e-3, min(100.0, upper))
    lo = np.percentile(image, lower)
    hi = np.percentile(image, upper)
    if hi <= lo:
        hi = lo + 1e-6
    clipped = np.clip(image, lo, hi)
    clipped -= clipped.min()
    max_val = clipped.max()
    if max_val > 0:
        clipped /= max_val
    return clipped.astype(np.float32, copy=False)


def _resize_image(image: np.ndarray, scale_y: float, scale_x: float) -> np.ndarray:
    if scale_y == 1.0 and scale_x == 1.0:
        return image
    new_h = max(1, int(round(image.shape[0] * scale_y)))
    new_w = max(1, int(round(image.shape[1] * scale_x)))
    if new_h == image.shape[0] and new_w == image.shape[1]:
        return image
    resized = resize(
        image,
        (new_h, new_w),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32, copy=False)


def _resize_by_scale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    new_h = max(1, int(round(image.shape[0] * scale)))
    new_w = max(1, int(round(image.shape[1] * scale)))
    resized = resize(
        image,
        (new_h, new_w),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32, copy=False)


def _compute_downsample_scale(shape: Sequence[int], max_dim: int) -> float:
    max_current = float(max(shape))
    if max_current <= max_dim:
        return 1.0
    return max_dim / max_current


def _evaluate_angles(
    moving_ds: np.ndarray,
    fixed_ds: np.ndarray,
    settings: XYMIPPrematchSettings,
) -> dict[str, Any] | None:
    evaluated: dict[float, dict[str, Any]] = {}
    angle_records: list[dict[str, float]] = []
    best: dict[str, Any] | None = None

    def _record(angle: float, score: float, peak, matched_center):
        angle_records.append(
            {
                "angle_deg": float(angle),
                "score": float(score),
            }
        )
        return {
            "angle_deg": float(angle),
            "score": float(score),
            "peak_index": (int(peak[0]), int(peak[1])),
            "matched_centre": (float(matched_center[0]), float(matched_center[1])),
        }

    def _evaluate(angle: float) -> dict[str, Any]:
        rotated = rotate(
            moving_ds,
            angle,
            resize=False,
            order=1,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        ).astype(np.float32, copy=False)
        response = match_template(rotated, fixed_ds)
        peak = np.unravel_index(np.argmax(response), response.shape)
        score = float(response[peak])
        matched_centre_y = peak[0] + fixed_ds.shape[0] / 2.0
        matched_centre_x = peak[1] + fixed_ds.shape[1] / 2.0
        delta_y = matched_centre_y - rotated.shape[0] / 2.0
        delta_x = matched_centre_x - rotated.shape[1] / 2.0
        record = _record(angle, score, peak, (matched_centre_y, matched_centre_x))
        record["delta_pixels"] = (float(delta_y), float(delta_x))
        evaluated[round(angle % 360.0, 4)] = record
        return record

    coarse_step = max(settings.rotation_coarse_step_deg, 1e-3)
    n_steps = max(1, int(math.ceil(360.0 / coarse_step)))
    coarse_angles = [(i * coarse_step) % 360.0 for i in range(n_steps)]

    for angle in coarse_angles:
        record = _evaluate(angle)
        if best is None or record["score"] > best["score"]:
            best = record

    if best is None:
        return None

    fine_half_window = max(settings.rotation_fine_window_deg, settings.rotation_fine_step_deg)
    fine_step = max(settings.rotation_fine_step_deg, 1e-3)
    fine_angles: Iterable[float]
    fine_angles = np.arange(
        best["angle_deg"] - fine_half_window,
        best["angle_deg"] + fine_half_window + fine_step / 2.0,
        fine_step,
    )
    for angle in fine_angles:
        key = round(angle % 360.0, 4)
        if key in evaluated:
            continue
        record = _evaluate(angle % 360.0)
        if record["score"] > best["score"]:
            best = record

    if settings.evaluate_opposite_flip:
        flip_angle = (best["angle_deg"] + 180.0) % 360.0
        key = round(flip_angle, 4)
        if key not in evaluated:
            record = _evaluate(flip_angle)
            if record["score"] > best["score"]:
                best = record

    if best is None:
        return None

    best["angle_records"] = angle_records
    return best
