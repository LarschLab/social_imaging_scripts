"""Prototype functional→anatomy registration using coarse template matching.

Each functional plane is matched against all anatomy slices using
scale-constrained NCC.  The solver searches a discrete set of scales within
``scale_range`` and finds the best (z, y, x) placement via FFT-based template
matching.  Optional subpixel refinement is available through phase correlation.

The intent is to mimic the previous hierarchical matcher: optimise for z index,
uniform in-plane scale, and x/y translation only.  No rotations or shearing are
estimated, which keeps the recovered transform interpretable for downstream
quality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.feature import match_template
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale


def _normalised_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.astype(np.float32, copy=False).ravel()
    b_flat = b.astype(np.float32, copy=False).ravel()
    a_flat -= a_flat.mean()
    b_flat -= b_flat.mean()
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _to_sitk_image(volume: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    img = sitk.GetImageFromArray(volume.astype(np.float32, copy=False))
    img.SetSpacing(spacing)
    return img


def _shrink_if_needed(img: sitk.Image, max_dim: int = 256) -> sitk.Image:
    size = img.GetSize()
    scale = max(size) / max_dim
    if scale <= 1.0:
        return img
    new_size = [int(round(s / scale)) for s in size]
    return sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        [sp * scale for sp in img.GetSpacing()],
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )


def _zscore(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - mean) / std


def _smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image.astype(np.float32, copy=False)
    return gaussian(image, sigma=sigma, preserve_range=True).astype(np.float32, copy=False)


def _prepare_anatomy_slices(anatomy: np.ndarray, sigma: float) -> np.ndarray:
    processed = np.empty_like(anatomy, dtype=np.float32)
    for z in range(anatomy.shape[0]):
        processed[z] = _zscore(_smooth(anatomy[z], sigma))
    return processed


def _prepare_templates(
    plane_smoothed: np.ndarray,
    plane_original: np.ndarray,
    scales: Sequence[float],
) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    templates = []
    for scale in scales:
        if scale <= 0.0:
            continue
        templ_smoothed = rescale(
            plane_smoothed,
            scale,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32, copy=False)
        if templ_smoothed.shape[0] < 4 or templ_smoothed.shape[1] < 4:
            continue
        templ_norm = _zscore(templ_smoothed)
        if np.allclose(templ_norm, 0.0):
            continue
        templ_orig = rescale(
            plane_original,
            scale,
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32, copy=False)
        templates.append((float(scale), templ_norm, templ_orig))
    return templates


def _match_template(
    fixed: np.ndarray,
    templ: np.ndarray,
    *,
    allow_subpixel: bool,
) -> Tuple[float, float, float] | None:
    """Return (y, x, score) in fixed coordinates or None if template does not fit."""
    if templ.shape[0] > fixed.shape[0] or templ.shape[1] > fixed.shape[1]:
        return None
    response = match_template(fixed, templ, pad_input=False)
    if response.size == 0:
        return None
    ij = np.unravel_index(int(np.argmax(response)), response.shape)
    y = float(ij[0])
    x = float(ij[1])
    score = float(response[ij])
    if allow_subpixel:
        y, x, score = _refine_subpixel_xy(fixed, templ, y, x)
    return y, x, score


def _refine_subpixel_xy(fixed: np.ndarray, templ: np.ndarray, y: float, x: float) -> Tuple[float, float, float]:
    """Subpixel refinement using phase correlation around (y, x)."""
    th, tw = templ.shape
    H, W = fixed.shape
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = max(0, y0)
    x1 = max(0, x0)
    y2 = min(H, y0 + th)
    x2 = min(W, x0 + tw)

    crop = fixed[y1:y2, x1:x2]
    if crop.shape != templ.shape:
        pad_y = th - crop.shape[0]
        pad_x = tw - crop.shape[1]
        crop = np.pad(crop, ((0, pad_y), (0, pad_x)), mode="edge")

    shift, _, _ = phase_cross_correlation(crop, templ, upsample_factor=10)
    y_ref = float(y1) - float(shift[0])
    x_ref = float(x1) - float(shift[1])

    yr = int(np.floor(y_ref))
    xr = int(np.floor(x_ref))
    yr1 = max(0, yr)
    xr1 = max(0, xr)
    yr2 = min(H, yr + th)
    xr2 = min(W, xr + tw)
    crop2 = fixed[yr1:yr2, xr1:xr2]
    if crop2.shape != templ.shape:
        pad_y = th - crop2.shape[0]
        pad_x = tw - crop2.shape[1]
        crop2 = np.pad(crop2, ((0, pad_y), (0, pad_x)), mode="edge")

    score = _normalised_cross_correlation(crop2, templ)
    return y_ref, x_ref, score


def _compose_warped_volume(
    anatomy_shape: Tuple[int, int, int],
    template: np.ndarray,
    *,
    z_index: int,
    y_pos: float,
    x_pos: float,
    allow_subpixel: bool,
) -> np.ndarray:
    volume = np.zeros(anatomy_shape, dtype=np.float32)
    plane = template
    y_floor = int(np.floor(y_pos))
    x_floor = int(np.floor(x_pos))
    y_frac = y_pos - y_floor
    x_frac = x_pos - x_floor

    if allow_subpixel and (abs(y_frac) > 1e-3 or abs(x_frac) > 1e-3):
        plane = ndi.shift(plane, shift=(-y_frac, -x_frac), order=1, mode="nearest")

    dest_y = y_floor
    dest_x = x_floor
    src_y = 0
    src_x = 0
    if dest_y < 0:
        src_y = -dest_y
        dest_y = 0
    if dest_x < 0:
        src_x = -dest_x
        dest_x = 0

    height = min(plane.shape[0] - src_y, anatomy_shape[1] - dest_y)
    width = min(plane.shape[1] - src_x, anatomy_shape[2] - dest_x)
    if height <= 0 or width <= 0:
        return volume

    z = int(np.clip(z_index, 0, anatomy_shape[0] - 1))
    volume[z, dest_y : dest_y + height, dest_x : dest_x + width] = plane[src_y : src_y + height, src_x : src_x + width]
    return volume


def _make_affine(scale: float, x: float, y: float, z: int) -> sitk.Transform:
    tx = sitk.AffineTransform(3)
    tx.SetMatrix([scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0])
    tx.SetTranslation((float(x), float(y), float(z)))
    return tx


@dataclass
class PlaneRegistrationResult:
    plane_index: int
    best_z: int
    ncc: float
    success: bool
    message: str
    transform: sitk.Transform
    warped_volume: np.ndarray


def register_planes_pass1(
    *,
    anatomy_stack: np.ndarray,
    functional_planes: Iterable[np.ndarray],
    plane_indices: Iterable[int],
    downscale_if_needed: bool = False,
    scale_range: Tuple[float, float] = (0.5, 1.0),
    n_scales: int = 12,
    z_stride_coarse: int = 5,
    z_refine_radius: int = 4,
    gaussian_sigma: float = 0.5,
    do_subpixel: bool = False,
) -> List[PlaneRegistrationResult]:
    anatomy_img = _to_sitk_image(anatomy_stack)
    if downscale_if_needed:
        anatomy_img = _shrink_if_needed(anatomy_img)
    anatomy_array = sitk.GetArrayFromImage(anatomy_img)
    anatomy_norm = _prepare_anatomy_slices(anatomy_array, gaussian_sigma)

    min_scale, max_scale = scale_range
    if min_scale <= 0.0 or max_scale <= 0.0:
        raise ValueError("scale_range must contain positive values")
    if max_scale < min_scale:
        raise ValueError("scale_range upper bound must be >= lower bound")
    if n_scales < 1:
        raise ValueError("n_scales must be >= 1")

    scales = np.linspace(min_scale, max_scale, n_scales, dtype=np.float32)
    z_count = anatomy_array.shape[0]
    coarse_z_candidates = sorted(set(range(0, z_count, max(1, z_stride_coarse))) | {0, z_count - 1})

    results: List[PlaneRegistrationResult] = []

    for plane_idx, plane in zip(plane_indices, functional_planes):
        plane_original = plane.astype(np.float32, copy=False)
        plane_smoothed = _smooth(plane_original, gaussian_sigma)
        templates = _prepare_templates(plane_smoothed, plane_original, scales)

        if not templates:
            results.append(
                PlaneRegistrationResult(
                    plane_index=plane_idx,
                    best_z=-1,
                    ncc=0.0,
                    success=False,
                    message="registration failed: no valid scales (template too small)",
                    transform=sitk.Transform(),
                    warped_volume=np.zeros_like(anatomy_array, dtype=np.float32),
                )
            )
            continue

        best = None
        for tpl_idx, (scale, templ_norm, _) in enumerate(templates):
            for z in coarse_z_candidates:
                match = _match_template(anatomy_norm[z], templ_norm, allow_subpixel=False)
                if match is None:
                    continue
                y, x, score = match
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "scale": scale,
                        "z": z,
                        "y": y,
                        "x": x,
                        "template": tpl_idx,
                    }

        if best is None:
            results.append(
                PlaneRegistrationResult(
                    plane_index=plane_idx,
                    best_z=-1,
                    ncc=0.0,
                    success=False,
                    message="registration failed: template never overlapped anatomy slice",
                    transform=sitk.Transform(),
                    warped_volume=np.zeros_like(anatomy_array, dtype=np.float32),
                )
            )
            continue

        refine_z_min = max(0, best["z"] - z_refine_radius)
        refine_z_max = min(z_count - 1, best["z"] + z_refine_radius)
        refine_z_candidates = range(refine_z_min, refine_z_max + 1)

        for tpl_idx, (scale, templ_norm, _) in enumerate(templates):
            for z in refine_z_candidates:
                match = _match_template(anatomy_norm[z], templ_norm, allow_subpixel=do_subpixel)
                if match is None:
                    continue
                y, x, score = match
                if score > best["score"]:
                    best.update(
                        {
                            "score": score,
                            "scale": scale,
                            "z": z,
                            "y": y,
                            "x": x,
                            "template": tpl_idx,
                        }
                    )

        tpl_scale, _, tpl_original = templates[best["template"]]
        warped_volume = _compose_warped_volume(
            anatomy_shape=anatomy_array.shape,
            template=tpl_original,
            z_index=best["z"],
            y_pos=best["y"],
            x_pos=best["x"],
            allow_subpixel=do_subpixel,
        )

        transform = _make_affine(tpl_scale, best["x"], best["y"], best["z"])
        message = f"scale={tpl_scale:.3f}, shift_yx=({best['y']:.2f}, {best['x']:.2f}), z={best['z']}"

        results.append(
            PlaneRegistrationResult(
                plane_index=plane_idx,
                best_z=int(best["z"]),
                ncc=float(best["score"]),
                success=True,
                message=message,
                transform=transform,
                warped_volume=warped_volume,
            )
        )

    return results
