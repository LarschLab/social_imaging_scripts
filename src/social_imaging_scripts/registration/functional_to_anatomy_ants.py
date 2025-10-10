"""Prototype functional→anatomy registration using coarse template matching.

Each functional plane is matched against all anatomy slices using
scale-constrained NCC.  The solver searches a discrete set of scales within
``scale_range`` and finds the best (z, y, x) placement via FFT-based template
matching.  Progress updates can be emitted by passing ``progress=print`` (or any
callable), and the search can short-circuit when ``early_stop_score`` is reached.

The intent is to mimic the previous hierarchical matcher: optimise for z index,
uniform in-plane scale, and x/y translation only.  No rotations or shearing are
estimated, which keeps the recovered transform interpretable for downstream
quality checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from skimage.feature import match_template
from skimage.filters import gaussian
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


def _match_template(fixed: np.ndarray, templ: np.ndarray) -> Tuple[float, float, float] | None:
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
    return y, x, score


def _compose_warped_volume(
    anatomy_shape: Tuple[int, int, int],
    template: np.ndarray,
    *,
    z_index: int,
    y_pos: float,
    x_pos: float,
) -> np.ndarray:
    volume = np.zeros(anatomy_shape, dtype=np.float32)
    plane = template
    y_floor = int(np.floor(y_pos))
    x_floor = int(np.floor(x_pos))
    y_frac = y_pos - y_floor
    x_frac = x_pos - x_floor

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
    progress: Optional[Callable[[str], None]] = None,
    early_stop_score: Optional[float] = None,
) -> List[PlaneRegistrationResult]:
    def _report(msg: str) -> None:
        if progress is not None:
            progress(msg)

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

        _report(
            f"[plane {plane_idx}] scales={len(templates)} coarse_z={len(coarse_z_candidates)}"
        )

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
        stop_score = early_stop_score
        coarse_evaluations = 0
        for tpl_idx, (scale, templ_norm, _) in enumerate(templates):
            for z in coarse_z_candidates:
                match = _match_template(anatomy_norm[z], templ_norm)
                if match is None:
                    continue
                y, x, score = match
                coarse_evaluations += 1
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "scale": scale,
                        "z": z,
                        "y": y,
                        "x": x,
                        "template": tpl_idx,
                    }
                if stop_score is not None and best["score"] >= stop_score:
                    break
            if stop_score is not None and best is not None and best["score"] >= stop_score:
                break

        _report(
            f"[plane {plane_idx}] coarse evals={coarse_evaluations}, "
            f"best z={best['z'] if best else 'n/a'} score={best['score'] if best else 'n/a'}"
        )

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

        refine_evaluations = 0
        for tpl_idx, (scale, templ_norm, _) in enumerate(templates):
            for z in refine_z_candidates:
                match = _match_template(anatomy_norm[z], templ_norm)
                if match is None:
                    continue
                y, x, score = match
                refine_evaluations += 1
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
                if stop_score is not None and best["score"] >= stop_score:
                    break
            if stop_score is not None and best["score"] >= stop_score:
                break

        _report(
            f"[plane {plane_idx}] refine evals={refine_evaluations}, "
            f"final z={best['z']} score={best['score']:.3f} scale={best['scale']:.3f}"
        )

        tpl_scale, _, tpl_original = templates[best["template"]]
        warped_volume = _compose_warped_volume(
            anatomy_shape=anatomy_array.shape,
            template=tpl_original,
            z_index=best["z"],
            y_pos=best["y"],
            x_pos=best["x"],
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
