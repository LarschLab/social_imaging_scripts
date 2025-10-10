# utilities

# Setup & imports
import sys, math, warnings, json, re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
_HAS_WIDGETS = True

import skimage
from skimage.feature import match_template
from skimage.transform import rescale, resize
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float32
from skimage.filters import gaussian
import nrrd
from scipy.ndimage import uniform_filter1d


import pandas as pd

from pathlib import Path
import numpy as np
import tifffile as tiff


def _fits_strictly_inside(fixed_shape, templ_shape, y, x, margin=1):
    """Return True if template at (y,x) fits strictly inside the fixed image with a pixel margin."""
    H, W = int(fixed_shape[0]), int(fixed_shape[1])
    th, tw = int(templ_shape[0]), int(templ_shape[1])
    return (y >= margin) and (x >= margin) and (y + th <= H - margin) and (x + tw <= W - margin)

def _clamp_inside(fixed_shape, templ_shape, y, x, margin=1):
    """Clamp (y,x) so that the template fits strictly inside with the given margin."""
    H, W = int(fixed_shape[0]), int(fixed_shape[1])
    th, tw = int(templ_shape[0]), int(templ_shape[1])
    y_min, y_max = margin, max(margin, H - margin - th)
    x_min, x_max = margin, max(margin, W - margin - tw)
    y_cl = float(np.clip(y, y_min, y_max))
    x_cl = float(np.clip(x, x_min, x_max))
    return y_cl, x_cl

def _ncc_at(fixed_plane, templ, y, x):
    """Compute NCC of templ placed at (y,x) using a safe crop (edge-padded if needed)."""
    H, W = fixed_plane.shape
    th, tw = templ.shape
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1 = max(0, y0); x1 = max(0, x0)
    y2 = min(H, y0 + th); x2 = min(W, x0 + tw)
    if y2 <= y1 or x2 <= x1:
        return -1.0
    crop = fixed_plane[y1:y2, x1:x2]
    # edge-pad to template size if partial overlap
    pad_y = th - crop.shape[0]
    pad_x = tw - crop.shape[1]
    if pad_y > 0 or pad_x > 0:
        crop = np.pad(crop, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")
    c = crop.astype(np.float32); t = templ.astype(np.float32)
    num = np.sum((c - c.mean()) * (t - t.mean()))
    den = np.sqrt(np.sum((c - c.mean())**2) * np.sum((t - t.mean())**2)) + 1e-9
    return float(num / den)


def normalize_image(im, eps=1e-6):
    """Z-score normalize 2D image; robust to outliers."""
    im = img_as_float32(im)
    m = np.nanmedian(im)
    s = np.nanmedian(np.abs(im - m)) * 1.4826 + eps  # robust std (MAD)
    out = (im - m) / s
    # Clip extreme tails to stabilize correlation
    return np.clip(out, -6, 6)

def build_pyramid(img, downscale=2, min_size=128):
    """Gaussian pyramid (list: coarse->fine) for a single 2D image."""
    levels = []
    current = img
    while min(current.shape) >= min_size:
        levels.append(current)
        if min(current.shape) // downscale < min_size:
            break
        current = gaussian(rescale(current, 1.0/downscale, anti_aliasing=True, preserve_range=True), sigma=1)
    return levels[::-1]  # return coarse->fine

def build_stack_pyramid(stack, downscale=2, min_size=128):
    """Pyramid for a 3D stack applied slice-wise (returns list of stacks)."""
    # To save memory/time, we downscale each plane independently.
    pyramids = []
    current = stack
    while min(current.shape[1:]) >= min_size:
        pyramids.append(current)
        # Prepare next scale if still large enough
        next_y = int(current.shape[1] / downscale)
        next_x = int(current.shape[2] / downscale)
        if min(next_y, next_x) < min_size:
            break
        # Downscale slice-wise
        ds = np.empty((current.shape[0], next_y, next_x), dtype=np.float32)
        for z in range(current.shape[0]):
            ds[z] = gaussian(resize(current[z], (next_y, next_x), anti_aliasing=True, preserve_range=True), sigma=1)
        current = ds
    return pyramids[::-1]  # coarse->fine

def template_match_best_xy(fixed_plane, templ):
    """
    Normalized cross-correlation (FFT-based). Returns best y, x in fixed image,
    score in [-1, 1]. templ must be smaller than fixed_plane.
    """
    if templ.shape[0] > fixed_plane.shape[0] or templ.shape[1] > fixed_plane.shape[1]:
        return None
    # skimage.match_template expects image >= template; returns response map
    resp = match_template(fixed_plane, templ, pad_input=False)  # pad to allow edges
    ij = np.unravel_index(np.argmax(resp), resp.shape)
    peak = float(resp[ij])
    y, x = int(ij[0]), int(ij[1])
    return y, x, peak

def refine_subpixel_xy(fixed_plane, templ, yx_initial, pad=16):
    """
    Subpixel refine around initial (y, x) by cropping fixed region the same size as templ.
    Returns refined (y, x) in fixed coords and an updated NCC score.
    """
    y0, x0 = yx_initial
    th, tw = templ.shape

    # Anchor the crop on integer pixel coordinates (top-left). This is crucial.
    y1 = max(0, int(np.floor(y0)))
    x1 = max(0, int(np.floor(x0)))
    y2 = min(fixed_plane.shape[0], y1 + th)
    x2 = min(fixed_plane.shape[1], x1 + tw)
    crop = fixed_plane[y1:y2, x1:x2]

    # If crop smaller than template due to edges, pad the crop
    if crop.shape != templ.shape:
        pad_y = max(0, th - crop.shape[0])
        pad_x = max(0, tw - crop.shape[1])
        crop = np.pad(crop, ((0, pad_y), (0, pad_x)), mode='edge')

    # phase_cross_correlation(reference=crop, moving=templ) -> shift to apply to MOVING
    dy, dx = phase_cross_correlation(crop, templ, upsample_factor=10)[0]

    # Convert to fixed coordinates (top-left of template in fixed coords)
    # KEY: subtract the shift returned by phase correlation
    y_ref = float(y1) - float(dy)
    x_ref = float(x1) - float(dx)

    # Recompute NCC at refined (rounded) position for a stable score
    yr = int(round(y_ref)); xr = int(round(x_ref))
    yr2 = min(fixed_plane.shape[0], yr + th)
    xr2 = min(fixed_plane.shape[1], xr + tw)
    crop2 = fixed_plane[yr:yr2, xr:xr2]
    if crop2.shape != templ.shape:
        pad_y = max(0, th - crop2.shape[0])
        pad_x = max(0, tw - crop2.shape[1])
        crop2 = np.pad(crop2, ((0, pad_y), (0, pad_x)), mode='edge')

    c = crop2.astype(np.float32); t = templ.astype(np.float32)
    num = np.sum((c - c.mean()) * (t - t.mean()))
    den = np.sqrt(np.sum((c - c.mean())**2) * np.sum((t - t.mean())**2)) + 1e-9
    score = float(num / den)

    return y_ref, x_ref, score


# ### Core registration (coarse-to-fine search over z and scale)

def match_in_window(fixed_plane, templ, yx_hint, win_radius):
    """
    Search for best (y,x) within +/- win_radius of yx_hint using a cropped ROI.
    Returns y, x, score in full-image coordinates.
    """
    yh, xh = int(round(yx_hint[0])), int(round(yx_hint[1]))
    th, tw = templ.shape

    y1 = max(0, yh - win_radius)
    x1 = max(0, xh - win_radius)
    y2 = min(fixed_plane.shape[0], yh + win_radius + th)
    x2 = min(fixed_plane.shape[1], xh + win_radius + tw)

    roi = fixed_plane[y1:y2, x1:x2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return yh, xh, -np.inf  # nothing to search

    # pad_input=False => response indexes are top-left within ROI
    resp = match_template(roi, templ, pad_input=False)
    ij = np.unravel_index(np.argmax(resp), resp.shape)
    peak = float(resp[ij])
    yy = int(ij[0]) + y1
    xx = int(ij[1]) + x1
    return yy, xx, peak

def _assert_strictly_inside(fixed_plane_shape, templ_shape, y, x, margin=1, ctx=""):
    """
    Ensure the template placed at (y,x) fits strictly inside fixed_plane with a pixel margin on all sides.
    fixed_plane_shape: (Y, X)
    templ_shape: (th, tw)
    y, x: floats in fixed coords (top-left of template)
    margin: >=1 requires at least that many pixels to the border
    Raises RuntimeError if the placement is invalid.
    """
    H, W = int(fixed_plane_shape[0]), int(fixed_plane_shape[1])
    th, tw = int(templ_shape[0]), int(templ_shape[1])

    if th <= 0 or tw <= 0:
        raise RuntimeError(f"{ctx} invalid template size th={th}, tw={tw}")

    # template must fit strictly inside with margin
    if not (
        (y >= margin) and
        (x >= margin) and
        (y + th <= H - margin) and
        (x + tw <= W - margin)
    ):
        raise RuntimeError(
            f"{ctx} registration out-of-bounds (or touching edge): "
            f"y={y:.2f}, x={x:.2f}, th={th}, tw={tw}, fixed(H,W)=({H},{W}), margin={margin}"
        )


def register_moving_plane_to_fixed_stack(
    fixed_stack,
    moving_plane,
    scale_range=(0.3, 1.2),
    n_scales=10,
    z_stride_coarse=4,
    z_refine_radius=3,
    pyramid_downscale=2,
    pyramid_min_size=160,
    gaussian_sigma=0.5,
    verbose=False,
    do_subpixel=False,   # <<< NEW: default off (skip subpixel)
):
    """
    Find best (z, y, x, scale) in fixed_stack for a single moving_plane.
    Returns dict with z_index, y, x in fixed coords (pixels), scale, z_score, final_ncc.
    """
    # Pre-normalize
    fixed_norm = np.stack([normalize_image(z) for z in fixed_stack], axis=0).astype(np.float32)
    mov_norm = normalize_image(moving_plane)

    # Build pyramids (coarse->fine)
    fixed_pyr = build_stack_pyramid(fixed_norm, downscale=pyramid_downscale, min_size=pyramid_min_size)
    mov_pyr   = build_pyramid(mov_norm, downscale=pyramid_downscale, min_size=pyramid_min_size)

    # Prepare scale grid (global, refined later)
    smin, smax = scale_range
    coarse_scales = np.geomspace(max(1e-3, smin), smax, num=n_scales)

    best = {
        "z_index": 0, "y": 0.0, "x": 0.0, "scale": 1.0,
        "ncc": -np.inf, "pyr_level": None
    }

    # ---- Coarse search on coarsest pyramid level
    f0 = fixed_pyr[0]
    m0 = mov_pyr[0]
    z_candidates = list(range(0, f0.shape[0], max(1, z_stride_coarse)))
    for s in coarse_scales:
        templ = m0
        if not math.isclose(s, 1.0, rel_tol=1e-6):
            # Rescale moving plane; keep mode='reflect' edges to avoid artifacts
            new_y = max(8, int(round(m0.shape[0] * s)))
            new_x = max(8, int(round(m0.shape[1] * s)))
            templ = resize(m0, (new_y, new_x), anti_aliasing=True, preserve_range=True)
            templ = normalize_image(templ)
        # Skip impossible template sizes
        if min(templ.shape) < 8:
            continue

        for z in z_candidates:
            yzxp = template_match_best_xy(f0[z], templ)
            if yzxp is None:
                continue

            # Unpack the match result
            y, x, score = yzxp

            # Only accept in-bounds candidates
            if score > best["ncc"] and _fits_strictly_inside(f0[z].shape, templ.shape, y, x, margin=1):
                best.update({
                    "z_index": z,
                    "y": float(y),
                    "x": float(x),
                    "scale": float(s),
                    "ncc": float(score),
                    "pyr_level": 0
                })

    if verbose:
        print(f"[Coarse] best @ level 0 -> z={best['z_index']}, yx=({best['y']:.1f},{best['x']:.1f}), s={best['scale']:.3f}, ncc={best['ncc']:.4f}")

    # ---- Refinement over pyramid levels (coarse->fine)
    # For each finer level, refine around previous best z and scale with smaller steps
    for lvl in range(1, len(fixed_pyr)):
        fL = fixed_pyr[lvl]
        # Upscale the current best coordinates to the next level
        scale_factor = (pyramid_downscale ** (lvl))
        yL = best["y"] * pyramid_downscale
        xL = best["x"] * pyramid_downscale
        zL = min(fL.shape[0]-1, int(round(best["z_index"] * 1)))  # z doesn't change with in-plane pyramid
        s_center = best["scale"]

        # Local scale sweep
        s_win = max(0.05, (coarse_scales[-1] / coarse_scales[0]) ** (1 / len(fixed_pyr)) - 1)
        # Build a small symmetric set around s_center
        #s_candidates = np.unique(np.clip(np.array([s_center*(1+ds) for ds in (-2*s_win, -s_win, 0, s_win, 2*s_win)]), smin, smax))
        s_step = max(0.02, 0.5 * (smax - smin) / max(3, len(fixed_pyr)))  # ~2–5%
        s_candidates = np.array([s_center - 2*s_step, s_center - s_step, s_center,
                                s_center + s_step, s_center + 2*s_step])
        s_candidates = np.unique(np.clip(s_candidates, smin, smax))
        # Local z neighborhood
        z_candidates = list(range(max(0, zL - z_refine_radius), min(fL.shape[0], zL + z_refine_radius + 1)))

        # Prepare moving template at this pyramid level
        # Determine the pyramid level factor to bring original moving to current level
        mL = mov_pyr[lvl if lvl < len(mov_pyr) else -1]

        improved = best.copy()
        for s in s_candidates:
            templ = mL
            if not math.isclose(s, 1.0, rel_tol=1e-6):
                new_y = max(8, int(round(mL.shape[0] * s)))
                new_x = max(8, int(round(mL.shape[1] * s)))
                templ = resize(mL, (new_y, new_x), anti_aliasing=True, preserve_range=True)
                templ = normalize_image(templ)
            if min(templ.shape) < 8:
                continue

            for z in z_candidates:

                yzxp = template_match_best_xy(fL[z], templ)
                if yzxp is None:
                    continue

                # Global coarse candidate at this level
                y_g, x_g, score_g = yzxp

                # Seed from previous level (upscaled by pyramid_downscale)
                seed_y = best["y"] * pyramid_downscale
                seed_x = best["x"] * pyramid_downscale

                # Local windowed candidate around the seed
                #win_radius = max(6, int(32 / (pyramid_downscale ** (lvl - 1))))
                win_radius = max(10, int(40 / (pyramid_downscale ** (lvl-1))))
                y_w, x_w, score_w = match_in_window(fL[z], templ, (seed_y, seed_x), win_radius)

                # Choose between global and windowed, but require strict in-bounds
                cands = []
                if _fits_strictly_inside(fL[z].shape, templ.shape, y_g, x_g, margin=1):
                    cands.append((y_g, x_g, score_g))
                if _fits_strictly_inside(fL[z].shape, templ.shape, y_w, x_w, margin=1):
                    cands.append((y_w, x_w, score_w))

                # If neither candidate is valid, try next z/scale without touching 'improved'
                if not cands:
                    continue

                # Pick the best valid candidate
                y_sel, x_sel, sc_sel = max(cands, key=lambda t: t[2])

                # --- Improvement A: tolerant update ---
                EPS_NCC = 0.02  # allow refinement even if NCC is slightly lower
                if sc_sel > improved["ncc"] - EPS_NCC:
                    improved.update({
                        "z_index": z,
                        "y": float(y_sel),
                        "x": float(x_sel),
                        "scale": float(s),
                        "ncc": float(sc_sel),
                        "pyr_level": lvl
                    })

        if improved["ncc"] == best["ncc"] and verbose:
            print(f"[Refine L{lvl}] no in-bounds improvement; keeping previous best")

        best = improved
        if verbose:
            print(f"[Refine L{lvl}] z={best['z_index']}, yx=({best['y']:.1f},{best['x']:.1f}), s={best['scale']:.3f}, ncc={best['ncc']:.4f}")

    # ---- Final subpixel XY refinement on full-res level (robust, in-bounds) ----
    # ---- Final placement & (optional) subpixel refinement ----
    # ---- Final placement at full-res + optional subpixel ----
    z_fin = int(best["z_index"])
    s_fin = float(best["scale"])

    # full-res template according to s_fin
    templ_full = mov_norm
    if not math.isclose(s_fin, 1.0, rel_tol=1e-6):
        new_y = max(8, int(round(mov_norm.shape[0] * s_fin)))
        new_x = max(8, int(round(mov_norm.shape[1] * s_fin)))
        templ_full = resize(mov_norm, (new_y, new_x), anti_aliasing=True, preserve_range=True)
        templ_full = normalize_image(templ_full)

    fixed_slice = fixed_norm[z_fin]
    th, tw = templ_full.shape

    # 1) map coords from the level they were found at -> full-res
    lvl_found = int(best.get("pyr_level", len(fixed_pyr)-1))      # 0 = coarsest
    levels_total = len(fixed_pyr)
    scale_to_full = (pyramid_downscale ** (levels_total - 1 - lvl_found))
    y_ref = float(best["y"]) * scale_to_full
    x_ref = float(best["x"]) * scale_to_full

    qc_notes = []

    # 2) ensure strictly inside at full-res
    if not _fits_strictly_inside(fixed_slice.shape, (th, tw), y_ref, x_ref, margin=1):
        y_ref, x_ref = _clamp_inside(fixed_slice.shape, (th, tw), y_ref, x_ref, margin=1)
        qc_notes.append("clamped_final_int")

    # helper: exact NCC (no padding) at integer-rounded coords
    def _ncc_exact_full(fixed_plane, templ, y, x):
        H, W = fixed_plane.shape
        th, tw = templ.shape
        yr = int(round(y)); xr = int(round(x))
        if yr < 0 or xr < 0 or yr + th > H or xr + tw > W:
            return None
        c = fixed_plane[yr:yr+th, xr:xr+tw].astype(np.float32, copy=False)
        t = templ.astype(np.float32, copy=False)
        c0 = c - c.mean(); t0 = t - t.mean()
        den = np.sqrt((c0**2).sum() * (t0**2).sum()) + 1e-9
        return float((c0 * t0).sum() / den)

    # 3) integer placement NCC
    ncc_final = _ncc_exact_full(fixed_slice, templ_full, y_ref, x_ref)
    if ncc_final is None:
        ncc_final = _ncc_at(fixed_slice, templ_full, y_ref, x_ref)
        qc_notes.append("ncc_padded_fallback")

    # 4) OPTIONAL: subpixel starting from full-res (y_ref, x_ref)
    if do_subpixel:
        try:
            y_sp, x_sp, ncc_sp = refine_subpixel_xy(fixed_slice, templ_full, (y_ref, x_ref))
            # keep only if still strictly inside and improves exact NCC
            if _fits_strictly_inside(fixed_slice.shape, (th, tw), y_sp, x_sp, margin=1):
                ncc_exact_sp = _ncc_exact_full(fixed_slice, templ_full, y_sp, x_sp)
                if ncc_exact_sp is not None and ncc_exact_sp > ncc_final:
                    y_ref, x_ref = y_sp, x_sp
                    ncc_final = ncc_exact_sp
                else:
                    qc_notes.append("subpixel_rejected_no_gain")
            else:
                qc_notes.append("subpixel_rejected_oob")
        except Exception:
            qc_notes.append("subpixel_failed")

    if verbose:
        print(f"[Final] z={z_fin}, yx=({y_ref:.2f},{x_ref:.2f}), s={s_fin:.3f}, ncc={ncc_final:.4f}"
            + (" (int+subpixel)" if do_subpixel else " (int)"))

    result = {
        "z_index": int(z_fin),
        "y_px": float(y_ref),
        "x_px": float(x_ref),
        "scale_moving_to_fixed": s_fin,
        "ncc_score": float(ncc_final),
        "qc_flag": "|".join(qc_notes) if qc_notes else "",
    }
    return result






# ### Batch registration for the whole moving stack



# %%
def register_moving_stack(
    fixed_stack,
    moving_stack,
    fixed_z_spacing_um=1.0,
    scale_range=(0.3, 1.2),
    n_scales=12,
    z_stride_coarse=5,
    z_refine_radius=4,
    pyramid_downscale=2,
    pyramid_min_size=160,
    gaussian_sigma=0.5,
    verbose=False,
    animal_id=None,   # <-- NEW
    do_subpixel=False,   # <<< NEW: default off (skip subpixel)
):
    """
    Registers each plane of moving_stack to fixed_stack.
    Returns a pandas DataFrame with columns:
        moving_plane, z_index, z_um, y_px, x_px, scale_moving_to_fixed, ncc_score
    """
    assert fixed_stack.ndim == 3 and moving_stack.ndim == 3, "Both stacks must be 3D arrays (Z,Y,X)."
    # Ensure float32
    fixed = img_as_float32(fixed_stack)
    moving = img_as_float32(moving_stack)

    results = []
    for zi in range(moving.shape[0]):
        if verbose:
            print(f"\n=== Registering moving plane {zi}/{moving.shape[0]-1} ===")
        res = register_moving_plane_to_fixed_stack(
            fixed, moving[zi],
            scale_range=scale_range,
            n_scales=n_scales,
            z_stride_coarse=z_stride_coarse,
            z_refine_radius=z_refine_radius,
            pyramid_downscale=pyramid_downscale,
            pyramid_min_size=pyramid_min_size,
            gaussian_sigma=gaussian_sigma,
            verbose=verbose,
            do_subpixel=do_subpixel,  
        )
        res.update({
            "animal": animal_id,
            "moving_plane": zi,
            "z_um": res["z_index"] * float(fixed_z_spacing_um),
        })
        results.append(res)

    #df = pd.DataFrame(results, columns=["moving_plane","z_index","z_um","y_px","x_px","scale_moving_to_fixed","ncc_score"])
    df = pd.DataFrame(
        results,
        columns=["animal","moving_plane","z_index","z_um",
                 "y_px","x_px","scale_moving_to_fixed","ncc_score","qc_flag"]
    )
    
    return df


def show_match(fixed_stack, moving_stack, df_results, moving_index=0, alpha_overlay=0.35):
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize

    assert 0 <= moving_index < moving_stack.shape[0], "moving_index out of range"
    row = df_results[df_results["moving_plane"] == moving_index]
    if row.empty:
        raise ValueError("No results found for the given moving_index in df_results.")
    row = row.iloc[0]

    # fields
    z   = int(row["z_index"])
    y   = float(row["y_px"])
    x   = float(row["x_px"])
    s   = float(row["scale_moving_to_fixed"])
    animal = str(row["animal"]) if "animal" in row.index else "?"

    fixed  = fixed_stack[z].astype(np.float32)
    moving = moving_stack[moving_index].astype(np.float32)

    # helpers
    def _safe_crop(img, y, x, h, w):
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = y0 + int(h), x0 + int(w)
        H, W   = img.shape
        # clamp paste area
        ya, yb = max(0, y0), min(H, y1)
        xa, xb = max(0, x0), min(W, x1)
        if ya >= yb or xa >= xb:
            return np.zeros((int(h), int(w)), dtype=img.dtype)
        crop = img[ya:yb, xa:xb]
        # pad to (h,w) if partially outside
        pad_top  = ya - y0
        pad_left = xa - x0
        pad_bot  = (y0 + int(h)) - yb
        pad_right= (x0 + int(w)) - xb
        if any(v > 0 for v in (pad_top, pad_left, pad_bot, pad_right)):
            crop = np.pad(crop,
                          ((max(0,pad_top),  max(0,pad_bot)),
                           (max(0,pad_left), max(0,pad_right))),
                          mode="edge")
        return crop

    def _rescale(img, scale):
        out_h = max(1, int(round(img.shape[0] * scale)))
        out_w = max(1, int(round(img.shape[1] * scale)))
        return resize(img, (out_h, out_w), anti_aliasing=True, preserve_range=True)

    def _norm(im):
        im = im.astype(np.float32)
        p1, p99 = np.percentile(im, (1, 99))
        if p99 <= p1:
            p1, p99 = im.min(), max(im.max(), im.min()+1e-3)
        return np.clip((im - p1) / (p99 - p1 + 1e-6), 0, 1)

    # scale moving + prepare matched crop
    moving_scaled = _rescale(moving, s)
    h, w          = moving_scaled.shape
    fixed_n       = _norm(fixed)
    moving_n      = _norm(moving_scaled)
    matched_crop  = _safe_crop(fixed, y, x, h, w)
    matched_n     = _norm(matched_crop)

    # ---- Build LEFT overlay as an RGB image in numpy ----
    H, W = fixed_n.shape
    overlay = np.stack([fixed_n, fixed_n, fixed_n], axis=-1)  # RGB from fixed

    # alpha-blend the (green) moving onto overlay at (y,x)
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(H, y0 + h), min(W, x0 + w)
    # align source window inside moving_n
    sy1 = ya - y0; sx1 = xa - x0
    sy2 = sy1 + (yb - ya); sx2 = sx1 + (xb - xa)
    if yb > ya and xb > xa:
        # green channel blend
        dstG = overlay[ya:yb, xa:xb, 1]
        src  = moving_n[sy1:sy2, sx1:sx2]
        overlay[ya:yb, xa:xb, 1] = (1 - alpha_overlay) * dstG + alpha_overlay * src
        # rectangle (lime)
        rr_thick = 2
        # top/bottom
        y_top = slice(max(0, y0), min(H, y0 + rr_thick))
        y_bot = slice(max(0, y0 + h - rr_thick), min(H, y0 + h))
        x_all = slice(max(0, x0), min(W, x0 + w))
        overlay[y_top, x_all, :] = [0, 1, 0]
        overlay[y_bot, x_all, :] = [0, 1, 0]
        # left/right
        x_left  = slice(max(0, x0), min(W, x0 + rr_thick))
        x_right = slice(max(0, x0 + w - rr_thick), min(W, x0 + w))
        y_all   = slice(max(0, y0), min(H, y0 + h))
        overlay[y_all, x_left,  :] = [0, 1, 0]
        overlay[y_all, x_right, :] = [0, 1, 0]

    if matched_n.shape != moving_n.shape:
        matched_n = resize(matched_n, moving_n.shape, preserve_range=True, anti_aliasing=True)

    right_disp = np.concatenate([moving_n, matched_n], axis=1)

    # ---- Plot
    fig = plt.figure(figsize=(14, 6))

    # left
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(np.clip(overlay, 0, 1), interpolation="nearest")
    ax1.set_title("Fixed with moving overlay")
    ax1.set_axis_off()

    # right
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(right_disp, cmap="gray", interpolation="nearest")
    ax2.set_title("Left: moving (scaled) | Right: matched crop")
    ax2.set_axis_off()

    # info box (2 columns, 3-dec floats)
    info_items = []
    for col in df_results.columns:
        val = row[col]
        if isinstance(val, (float, np.floating)):
            info_items.append(f"{col}={val:.3f}")
        else:
            info_items.append(f"{col}={val}")
    mid = (len(info_items) + 1) // 2
    left_items, right_items = info_items[:mid], info_items[mid:]
    if len(right_items) < len(left_items):
        right_items += [""] * (len(left_items) - len(right_items))
    text_lines = [f"{a:<28} {b}" for a, b in zip(left_items, right_items)]
    text_str = f"{animal}, plane {moving_index}\n\n" + "\n".join(text_lines)

    ax1.text(0.02, 0.98, text_str,
             transform=ax1.transAxes, fontsize=8, va="top", ha="left",
             color="yellow", family="monospace",
             bbox=dict(facecolor="black", alpha=0.5, pad=4, edgecolor="none"))

    plt.tight_layout()
    plt.show()






def interactive_checker(fixed_stack, moving_stack, df_results):
    """Slider UI to browse matches per moving plane."""

    slider = widgets.IntSlider(
        value=0, min=0, max=moving_stack.shape[0]-1, step=1, description="moving plane"
    )
    out = widgets.interactive_output(
        lambda moving_index: show_match(fixed_stack, moving_stack, df_results, moving_index=moving_index),
        {"moving_index": slider},
    )
    display(widgets.HBox([slider]), out)

# Build hyperstack (T, Z, Y, X) from functional plane stacks using registration results


# ---------- helpers ----------

def _load_timeseries_stack(path):
    arr = tiff.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,Y,X) or (Y,X); got shape {arr.shape} for {path}")
    return arr.astype(np.float32, copy=False)

def _temporal_filter_and_subsample(ts, movavg=10, limit=1000, step=10):
    T = min(limit, ts.shape[0])
    ts = ts[:T]
    ts_f = uniform_filter1d(ts, size=movavg, axis=0, mode='reflect')  # centered MA
    return ts_f[::step]  # (100, Y, X)

def _process_plane_stack(path, scale_xy, bg_value=0.0):
    """Load -> limit to 1000 -> MA(10) -> every 10th (100 frames) -> resize by scale_xy."""
    ts = _load_timeseries_stack(path)                     # (T,512,512)
    ts = _temporal_filter_and_subsample(ts, 10, 1000, 10) # (100,512,512)

    T_out, H, W = ts.shape
    Ys = max(1, int(round(H * scale_xy)))
    Xs = max(1, int(round(W * scale_xy)))
    out = np.empty((T_out, Ys, Xs), dtype=np.float32)

    for t in range(T_out):
        fr = ts[t]
        out[t] = resize(fr, (Ys, Xs), anti_aliasing=True, preserve_range=True).astype(np.float32, copy=False)
    return out

def _safe_paste_max(dst2d, src2d, top, left):
    """Paste src2d into dst2d at (top,left) with clipping; merge via np.maximum."""
    H, W = dst2d.shape
    h, w = src2d.shape
    y1 = int(np.floor(top)); x1 = int(np.floor(left))
    y2 = y1 + h;             x2 = x1 + w
    dy1 = max(0, y1); dx1 = max(0, x1)
    dy2 = min(H, y2); dx2 = min(W, x2)
    if dy2 <= dy1 or dx2 <= dx1:
        return
    sy1 = dy1 - y1; sx1 = dx1 - x1
    sy2 = sy1 + (dy2 - dy1); sx2 = sx1 + (dx2 - dx1)
    dst2d[dy1:dy2, dx1:dx2] = np.maximum(dst2d[dy1:dy2, dx1:dx2],
                                         src2d[sy1:sy2, sx2 - (dx2 - dx1) + sx1:sx2] if False else src2d[sy1:sy2, sx1:sx2])
    # note: the odd slice in the 'if False' is a guard for accidental mistakes; current branch is correct.

def _scale_to_uint8(arr, vmin, vmax):
    eps = 1e-6
    out = (arr - vmin) * (255.0 / max(vmax - vmin, eps))
    return np.clip(out, 0, 255).astype(np.uint8, copy=False)

# ---------- main (simple, in-memory) ----------

def build_hyperstack_uint8_in_memory(
    funcStacks,            # list of paths, one per moving plane (same order as df_results.moving_plane)
    df_results,            # DataFrame: moving_plane, z_index, y_px, x_px, scale_moving_to_fixed
    fixed_stack_shape,     # (Zf, Yf, Xf) e.g., (216, 512, 512)
    use_percentiles=None,  # None -> global min/max; or (1,99) robust scaling
    save_path=None         # optional TIFF path to save the TZYX hyperstack
):
    Zf, Yf, Xf = fixed_stack_shape
    T_out = 100  # 1000 -> /10

    # Map plane -> (z,y,x,scale)
    meta = {}
    for _, r in df_results.iterrows():
        meta[int(r["moving_plane"])] = (
            int(r["z_index"]),
            float(r["y_px"]),
            float(r["x_px"]),
            float(r["scale_moving_to_fixed"])
        )

    # 1) Process all planes (scale to anatomy space) and keep in RAM (list)
    processed = []  # list of dicts: {"idx": i, "z": z, "y": y, "x": x, "s": s, "data": (100,Ys,Xs)}
    for i, p in enumerate(funcStacks):
        if i not in meta:
            print(f"[skip] plane {i}: no registration row")
            continue
        z, y, x, s = meta[i]
        arr = _process_plane_stack(p, s)   # (100, Ys, Xs) float32
        processed.append({"idx": i, "z": z, "y": y, "x": x, "s": s, "data": arr})
        print(f"[proc] plane {i}: -> z={z}, pos=({y:.1f},{x:.1f}), scale={s:.3f}, shape={arr.shape}")

    # 2) Global intensity scaling
    if use_percentiles is None:
        g_min = min(float(np.nanmin(p["data"])) for p in processed)
        g_max = max(float(np.nanmax(p["data"])) for p in processed)
    else:
        p_lo, p_hi = use_percentiles
        samples = []
        for p in processed:
            d = p["data"]
            # take a few million random pixels total for robust percentiles
            n_pick = min(2_000_000 // len(processed), d.size)
            idx = np.random.default_rng(123).integers(0, d.size, n_pick)
            samples.append(d.ravel()[idx])
        samples = np.concatenate(samples) if samples else np.array([0.0], dtype=np.float32)
        g_min = float(np.percentile(samples[samples>1.], p_lo))
        g_max = float(np.percentile(samples[samples>1.], p_hi))
    print(f"[scale] global intensity >1 -> {g_min:.6f} .. {g_max:.6f}")

    # 3) Allocate the uint8 hyperstack in RAM
    hyper = np.zeros((T_out, Zf, Yf, Xf), dtype=np.uint8)  # ~5.6 GB

    # 4) Paste each plane frame-by-frame (np.maximum)
    for P in processed:
        z, y, x = P["z"], P["y"], P["x"]
        dat = P["data"]  # (100, Ys, Xs) float32
        for t in range(T_out):
            frame8 = _scale_to_uint8(dat[t], g_min, g_max)
            _safe_paste_max(hyper[t, z], frame8, top=y, left=x)
        print(f"[merge] plane {P['idx']} pasted at z={z}")

    if save_path:
        tiff.imwrite(save_path, hyper, bigtiff=True, metadata={'axes': 'TZYX'})
        print(f"[saved] {save_path}")

    return hyper


def write_combined_log(base_folder, out_csv=None, out_full_csv=None):
    """Aggregate all per-animal CSVs into one summary log + full log."""
    base = Path(base_folder)
    per_animal_csvs = sorted(base.glob("*/02_reg/07_2pf-a/*_registration_results.csv"))

    if not per_animal_csvs:
        print("[WARN] no per-animal results found.")
        return None, None

    dfs = [pd.read_csv(csv_path) for csv_path in per_animal_csvs]
    big = pd.concat(dfs, ignore_index=True)

    agg = (big.groupby("animal")["ncc_score"]
             .agg(n_planes="count",
                  ncc_mean="mean",
                  ncc_median="median",
                  ncc_min="min",
                  ncc_max="max")
             .reset_index())

    def _qual(median_ncc):
        if median_ncc >= 0.85: return "excellent"
        if median_ncc >= 0.65: return "good"
        if median_ncc >= 0.40: return "ok"
        return "poor"

    agg["quality"] = agg["ncc_median"].apply(_qual)

    out_csv = out_csv or (base / "combined_registration_log.csv")
    agg.to_csv(out_csv, index=False)
    print(f"[SAVED] Combined summary: {out_csv}")

    out_full_csv = out_full_csv or (base / "combined_registration_full.csv")
    big.to_csv(out_full_csv, index=False)
    print(f"[SAVED] Combined full log: {out_full_csv}")

    return agg, big


def _read_stack_zyx(path: Path) -> np.ndarray:
    """
    Core reader that loads NRRD or TIFF volumes and returns an array shaped (Z, Y, X)
    preserving the on-disk bit depth.
    """
    path = Path(path)
    try:
        data, _ = nrrd.read(str(path))
    except (nrrd.NRRDError, UnicodeDecodeError):
        try:
            im = tiff.imread(str(path))
        except Exception as tif_err:
            raise RuntimeError(f"Failed to read anatomy stack {path}: {tif_err}") from tif_err
        im = np.asarray(im, order="C")
        if im.ndim == 2:
            im = im[np.newaxis, :, :]
        elif im.ndim != 3:
            raise ValueError(f"Unexpected TIFF shape {im.shape} for {path}")
    else:
        im = np.asarray(data, order="C")
        if im.ndim != 3:
            raise ValueError(f"Expected 3D NRRD; got shape {im.shape} for {path}")
        # NRRD exports often arrive as (X, Y, Z); align to (Z, Y, X).
        im = np.moveaxis(im, 2, 0)
        im = np.moveaxis(im, 1, 2)
    return im


def read_stack_float32(path: Path) -> np.ndarray:
    """
    Load a volumetric stack as float32 with shape (Z, Y, X) while preserving dynamic range.
    """
    im = _read_stack_zyx(path)
    if im.dtype != np.float32:
        im = im.astype(np.float32, copy=False)
    return im


def read_good_nrrd_uint8(path: Path) -> np.ndarray:
    """
    Legacy helper that returns the volume as uint8. Prefer :func:`read_stack_float32`
    for new code paths so that bit depth is preserved.
    """
    im = _read_stack_zyx(path)
    if im.dtype != np.uint8:
        im = im.astype(np.uint8, copy=False)
    return im


def get_spacing_origin_ZYX(nrrd_path: Path):
    """
    Return (spacing_zyx, origin_zyx) aligned to arrays produced by
    :func:`read_stack_float32` / :func:`read_good_nrrd_uint8`. Handles true NRRDs as
    well as TIFF files that were mislabeled with a .nrrd suffix (ImageJ export with
    embedded NRRD metadata).
    """
    path = Path(nrrd_path)

    def _spacing_origin_from_head(head):
        spacing = origin = None
        if head is None:
            return spacing, origin
        if 'spacing' in head and head['spacing'] is not None:
            sp = np.array(head['spacing'], dtype=float)
            if np.ndim(sp) == 0:
                spacing = np.array([sp, sp, sp], dtype=float)
            else:
                spacing = np.array(sp, dtype=float)
        elif 'spacings' in head and head['spacings'] is not None:
            spacing = np.array(head['spacings'], dtype=float)
        elif 'space directions' in head and head['space directions'] is not None:
            sd = np.array(head['space directions'], dtype=float)
            spacing = np.linalg.norm(sd, axis=1)
        if 'space origin' in head and head['space origin'] is not None:
            origin = np.array(head['space origin'], dtype=float)
        return spacing, origin

    def _spacing_origin_from_tiff(path: Path):
        spacing = origin = None
        try:
            with tiff.TiffFile(str(path)) as tf:
                page0 = tf.pages[0]
                ij_tag = page0.tags.get('IJMetadata')
                info_str = None
                if ij_tag is not None:
                    ij_val = ij_tag.value
                    if isinstance(ij_val, dict):
                        info_str = ij_val.get('Info') or ij_val.get('info')
                    elif isinstance(ij_val, (bytes, str)):
                        info_str = ij_val
                if isinstance(info_str, bytes):
                    info_str = info_str.decode('utf-8', 'ignore')
                if isinstance(info_str, str):
                    for line in info_str.splitlines():
                        line = line.strip()
                        if not line or ':' not in line:
                            continue
                        key, val = line.split(':', 1)
                        key = key.strip().lower()
                        val = val.strip()
                        if key == 'space directions':
                            vecs = []
                            for part in re.findall(r'\(([^)]*)\)', val):
                                vec = np.fromstring(part, sep=',')
                                if vec.size:
                                    vecs.append(vec)
                            if len(vecs) == 3:
                                spacing = np.linalg.norm(np.vstack(vecs), axis=1)
                        elif key == 'spacings':
                            arr = np.fromstring(val.replace(',', ' '), sep=' ')
                            if arr.size == 3:
                                spacing = arr
                        elif key == 'space origin':
                            arr = np.fromstring(val.strip('()'), sep=',')
                            if arr.size == 3:
                                origin = arr
                desc_tag = page0.tags.get('ImageDescription')
                desc = desc_tag.value if desc_tag is not None else None
                if isinstance(desc, bytes):
                    desc = desc.decode('utf-8', 'ignore')
                if isinstance(desc, str):
                    for line in desc.splitlines():
                        line = line.strip()
                        if line.lower().startswith('spacing='):
                            try:
                                z_spacing = float(line.split('=', 1)[1])
                            except ValueError:
                                z_spacing = None
                            if z_spacing is not None:
                                if spacing is None:
                                    spacing = np.array([1.0, 1.0, z_spacing], dtype=float)
                                else:
                                    spacing = np.array(spacing, dtype=float)
                                    if spacing.size >= 3:
                                        spacing[2] = z_spacing
                                    else:
                                        spacing = np.array([spacing[0], spacing[1], z_spacing], dtype=float)
                            break
        except Exception:
            pass
        return spacing, origin

    spacing_raw = origin_raw = None
    try:
        _, head = nrrd.read(str(path))
    except (nrrd.NRRDError, UnicodeDecodeError):
        head = None
    else:
        spacing_raw, origin_raw = _spacing_origin_from_head(head)

    if spacing_raw is None or origin_raw is None:
        spacing_alt, origin_alt = _spacing_origin_from_tiff(path)
        if spacing_raw is None and spacing_alt is not None:
            spacing_raw = spacing_alt
        if origin_raw is None and origin_alt is not None:
            origin_raw = origin_alt

    if spacing_raw is None:
        spacing_raw = np.array([1.0, 1.0, 1.0], dtype=float)
    if origin_raw is None or origin_raw.size != 3:
        origin_raw = np.array([0.0, 0.0, 0.0], dtype=float)

    spacing_zyx = np.array([spacing_raw[2], spacing_raw[0], spacing_raw[1]], dtype=float)
    origin_zyx  = np.array([origin_raw[2], origin_raw[1], origin_raw[0]], dtype=float)

    return spacing_zyx, origin_zyx


def indexZYX_to_physZYX(idx_zyx: np.ndarray, spacing_zyx, origin_zyx):
    """idx_zyx: (N,3) in (z,y,x) -> physical (z,y,x)."""
    return origin_zyx + idx_zyx * spacing_zyx

def physZYX_to_indexZYX(phys_zyx: np.ndarray, spacing_zyx, origin_zyx):
    """phys_zyx: (N,3) in (z,y,x) -> index (z,y,x)."""
    return (phys_zyx - origin_zyx) / spacing_zyx
