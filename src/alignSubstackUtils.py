# utilities

# Setup & imports
import sys, math, warnings, json
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
_HAS_WIDGETS = True

import skimage
from skimage.feature import match_template
from skimage.transform import rescale, resize, rotate
from skimage.registration import phase_cross_correlation
from skimage.util import img_as_float32
from skimage.filters import gaussian

from scipy.ndimage import uniform_filter1d


import pandas as pd

from pathlib import Path
import numpy as np
import tifffile as tiff



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
    Uses phase correlation for subpixel shift between templ and the cropped region.
    Returns refined (y, x) in fixed coords and an updated NCC score for the refined crop.
    """
    y0, x0 = yx_initial
    th, tw = templ.shape
    # Crop region around initial match; clamp to image bounds
    y1 = max(0, y0)
    x1 = max(0, x0)
    y2 = min(fixed_plane.shape[0], y0 + th)
    x2 = min(fixed_plane.shape[1], x0 + tw)
    crop = fixed_plane[y1:y2, x1:x2]

    # If crop smaller than template due to edges, pad the crop
    if crop.shape != templ.shape:
        pad_y = max(0, templ.shape[0] - crop.shape[0])
        pad_x = max(0, templ.shape[1] - crop.shape[1])
        crop = np.pad(crop, ((0, pad_y), (0, pad_x)), mode='edge')

    # Phase correlation expects same-size arrays
    shift, error, phasediff = phase_cross_correlation(crop, templ, upsample_factor=10)
    dy, dx = shift
    # Convert to fixed coordinates
    y_ref = y1 + dy
    x_ref = x1 + dx

    # Recompute NCC score at refined location (rounded)
    yr = int(round(y_ref))
    xr = int(round(x_ref))
    yr2 = min(fixed_plane.shape[0], yr + th)
    xr2 = min(fixed_plane.shape[1], xr + tw)
    crop2 = fixed_plane[yr:yr2, xr:xr2]
    if crop2.shape != templ.shape:
        pad_y = max(0, templ.shape[0] - crop2.shape[0])
        pad_x = max(0, templ.shape[1] - crop2.shape[1])
        crop2 = np.pad(crop2, ((0, pad_y), (0, pad_x)), mode='edge')

    # NCC score
    num = np.sum((crop2 - crop2.mean()) * (templ - templ.mean()))
    den = np.sqrt(np.sum((crop2 - crop2.mean())**2) * np.sum((templ - templ.mean())**2)) + 1e-9
    score = float(num / den)
    return float(y_ref), float(x_ref), score


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
            y, x, score = yzxp
            if score > best["ncc"]:
                best.update({"z_index": z, "y": float(y), "x": float(x), "scale": float(s), "ncc": float(score), "pyr_level": 0})

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
                yzxp = template_match_best_xy(fL[z], templ)
                if yzxp is None:
                    continue
                # global coarse guess at this level:
                y_g, x_g, _ = yzxp

                # seed from previous level (upscaled by pyramid_downscale)
                seed_y = best["y"] * pyramid_downscale
                seed_x = best["x"] * pyramid_downscale

                # search in a local window around the seed; radius can shrink with levels
                win_radius = max(6, int(32 / (pyramid_downscale ** (lvl-1))))  # e.g., 32 -> 21 -> 14 -> 9
                y_w, x_w, score_w = match_in_window(fL[z], templ, (seed_y, seed_x), win_radius)

                # take the better of global vs windowed
                if score_w > _:
                    y, x, score = y_w, x_w, score_w
                else:
                    y, x, score = y_g, x_g, _
                if score > improved["ncc"]:
                    improved.update({"z_index": z, "y": float(y), "x": float(x), "scale": float(s), "ncc": float(score), "pyr_level": lvl})

        best = improved
        if verbose:
            print(f"[Refine L{lvl}] z={best['z_index']}, yx=({best['y']:.1f},{best['x']:.1f}), s={best['scale']:.3f}, ncc={best['ncc']:.4f}")

    # ---- Final subpixel XY refinement on full-res level
    z_fin = int(best["z_index"])
    s_fin = float(best["scale"])
    templ_full = mov_norm
    if not math.isclose(s_fin, 1.0, rel_tol=1e-6):
        new_y = max(8, int(round(mov_norm.shape[0] * s_fin)))
        new_x = max(8, int(round(mov_norm.shape[1] * s_fin)))
        templ_full = resize(mov_norm, (new_y, new_x), anti_aliasing=True, preserve_range=True)
        templ_full = normalize_image(templ_full)

    y0, x0 = int(round(best["y"])), int(round(best["x"]))
    y_ref, x_ref, ncc_ref = refine_subpixel_xy(fixed_norm[z_fin], templ_full, (y0, x0))

    # Compose result
    result = {
        "z_index": int(z_fin),
        "y_px": float(y_ref),
        "x_px": float(x_ref),
        "scale_moving_to_fixed": s_fin,
        "ncc_score": float(ncc_ref),
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
        )
        res.update({
            "moving_plane": zi,
            "z_um": res["z_index"] * float(fixed_z_spacing_um),
        })
        results.append(res)

    df = pd.DataFrame(results, columns=["moving_plane","z_index","z_um","y_px","x_px","scale_moving_to_fixed","ncc_score"])
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

    z = int(row["z_index"])
    y = float(row["y_px"])
    x = float(row["x_px"])
    s = float(row["scale_moving_to_fixed"])

    fixed = fixed_stack[z].astype(np.float32)
    moving = moving_stack[moving_index].astype(np.float32)

    # --- helpers (same as before) ---
    def _safe_crop(img, y, x, h, w):
        y, x = int(np.floor(y)), int(np.floor(x))
        y2, x2 = y + int(h), x + int(w)
        y1c, x1c = max(0, y), max(0, x)
        y2c, x2c = min(img.shape[0], y2), min(img.shape[1], x2)
        crop = img[y1c:y2c, x1c:x2c]
        pad_y_top = y1c - y
        pad_x_left = x1c - x
        pad_y_bot = (y + h) - y2c
        pad_x_right = (x + w) - x2c
        if any(v > 0 for v in (pad_y_top, pad_x_left, pad_y_bot, pad_x_right)):
            crop = np.pad(crop, ((pad_y_top, pad_y_bot), (pad_x_left, pad_x_right)), mode="edge")
        return crop

    def _rescale(img, scale):
        out_h = max(1, int(round(img.shape[0] * scale)))
        out_w = max(1, int(round(img.shape[1] * scale)))
        return resize(img, (out_h, out_w), anti_aliasing=True, preserve_range=True)

    def norm(im):
        im = im.astype(np.float32)
        p1, p99 = np.percentile(im, (1, 99))
        if p99 <= p1:
            p1, p99 = im.min(), max(im.max(), im.min()+1e-3)
        return np.clip((im - p1) / (p99 - p1 + 1e-6), 0, 1)

    # Scale moving to the estimated scale
    moving_scaled = _rescale(moving, s)
    h, w = moving_scaled.shape

    # Crop the matched region from the fixed slice (same size as moving_scaled)
    matched_crop = _safe_crop(fixed, y, x, h, w)

    # Normalize for display
    fixed_n = norm(fixed)
    moving_scaled_n = norm(moving_scaled)
    matched_crop_n = norm(matched_crop)

    # --- Plot
    fig = plt.figure(figsize=(12, 5))

    # Left: overlay in RGB (fixed stays the reference canvas)
    ax1 = fig.add_subplot(1, 2, 1)

    H, W = fixed_n.shape
    fixed_rgb = np.stack([fixed_n, fixed_n, fixed_n], axis=-1)

    # 1) Draw fixed with explicit extent
    ax1.imshow(fixed_rgb, interpolation="nearest", extent=[0, W, H, 0])

    # 2) Freeze axes to fixed bounds so the center never moves
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_autoscale_on(False)  # <- prevents any artist (overlay/rect) from changing limits

    # Overlay the scaled moving plane in green at (x,y) without shifting the axes
    extent = [x, x + w, y + h, y]  # left, right, bottom, top in FIXED coords
    ax1.imshow(moving_scaled_n, cmap="Greens", alpha=alpha_overlay,
            extent=extent, interpolation="nearest", clip_on=True, zorder=2)

    # Rectangle (purely visual; also won't change limits)
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=2, color="lime", zorder=3)
    ax1.add_patch(rect)

    ax1.set_title(f"Fixed z={z} with green overlay\n(y={y:.1f}, x={x:.1f}, scale={s:.3f})")
    ax1.set_axis_off()

    # Right: moving (scaled) vs matched crop (for inspection only)
    ax2 = fig.add_subplot(1, 2, 2)
    if matched_crop_n.shape != moving_scaled_n.shape:
        from skimage.transform import resize
        matched_crop_n = resize(matched_crop_n, moving_scaled_n.shape, preserve_range=True, anti_aliasing=True)
    ax2.imshow(np.concatenate([moving_scaled_n, matched_crop_n], axis=1),
            cmap="gray", interpolation="nearest")
    ax2.set_title("Left: moving (scaled) | Right: matched crop (fixed)")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()

def interactive_checker(fixed_stack, moving_stack, df_results):
    """Slider UI to browse matches per moving plane."""
    if not _HAS_WIDGETS:
        print("ipywidgets not available; showing the first plane statically. Install ipywidgets for a slider UI.")
        show_match(fixed_stack, moving_stack, df_results, moving_index=0)
        return

    slider = widgets.IntSlider(
        value=0, min=0, max=moving_stack.shape[0]-1, step=1, description="moving plane"
    )
    out = widgets.interactive_output(
        lambda moving_index: show_match(fixed_stack, moving_stack, df_results, moving_index=moving_index),
        {"moving_index": slider}
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

def _process_plane_stack(path, scale_xy, rot_deg=140.0, bg_value=0.0):
    """
    Load -> limit to 1000 -> MA(10) -> every 10th (100 frames)
    -> rotate 140° (no canvas growth) -> resize by scale_xy.
    Returns float32 (100, Ys, Xs).
    """
    ts = _load_timeseries_stack(path)                     # (T,512,512)
    ts = _temporal_filter_and_subsample(ts, 10, 1000, 10) # (100,512,512)

    T_out, H, W = ts.shape
    Ys = max(1, int(round(H * scale_xy)))
    Xs = max(1, int(round(W * scale_xy)))
    out = np.empty((T_out, Ys, Xs), dtype=np.float32)

    for t in range(T_out):
        fr = ts[t]
        fr_r = rotate(fr, angle=rot_deg, resize=False, mode='constant', cval=bg_value,
                      order=1, preserve_range=True)
        out[t] = resize(fr_r, (Ys, Xs), anti_aliasing=True,
                        preserve_range=True).astype(np.float32, copy=False)
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

    # 1) Process all planes (rotation + scale) and keep in RAM (list)
    processed = []  # list of dicts: {"idx": i, "z": z, "y": y, "x": x, "s": s, "data": (100,Ys,Xs)}
    for i, p in enumerate(funcStacks):
        if i not in meta:
            print(f"[skip] plane {i}: no registration row")
            continue
        z, y, x, s = meta[i]
        arr = _process_plane_stack(p, s, rot_deg=140.0)   # (100, Ys, Xs) float32
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

