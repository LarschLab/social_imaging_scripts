"""Manual prematch GUI for confocal→anatomy using RAW confocal stacks.

This module extracts the GUI logic out of the notebook so it is easier to
maintain. It loads RAW confocal LSM data, computes MIPs, rescales to match
anatomy pixel size, and provides a simple overlay GUI with rotation/translation
sliders. Saving writes to the processing log under the manual prematch section.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile

try:  # Optional at import; only needed inside notebooks
    import ipywidgets as widgets  # type: ignore
    from IPython.display import display, clear_output  # type: ignore
except Exception:  # pragma: no cover
    widgets = None  # type: ignore
    display = None  # type: ignore
    clear_output = None  # type: ignore

from ..metadata.config import (
    ProjectConfig,
    load_project_config,
    resolve_output_path,
    resolve_raw_path,
)
from ..metadata.loader import load_animals
from ..metadata.models import AnatomySession, AnimalMetadata
from ..pipeline.processing_log import (
    AnimalProcessingLog,
    build_processing_log_path,
    load_processing_log,
    save_processing_log,
)


def _compute_applied_angle(raw_angle_deg: float, cfg: ProjectConfig) -> Tuple[float, float, bool]:
    """Return (applied_angle, offset_deg, offset_signed_flag).

    Applies cfg.confocal_preprocessing.gui_rotation_offset_deg and, if enabled,
    applies the offset with the sign of the raw angle. The result is normalised
    into (-180, 180].
    """

    off = float(getattr(cfg.confocal_preprocessing, "gui_rotation_offset_deg", 0.0))
    signed = bool(getattr(cfg.confocal_preprocessing, "gui_rotation_offset_signed", False))
    if signed:
        off = np.copysign(off, raw_angle_deg if abs(raw_angle_deg) > 1e-12 else 1.0)
    applied = raw_angle_deg + off
    applied = ((applied + 180.0) % 360.0) - 180.0
    if abs(applied - 180.0) < 1e-6:
        applied = -180.0
    return float(applied), float(off), bool(signed)


def _load_mip(stack_path: Path) -> np.ndarray:
    stack = tifffile.imread(str(stack_path))
    if stack.ndim == 2:
        return stack.astype(np.float32, copy=False)
    if stack.ndim == 3:
        return np.max(stack, axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Expected 2D or 3D stack, got {stack.ndim}D: {stack_path}")


def _normalize_image(img: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    vmin, vmax = np.percentile(img, [0.5, percentile])
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    return np.clip(img, 0.0, 1.0)


def _rescale_image(img: np.ndarray, scale_y: float, scale_x: float) -> np.ndarray:
    from scipy.ndimage import zoom  # type: ignore

    return zoom(img, (scale_y, scale_x), order=1)


def _apply_transform_2d(img: np.ndarray, x_shift: float, y_shift: float, rotation_deg: float) -> np.ndarray:
    from scipy.ndimage import rotate as nd_rotate, shift as nd_shift  # type: ignore

    out = img
    if abs(rotation_deg) > 1e-3:
        out = nd_rotate(out, rotation_deg, reshape=False, order=1, mode="constant", cval=0.0)
    if abs(x_shift) > 1e-3 or abs(y_shift) > 1e-3:
        out = nd_shift(out, shift=(y_shift, x_shift), order=1, mode="constant", cval=0.0)
    return out


def _create_overlay(fixed: np.ndarray, moving: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    f = _normalize_image(fixed)
    m = _normalize_image(moving)
    overlay = np.zeros((*f.shape, 3), dtype=np.float32)
    overlay[..., 0] = f * 0.8
    overlay[..., 1] = m * alpha + f * (1 - alpha) * 0.5
    overlay[..., 2] = f * 0.8
    return np.clip(overlay, 0.0, 1.0)


def _load_anatomy_mip(animal: AnimalMetadata, anatomy_session: AnatomySession, cfg: ProjectConfig) -> Tuple[np.ndarray, Tuple[float, float]]:
    root = resolve_output_path(animal.animal_id, cfg.anatomy_preprocessing.root_subdir, cfg=cfg)
    stack_path = root / cfg.anatomy_preprocessing.stack_filename_template.format(
        animal_id=animal.animal_id, session_id=anatomy_session.session_id
    )
    metadata_path = root / cfg.anatomy_preprocessing.metadata_filename_template.format(
        animal_id=animal.animal_id, session_id=anatomy_session.session_id
    )
    mip = _load_mip(stack_path)
    try:
        meta = Path(metadata_path).read_text(encoding="utf-8")
        import json

        data = json.loads(meta)
        px, py = data.get("pixel_size_xy_um", [1.0, 1.0])
        pixel_size = (float(py), float(px))  # (y, x)
    except Exception:
        pixel_size = (1.0, 1.0)
    return mip, pixel_size


def _load_raw_confocal_data(
    animal: AnimalMetadata, confocal_session: AnatomySession, cfg: ProjectConfig
) -> Tuple[np.ndarray, List[str], Tuple[float, float]]:
    """Load RAW confocal stack as (Z, C, Y, X) along with channel names and pixel size."""

    raw_rel = Path(str(confocal_session.session_data.raw_path))
    base = raw_rel
    rd = getattr(animal, "root_dir", None)
    if rd:
        base = Path(rd) / base
    raw_path = resolve_raw_path(base, cfg=cfg).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Confocal RAW not found: {raw_path}")

    with tifffile.TiffFile(str(raw_path)) as tf:
        series = tf.series[0]
        arr = series.asarray().astype(np.float32, copy=False)
        axes = getattr(series, "axes", "") or ""
        while arr.ndim > 4 and 1 in arr.shape[:-4]:
            arr = arr.reshape(arr.shape[-4:])
        if arr.ndim == 3:
            arr_zcyx = arr[:, None, :, :]
        elif arr.ndim == 4 and axes == "ZCYX":
            arr_zcyx = arr
        elif arr.ndim == 4 and axes == "CZYX":
            arr_zcyx = arr.transpose(1, 0, 2, 3)
        else:
            if arr.ndim == 4 and arr.shape[1] <= 16:
                arr_zcyx = arr
            elif arr.ndim == 4 and arr.shape[0] <= 16:
                arr_zcyx = arr.transpose(1, 0, 2, 3)
            else:
                raise ValueError(f"Unsupported RAW confocal shape/axes: {arr.shape} / {axes!r}")
        lsm = getattr(tf, "lsm_metadata", None) or {}
        vx = float(lsm.get("VoxelSizeX", 1.0)) * 1e6
        vy = float(lsm.get("VoxelSizeY", 1.0)) * 1e6
        pixel_size = (float(vy), float(vx))

    channels_meta = list(getattr(confocal_session.session_data, "channels", []) or [])
    channel_names: List[str] = []
    for idx in range(arr_zcyx.shape[1]):
        name = None
        if idx < len(channels_meta):
            name = getattr(channels_meta[idx], "name", None) or getattr(channels_meta[idx], "marker", None)
        if not name:
            name = f"channel{idx}"
        channel_names.append(str(name))

    return arr_zcyx, channel_names, pixel_size


def _save_prematch_to_log(
    *,
    animal: AnimalMetadata,
    confocal_session: AnatomySession,
    x_shift: float,
    y_shift: float,
    rotation_deg: float,
    flip_horizontal: bool,
    flip_z: bool,
    display_channel: str,
    cfg: ProjectConfig,
) -> None:
    log_path = build_processing_log_path(cfg.processing_log, animal.animal_id, base_dir=Path(cfg.output_base_dir))
    if log_path.exists():
        log = load_processing_log(log_path)
    else:
        log = AnimalProcessingLog(animal_id=animal.animal_id)
    stage = log.ensure_stage("confocal_to_anatomy_registration")
    manual = dict(stage.parameters.get("manual_prematch", {}))
    manual[confocal_session.session_id] = {
        "translation_x_px": float(x_shift),
        "translation_y_px": float(y_shift),
        "rotation_deg": float(rotation_deg),
        "flip_horizontal": bool(flip_horizontal),
        "flip_z": bool(flip_z),
        "display_channel": str(display_channel),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "method": "manual_gui",
    }
    stage.parameters["manual_prematch"] = manual
    save_processing_log(log, log_path)


class ManualConfocalPrematchGUI:
    """Interactive GUI widget for confocal→anatomy manual prematch."""

    def __init__(
        self,
        animal: AnimalMetadata,
        confocal_session: AnatomySession,
        anatomy_session: AnatomySession,
        cfg: ProjectConfig,
    ) -> None:
        if widgets is None:
            raise RuntimeError("ipywidgets not available; run inside a Jupyter environment")
        self.animal = animal
        self.confocal_session = confocal_session
        self.anatomy_session = anatomy_session
        self.cfg = cfg
        # Load images
        self.fixed_mip_base, self.fixed_pixel_size = _load_anatomy_mip(animal, anatomy_session, cfg)
        raw_stack, channel_names, moving_pixel_size = _load_raw_confocal_data(animal, confocal_session, cfg)
        self.raw_stack = raw_stack  # (Z, C, Y, X)
        self.channel_names = channel_names
        self.moving_pixel_size = moving_pixel_size
        self.scale_y = self.moving_pixel_size[0] / self.fixed_pixel_size[0]
        self.scale_x = self.moving_pixel_size[1] / self.fixed_pixel_size[1]
        self.moving_mip_base = np.zeros_like(self.fixed_mip_base, dtype=np.float32)
        self.fixed_mip = self.fixed_mip_base
        # Determine default channel index (reference channel if available)
        ref_name = str(getattr(cfg.confocal_to_anatomy_registration, "reference_channel_name", "gcamp")).lower()
        default_idx = 0
        for idx, name in enumerate(self.channel_names):
            if name.lower() == ref_name:
                default_idx = idx
                break
        self.channel_index = default_idx
        # Load existing prematch
        existing = self._get_existing_prematch()
        init_x = float(existing.get("translation_x_px", 0.0)) if existing else 0.0
        init_y = float(existing.get("translation_y_px", 0.0)) if existing else 0.0
        init_rot = float(existing.get("rotation_deg", 0.0)) if existing else 0.0
        init_flipx = bool(existing.get("flip_horizontal", getattr(cfg.confocal_preprocessing, "flip_horizontal", True))) if existing else bool(getattr(cfg.confocal_preprocessing, "flip_horizontal", True))
        init_flipz = bool(existing.get("flip_z", getattr(cfg.confocal_preprocessing, "flip_z", True))) if existing else bool(getattr(cfg.confocal_preprocessing, "flip_z", True))
        if existing and existing.get("display_channel"):
            disp = str(existing.get("display_channel")).lower()
            for idx, name in enumerate(self.channel_names):
                if name.lower() == disp:
                    self.channel_index = idx
                    break
        self.flip_x = init_flipx
        self.flip_z = init_flipz
        self.flip_x_default = init_flipx
        self.flip_z_default = init_flipz
        self.default_channel_index = self.channel_index
        # Build widgets
        self._create_widgets(init_x, init_y, init_rot, init_flipx, init_flipz)
        self._refresh_moving_mip()
        self._update_display(None)

    @staticmethod
    def _pad_to(shape: Tuple[int, int], img: np.ndarray) -> np.ndarray:
        h, w = shape
        ph = (h - img.shape[0]) // 2
        pw = (w - img.shape[1]) // 2
        return np.pad(img, ((ph, h - img.shape[0] - ph), (pw, w - img.shape[1] - pw)), mode="constant", constant_values=0)

    def _get_existing_prematch(self) -> Optional[Dict[str, float]]:
        try:
            log_path = build_processing_log_path(self.cfg.processing_log, self.animal.animal_id, base_dir=Path(self.cfg.output_base_dir))
            if not log_path.exists():
                return None
            log = load_processing_log(log_path)
            stage = log.stages.get("confocal_to_anatomy_registration")
            if not stage:
                return None
            manual = stage.parameters.get("manual_prematch", {})
            return manual.get(self.confocal_session.session_id)
        except Exception:
            return None

    def _create_widgets(
        self,
        init_x: float,
        init_y: float,
        init_rot: float,
        init_flipx: bool,
        init_flipz: bool,
    ) -> None:
        max_shift = max(self.fixed_mip.shape) // 2
        self.x_slider = widgets.FloatSlider(value=init_x, min=-max_shift, max=max_shift, step=1.0, description="X shift (px):", continuous_update=False, layout=widgets.Layout(width="500px"))
        self.y_slider = widgets.FloatSlider(value=init_y, min=-max_shift, max=max_shift, step=1.0, description="Y shift (px):", continuous_update=False, layout=widgets.Layout(width="500px"))
        self.rot_slider = widgets.FloatSlider(value=init_rot, min=-180, max=180, step=1.0, description="Rotation (°):", continuous_update=False, layout=widgets.Layout(width="500px"))
        self.alpha_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05, description="Moving alpha:", continuous_update=False, layout=widgets.Layout(width="500px"))
        channel_options = [(f"{name} (index {idx})", idx) for idx, name in enumerate(self.channel_names)]
        self.channel_dropdown = widgets.Dropdown(options=channel_options, value=self.channel_index, description="Channel:", layout=widgets.Layout(width="400px"))
        self.flipx_toggle = widgets.Checkbox(value=init_flipx, description="Flip X (horizontal)")
        self.flipz_toggle = widgets.Checkbox(value=init_flipz, description="Flip Z (reverse slices)")
        # Info labels
        self.info_label = widgets.HTML(value=(
            f"<b>Animal:</b> {self.animal.animal_id} | <b>Confocal:</b> {self.confocal_session.session_id} | <b>Anatomy:</b> {self.anatomy_session.session_id}<br>"
            f"<b>Fixed (magenta):</b> 2p anatomy ({self.fixed_pixel_size[1]:.3f}×{self.fixed_pixel_size[0]:.3f} µm/px) | "
            f"<b>Moving (green, RAW):</b> confocal ({self.moving_pixel_size[1]:.3f}×{self.moving_pixel_size[0]:.3f} µm/px)"
        ))
        applied, off, signed = _compute_applied_angle(self.rot_slider.value, self.cfg)
        flag = "signed" if signed else "fixed"
        self.angle_info = widgets.HTML(value=(
            f"<b>GUI rotation:</b> {self.rot_slider.value:.1f}° → <b>applied:</b> {applied:.1f}° (offset {off:.1f}°, {flag})"
        ))
        self.status_label = widgets.HTML(value="")
        self.output = widgets.Output()
        # Observers
        def _on_rot(change):
            self._update_display(None)
        def _on_channel(change):
            self.channel_index = int(change["new"])
            self._refresh_moving_mip()
            self._update_display(None)

        def _on_flip(change):
            self.flip_x = bool(self.flipx_toggle.value)
            self.flip_z = bool(self.flipz_toggle.value)
            self._refresh_moving_mip()
            self._update_display(None)

        self.x_slider.observe(self._update_display, names="value")
        self.y_slider.observe(self._update_display, names="value")
        self.rot_slider.observe(_on_rot, names="value")
        self.alpha_slider.observe(self._update_display, names="value")
        self.channel_dropdown.observe(_on_channel, names="value")
        self.flipx_toggle.observe(_on_flip, names="value")
        self.flipz_toggle.observe(_on_flip, names="value")
        # Buttons
        self.save_button = widgets.Button(description="Save to Metadata", button_style="success", icon="check")
        self.skip_button = widgets.Button(description="Skip", button_style="warning", icon="forward")
        self.reset_button = widgets.Button(description="Reset", button_style="info", icon="refresh")
        self.reset_button.on_click(self._on_reset)
        self.save_button.on_click(self._on_save)
        self.skip_button.on_click(self._on_skip)

    def _refresh_moving_mip(self) -> None:
        vol = self.raw_stack[:, self.channel_index, :, :]
        if self.flip_z:
            vol = vol[::-1, :, :]
        mip = np.max(vol, axis=0)
        rescaled = _rescale_image(mip, self.scale_y, self.scale_x)
        if self.flip_x:
            rescaled = np.flip(rescaled, axis=-1)
        fixed_padded, moving_padded = self._pad_pair(self.fixed_mip_base, rescaled)
        self.fixed_mip = fixed_padded
        self.moving_mip_base = moving_padded

    def _pad_pair(self, fixed: np.ndarray, moving: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H = max(fixed.shape[0], moving.shape[0])
        W = max(fixed.shape[1], moving.shape[1])
        return self._pad_to((H, W), fixed), self._pad_to((H, W), moving)

    def _update_display(self, change) -> None:
        x_shift = float(self.x_slider.value)
        y_shift = float(self.y_slider.value)
        raw_rot = float(self.rot_slider.value)
        alpha = float(self.alpha_slider.value)
        applied_rot, offset_deg, signed_offset = _compute_applied_angle(raw_rot, self.cfg)
        flag = "signed" if signed_offset else "fixed"
        self.angle_info.value = (
            f"<b>GUI rotation:</b> {raw_rot:.1f}° → <b>applied:</b> {applied_rot:.1f}° (offset {offset_deg:.1f}°, {flag})"
        )
        moving_t = _apply_transform_2d(self.moving_mip_base, x_shift, y_shift, applied_rot)
        overlay = _create_overlay(self.fixed_mip, moving_t, alpha=alpha)
        with self.output:
            clear_output(wait=True)
            import matplotlib.pyplot as plt  # type: ignore

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(overlay)
            chan_name = self.channel_names[self.channel_index]
            ax.set_title(
                f"Overlay [{chan_name}]: x={x_shift:.1f}px, y={y_shift:.1f}px, applied rot={applied_rot:.1f}°",
                fontsize=12,
            )
            ax.axis("off")
            plt.tight_layout()
            plt.show()

    def _on_reset(self, _):
        self.x_slider.value = 0.0
        self.y_slider.value = 0.0
        self.rot_slider.value = 0.0
        self.channel_dropdown.value = self.default_channel_index
        self.flipx_toggle.value = self.flip_x_default
        self.flipz_toggle.value = self.flip_z_default
        self.flip_x = self.flip_x_default
        self.flip_z = self.flip_z_default
        self._refresh_moving_mip()
        self._update_display(None)
        self.status_label.value = ""

    def _on_save(self, _):
        _save_prematch_to_log(
            animal=self.animal,
            confocal_session=self.confocal_session,
            x_shift=float(self.x_slider.value),
            y_shift=float(self.y_slider.value),
            rotation_deg=float(self.rot_slider.value),
            flip_horizontal=bool(self.flipx_toggle.value),
            flip_z=bool(self.flipz_toggle.value),
            display_channel=self.channel_names[self.channel_index],
            cfg=self.cfg,
        )
        self.status_label.value = "<span style='color:green;font-weight:bold;'>✓ Saved!</span>"

    def _on_skip(self, _):
        self.status_label.value = "<span style='color:orange;font-weight:bold;'>⊘ Skipped</span>"

    def display(self):  # pragma: no cover - UI plumbing
        display(
            widgets.VBox(
                [
                    self.info_label,
                    self.angle_info,
                    widgets.HBox([self.channel_dropdown, self.flipx_toggle, self.flipz_toggle]),
                    self.output,
                    self.x_slider,
                    self.y_slider,
                    self.rot_slider,
                    self.alpha_slider,
                    widgets.HBox([self.save_button, self.skip_button, self.reset_button]),
                    self.status_label,
                ]
            )
        )


def collect_confocal_sessions(
    animals: Iterable[AnimalMetadata], cfg: ProjectConfig, *, force: bool = False,
    target_animal_id: Optional[str] = None, target_confocal_session_id: Optional[str] = None,
) -> List[Tuple[AnimalMetadata, AnatomySession, AnatomySession]]:
    out: List[Tuple[AnimalMetadata, AnatomySession, AnatomySession]] = []
    for animal in animals:
        if target_animal_id and animal.animal_id != target_animal_id:
            continue
        # Select anatomy session (first two-photon anatomy)
        anat: Optional[AnatomySession] = None
        for s in animal.sessions:
            if s.session_type == "anatomy_stack" and getattr(s.session_data, "stack_type", "") == "two_photon":
                anat = s
                break
        if anat is None:
            continue
        # Load processing log to check existing prematch
        existing: Dict[str, Dict[str, float]] = {}
        try:
            log_path = build_processing_log_path(cfg.processing_log, animal.animal_id, base_dir=Path(cfg.output_base_dir))
            if log_path.exists():
                log = load_processing_log(log_path)
                stage = log.stages.get("confocal_to_anatomy_registration")
                if stage:
                    existing = stage.parameters.get("manual_prematch", {}) or {}
        except Exception:
            existing = {}
        for s in animal.sessions:
            if s.session_type != "anatomy_stack" or getattr(s.session_data, "stack_type", "") != "confocal":
                continue
            if target_confocal_session_id and s.session_id != target_confocal_session_id:
                continue
            if not force and s.session_id in existing:
                continue
            out.append((animal, s, anat))
    return out


def run_manual_prematch_gui(
    animals: Optional[List[AnimalMetadata]] = None,
    cfg: Optional[ProjectConfig] = None,
    *,
    force: bool = False,
    target_animal_id: Optional[str] = None,
    target_confocal_session_id: Optional[str] = None,
) -> Dict[str, int]:  # pragma: no cover - UI orchestrator
    """Launch the manual prematch GUI for sessions that need it (or a specific one).

    Returns a stats dict with counts of saved/skipped/total.
    """
    if widgets is None or display is None or clear_output is None:
        raise RuntimeError("This function must be run inside a Jupyter environment (ipywidgets)")
    cfg = cfg or load_project_config()
    if animals is None:
        animals = list(load_animals().animals)
    sessions = collect_confocal_sessions(
        animals, cfg, force=force, target_animal_id=target_animal_id, target_confocal_session_id=target_confocal_session_id
    )
    if not sessions:
        print("\n✓ No sessions need prematching")
        return {"saved": 0, "skipped": 0, "total": 0}

    state = {"idx": 0, "sessions": sessions, "stats": {"saved": 0, "skipped": 0, "total": len(sessions)}, "cfg": cfg}

    def _show_next():
        if state["idx"] >= len(state["sessions"]):
            clear_output(wait=True)
            print("\n" + "=" * 60)
            print("📊 Manual Prematch Complete")
            print("=" * 60)
            print(f"Total sessions:  {state['stats']['total']}")
            print(f"Saved:           {state['stats']['saved']}")
            print(f"Skipped:         {state['stats']['skipped']}")
            print("=" * 60)
            return
        animal, conf_s, anat_s = state["sessions"][state["idx"]]
        clear_output(wait=True)
        print("\n" + "=" * 60)
        print(f"Session {state['idx'] + 1}/{len(state['sessions'])}: {animal.animal_id} / {conf_s.session_id}")
        print("=" * 60 + "\n")
        gui = ManualConfocalPrematchGUI(animal, conf_s, anat_s, state["cfg"])  # noqa: F841

        # Wrap button callbacks to advance
        def _on_save(_):
            gui._on_save(_)
            state["stats"]["saved"] += 1
            state["idx"] += 1
            _show_next()

        def _on_skip(_):
            gui._on_skip(_)
            state["stats"]["skipped"] += 1
            state["idx"] += 1
            _show_next()

        gui.save_button.on_click(_on_save)
        gui.skip_button.on_click(_on_skip)
        gui.display()

    _show_next()
    return state["stats"]
