
from __future__ import annotations

import tifffile as tiff
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from skimage.transform import resize, rotate as sk_rotate

from ..metadata.config import load_project_config, resolve_output_path
from ..metadata.models import AnimalMetadata
from ..pipeline import iter_animals_with_yaml
from ..registration.align_substack import read_good_nrrd_uint8


@dataclass
class RegistrationArtefacts:
    fixed_stack: np.ndarray
    moving_stack: np.ndarray
    df_results: pd.DataFrame
    anatomy_path: Path
    projection_path: Path


class RegistrationQC:
    """Load and visualise functional→anatomy registration artefacts."""

    def __init__(self, animal_ids: Optional[Iterable[str]] = None) -> None:
        self._cfg = load_project_config()
        self._animal_lookup: Dict[str, AnimalMetadata] = {
            animal.animal_id: animal for animal in iter_animals_with_yaml()
        }
        if not self._animal_lookup:
            raise RuntimeError("No animal metadata found via iter_animals_with_yaml()")

        summary_path = Path(self._cfg.output_base_dir) / self._cfg.functional_to_anatomy_registration.summary_csv
        if not summary_path.exists():
            raise FileNotFoundError(f"Combined registration summary not found: {summary_path}")

        summary = pd.read_csv(summary_path)
        summary = summary.sort_values("ncc_median", ascending=False).reset_index(drop=True)
        self._summary = summary

        if animal_ids is None:
            priority = summary.loc[summary["quality"].ne("ok"), "animal"].tolist()
            ordered = summary["animal"].tolist()
            if priority:
                self._animals = priority + [a for a in ordered if a not in priority]
            else:
                self._animals = ordered
        else:
            valid = set(summary["animal"].tolist())
            unknown = [a for a in animal_ids if a not in valid]
            if unknown:
                raise KeyError(f"Animals not present in summary: {sorted(unknown)}")
            self._animals = list(animal_ids)

        self._data_cache: Dict[str, RegistrationArtefacts] = {}
        self.default_rotation = float(self._cfg.functional_to_anatomy_registration.rotation_angle_deg)

    @property
    def summary(self) -> pd.DataFrame:
        return self._summary

    @property
    def animals(self) -> list[str]:
        return self._animals

    # ------------------------------------------------------------------ path helpers

    def _first_session(self, meta: AnimalMetadata, session_type: str, predicate=None):
        for session in meta.sessions:
            if session.session_type != session_type:
                continue
            if predicate and not predicate(session):
                continue
            if getattr(session, "include_in_analysis", True):
                return session
        return None

    def _registration_sessions(self, animal_id: str):
        try:
            meta = self._animal_lookup[animal_id]
        except KeyError as exc:
            available = ', '.join(sorted(self._animal_lookup.keys()))
            raise KeyError(f"Animal {animal_id} not found in metadata (available: {available})") from exc

        functional = self._first_session(meta, "functional_stack")
        anatomy = self._first_session(
            meta,
            "anatomy_stack",
            predicate=lambda s: getattr(s.session_data, "stack_type", "") == "two_photon",
        )
        if functional is None or anatomy is None:
            raise RuntimeError(
                f"Animal {animal_id} missing functional or two-photon anatomy session in metadata"
            )
        return functional, anatomy

    def _format(self, template: str, *, animal_id: str, functional_session, anatomy_session):
        context = {
            "animal_id": animal_id,
            "functional_session_id": getattr(functional_session, "session_id", ""),
            "anatomy_session_id": getattr(anatomy_session, "session_id", ""),
        }
        try:
            return template.format(**context)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing placeholder '{missing}' for template '{template}'")

    def _resolve_paths(self, animal_id: str) -> Dict[str, Path]:
        functional_session, anatomy_session = self._registration_sessions(animal_id)

        functional_root = resolve_output_path(
            animal_id,
            self._cfg.functional_preprocessing.root_subdir,
            cfg=self._cfg,
        )
        motion_root = functional_root / self._cfg.motion_correction.motion_output_subdir
        projections_dir = motion_root / self._cfg.motion_correction.projections_subdir

        avg_projection = projections_dir / self._format(
            self._cfg.functional_to_anatomy_registration.avg_projection_filename_template,
            animal_id=animal_id,
            functional_session=functional_session,
            anatomy_session=anatomy_session,
        )
        max_projection = projections_dir / self._format(
            self._cfg.functional_to_anatomy_registration.max_projection_filename_template,
            animal_id=animal_id,
            functional_session=functional_session,
            anatomy_session=anatomy_session,
        )

        registration_root = resolve_output_path(
            animal_id,
            self._cfg.functional_to_anatomy_registration.registration_output_subdir,
            cfg=self._cfg,
        )
        per_animal_csv = registration_root / self._format(
            self._cfg.functional_to_anatomy_registration.registration_csv_template,
            animal_id=animal_id,
            functional_session=functional_session,
            anatomy_session=anatomy_session,
        )

        anatomy_root = resolve_output_path(
            animal_id,
            self._cfg.anatomy_preprocessing.root_subdir,
            cfg=self._cfg,
        )
        anatomy_stack = anatomy_root / self._format(
            self._cfg.anatomy_preprocessing.stack_filename_template,
            animal_id=animal_id,
            functional_session=functional_session,
            anatomy_session=anatomy_session,
        )

        return {
            "avg_projection": avg_projection,
            "max_projection": max_projection,
            "registration_csv": per_animal_csv,
            "anatomy_stack": anatomy_stack,
        }

    def _load_animal_data(self, animal_id: str) -> RegistrationArtefacts:
        cached = self._data_cache.get(animal_id)
        if cached is not None:
            return cached

        paths = self._resolve_paths(animal_id)
        anatomy_path = paths["anatomy_stack"]
        projection_path = paths["avg_projection"]

        fixed_stack = read_good_nrrd_uint8(
            anatomy_path,
            flip_horizontal=self._cfg.functional_to_anatomy_registration.flip_fixed_horizontal,
        )
        moving_stack = tiff.imread(str(projection_path))
        df_results = pd.read_csv(paths["registration_csv"])
        artefacts = RegistrationArtefacts(
            fixed_stack=fixed_stack,
            moving_stack=moving_stack,
            df_results=df_results,
            anatomy_path=anatomy_path,
            projection_path=projection_path,
        )
        self._data_cache[animal_id] = artefacts
        return artefacts

    # ------------------------------------------------------------------ plotting helpers

    @staticmethod
    def _normalise(im: np.ndarray) -> np.ndarray:
        im = im.astype(np.float32)
        p1, p99 = np.percentile(im, (1, 99))
        if p99 <= p1:
            p1, p99 = im.min(), max(im.max(), im.min() + 1e-3)
        return np.clip((im - p1) / (p99 - p1 + 1e-6), 0, 1)

    def _build_figure(self, artefacts: RegistrationArtefacts, moving_index: int, rot_deg: float) -> plt.Figure:
        df_results = artefacts.df_results
        if not (0 <= moving_index < artefacts.moving_stack.shape[0]):
            raise ValueError("moving_index out of range")
        row = df_results[df_results["moving_plane"] == moving_index]
        if row.empty:
            raise ValueError("No results found for the provided moving plane")
        row = row.iloc[0]

        z = int(row["z_index"])
        y = float(row["y_px"])
        x = float(row["x_px"])
        scale = float(row["scale_moving_to_fixed"])
        animal = str(row.get("animal", "?"))

        fixed_slice = artefacts.fixed_stack[z].astype(np.float32)
        moving_slice = artefacts.moving_stack[moving_index].astype(np.float32)

        moving_scaled = resize(moving_slice, (
            max(1, int(round(moving_slice.shape[0] * scale))),
            max(1, int(round(moving_slice.shape[1] * scale))),
        ), anti_aliasing=True, preserve_range=True)
        h, w = moving_scaled.shape

        fixed_norm = self._normalise(fixed_slice)
        moving_norm = self._normalise(moving_scaled)

        def _safe_crop(img: np.ndarray, top: float, left: float, height: int, width: int) -> np.ndarray:
            y0, x0 = int(np.floor(top)), int(np.floor(left))
            y1, x1 = y0 + height, x0 + width
            H, W = img.shape
            ya, xa = max(0, y0), max(0, x0)
            yb, xb = min(H, y1), min(W, x1)
            if ya >= yb or xa >= xb:
                return np.zeros((height, width), dtype=img.dtype)
            crop = img[ya:yb, xa:xb]
            pad_top, pad_left = ya - y0, xa - x0
            pad_bottom = (y0 + height) - yb
            pad_right = (x0 + width) - xb
            if any(v > 0 for v in (pad_top, pad_left, pad_bottom, pad_right)):
                crop = np.pad(
                    crop,
                    ((max(0, pad_top), max(0, pad_bottom)), (max(0, pad_left), max(0, pad_right))),
                    mode="edge",
                )
            return crop

        matched_crop = _safe_crop(fixed_slice, y, x, h, w)
        matched_norm = self._normalise(matched_crop)

        overlay = np.stack([fixed_norm, fixed_norm, fixed_norm], axis=-1)
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        ya, xa = max(0, y0), max(0, x0)
        yb, xb = min(fixed_slice.shape[0], y0 + h), min(fixed_slice.shape[1], x0 + w)
        sy1, sx1 = ya - y0, xa - x0
        sy2, sx2 = sy1 + (yb - ya), sx1 + (xb - xa)
        if yb > ya and xb > xa:
            dstG = overlay[ya:yb, xa:xb, 1]
            src = moving_norm[sy1:sy2, sx1:sx2]
            overlay[ya:yb, xa:xb, 1] = 0.65 * dstG + 0.35 * src
            rr_thick = 2
            overlay[max(0, y0):min(overlay.shape[0], y0 + rr_thick), max(0, x0):min(overlay.shape[1], x0 + w), :] = [0, 1, 0]
            overlay[max(0, y0 + h - rr_thick):min(overlay.shape[0], y0 + h), max(0, x0):min(overlay.shape[1], x0 + w), :] = [0, 1, 0]
            overlay[max(0, y0):min(overlay.shape[0], y0 + h), max(0, x0):min(overlay.shape[1], x0 + rr_thick), :] = [0, 1, 0]
            overlay[max(0, y0):min(overlay.shape[0], y0 + h), max(0, x0 + w - rr_thick):min(overlay.shape[1], x0 + w), :] = [0, 1, 0]

        def _rotate(img: np.ndarray) -> np.ndarray:
            if abs(float(rot_deg)) > 1e-6:
                return sk_rotate(img, angle=rot_deg, resize=False, preserve_range=True, mode="constant", cval=0.0, order=1)
            return img

        overlay_disp = _rotate(overlay)
        moving_disp = _rotate(moving_norm)
        matched_disp = _rotate(matched_norm)
        if matched_disp.shape != moving_disp.shape:
            matched_disp = resize(matched_disp, moving_disp.shape, preserve_range=True, anti_aliasing=True)

        right_disp = np.concatenate([moving_disp, matched_disp], axis=1)

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(np.clip(overlay_disp, 0, 1), interpolation="nearest")
        ax1.set_title("Fixed with moving overlay (rotated display)")
        ax1.set_axis_off()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(right_disp, cmap="gray", interpolation="nearest")
        ax2.set_title("Left: moving (scaled) | Right: matched crop  (rotated display)")
        ax2.set_axis_off()

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
        text_lines.append("")
        text_lines.append(f"fixed={artefacts.anatomy_path.name}")
        text_lines.append(f"moving={artefacts.projection_path.name}")
        text_str = f"{animal}, plane {moving_index}\n\n" + "\n".join(text_lines)
        ax1.text(
            0.02,
            0.98,
            text_str,
            transform=ax1.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            color="yellow",
            family="monospace",
            bbox=dict(facecolor="black", alpha=0.5, pad=4, edgecolor="none"),
        )

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ public helpers

    def create_viewer(self) -> widgets.VBox:
        if not self._animals:
            raise RuntimeError("No animals available for QC")

        dropdown = widgets.Dropdown(options=self._animals, description="Animal:")
        rotation = widgets.FloatSlider(
            value=self.default_rotation,
            min=-180.0,
            max=180.0,
            step=1.0,
            description="Rotation",
            continuous_update=False,
        )
        plane_slider = widgets.IntSlider(value=0, min=0, step=1, description="moving plane")
        output = widgets.Output()

        def _update_plane_bounds(animal_id: str):
            data = self._load_animal_data(animal_id)
            plane_slider.max = max(0, data.moving_stack.shape[0] - 1)
            plane_slider.value = min(plane_slider.value, plane_slider.max)

        def _render(*_):
            animal_id = dropdown.value
            data = self._load_animal_data(animal_id)
            fig = self._build_figure(data, plane_slider.value, rotation.value)
            with output:
                output.clear_output(wait=True)
                display(fig)
                plt.close(fig)

        def _on_animal(change):
            if change.get('new') is None:
                return
            _update_plane_bounds(change['new'])
            _render()

        dropdown.observe(_on_animal, names='value')
        rotation.observe(_render, names='value')
        plane_slider.observe(_render, names='value')

        _update_plane_bounds(self._animals[0])
        _render()

        controls = widgets.VBox([dropdown, rotation, plane_slider])
        return widgets.VBox([controls, output])

    def show_static(self, animal_id: str, plane_index: int = 0, rot_deg: Optional[float] = None) -> None:
        data = self._load_animal_data(animal_id)
        fig = self._build_figure(data, plane_index, self.default_rotation if rot_deg is None else rot_deg)
        display(fig)
        plt.close(fig)
