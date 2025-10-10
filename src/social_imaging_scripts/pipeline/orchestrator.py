"""High-level orchestration helpers for multi-animal processing runs."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Literal, Mapping, Optional

import pandas as pd
import tifffile

from ..metadata.config import (
    ProjectConfig,
    load_project_config,
    load_yaml_mapping,
    resolve_output_path,
    resolve_raw_path,
    resolve_reference_brain,
)
from ..metadata.loader import load_animals
from ..metadata.models import AnatomySession, AnimalMetadata, FunctionalSession
from ..preprocessing import functional_projections
from ..preprocessing.two_photon import anatomy as anatomy_preproc
from ..preprocessing.two_photon import functional as functional_preproc
from ..preprocessing.two_photon import motion as motion_correction
from ..registration import align_substack
from ..registration.functional_to_anatomy_ants import register_planes_pass1
from ..registration.fireants_pipeline import FireANTsRegistrationConfig, register_two_photon_anatomy
from .processing_log import (
    AnimalProcessingLog,
    ArtefactRef,
    build_processing_log_path,
    load_processing_log,
    save_processing_log,
)

logger = logging.getLogger(__name__)

_LOGGING_CONFIGURED = False

SessionStatus = Literal["success", "skipped", "failed"]
AnimalStatus = Literal["success", "partial", "skipped", "failed"]


@dataclass
class SessionResult:
    """Outcome of processing a single session for one animal."""

    animal_id: str
    session_id: str
    session_type: str
    status: SessionStatus
    message: Optional[str] = None
    outputs: dict[str, Path] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        return {
            "animal_id": self.animal_id,
            "session_id": self.session_id,
            "session_type": self.session_type,
            "status": self.status,
            "message": self.message,
            "outputs": {key: str(value) for key, value in self.outputs.items()},
        }


@dataclass
class AnimalResult:
    """Collection of session results for one animal."""

    animal_id: str
    sessions: list[SessionResult] = field(default_factory=list)

    @property
    def status(self) -> AnimalStatus:
        if any(result.status == "failed" for result in self.sessions):
            return "failed"
        if not self.sessions:
            return "skipped"
        if all(result.status == "skipped" for result in self.sessions):
            return "skipped"
        if all(result.status == "success" for result in self.sessions):
            return "success"
        return "partial"

    def to_records(self) -> list[dict[str, object]]:
        return [result.to_record() for result in self.sessions]


@dataclass
class PipelineResult:
    """Aggregate results for an entire run across multiple animals."""

    animals: list[AnimalResult]

    def iter_session_records(self) -> Iterator[dict[str, object]]:
        for animal in self.animals:
            yield from animal.to_records()

    def to_dataframe(self):  # pragma: no cover - optional dependency
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - best-effort helper
            raise RuntimeError("pandas is required for to_dataframe()") from exc
        return pd.DataFrame(self.iter_session_records())


def _default_metadata_dir() -> Path:
    """Return repository-level ``metadata/animals`` regardless of cwd."""

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "metadata" / "animals"


def _default_ops_template_path() -> Path:
    """Return the shared Suite2p ops template shipped with the repository."""

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "suite2p_ops_may2025.npy"



def iter_animals_with_yaml(yaml_dir: Optional[Path] = None) -> Iterator[AnimalMetadata]:
    """Yield :class:`AnimalMetadata` for each YAML file in *yaml_dir*."""

    directory = yaml_dir or _default_metadata_dir()
    collection = load_animals(base_dir=directory)
    for animal in collection.animals:
        yield animal


def _resolve_session_raw_path(
    *, animal: AnimalMetadata, relative_path: Path, cfg: ProjectConfig
) -> Path:
    base = Path(relative_path)
    if animal.root_dir:
        base = Path(animal.root_dir) / base
    return resolve_raw_path(base, cfg=cfg)


def _apply_overrides(target, overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise AttributeError(f"Unknown configuration field '{key}'")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_overrides(current, value)
        else:
            setattr(target, key, value)


def _prepare_fireants_config(
    base_config: Optional[FireANTsRegistrationConfig],
    overrides: Optional[Mapping[str, object]],
) -> FireANTsRegistrationConfig:
    config = copy.deepcopy(base_config) if base_config is not None else FireANTsRegistrationConfig()
    if overrides:
        _apply_overrides(config, overrides)
    return config


def _extract_plane_index(path: Path) -> int:
    suffix = path.stem.split("plane")[-1]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Unexpected plane filename: {path.name}") from exc


def _resolve_fps(
    fps_config: Optional[float],
    animal: AnimalMetadata,
    session: FunctionalSession,
) -> float:
    if fps_config is None:
        raise ValueError(
            f"Frame rate not provided for functional session {animal.animal_id}::{session.session_id}"
        )
    return float(fps_config)


def _load_session_pixel_size(metadata_path: Path) -> tuple[float, float]:
    data = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    pixel_size = data.get("pixel_size_xy_um")
    if (
        not isinstance(pixel_size, (list, tuple))
        or len(pixel_size) != 2
        or any(value is None for value in pixel_size)
    ):
        raise ValueError(f"Missing pixel_size_xy_um in metadata: {metadata_path}")
    return float(pixel_size[0]), float(pixel_size[1])


def _coerce_parameters(parameters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not parameters:
        return {}
    coerced: Dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, Path):
            coerced[key] = str(value)
        elif isinstance(value, (list, tuple)):
            coerced[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            coerced[key] = value
    return coerced


def _stage_key(session_type: str, session_id: str) -> str:
    return f"{session_type}:{session_id}"


def _update_processing_log_stage(
    log: AnimalProcessingLog,
    stage_name: str,
    result: SessionResult,
    parameters: Optional[Mapping[str, Any]] = None,
) -> None:
    record = log.ensure_stage(stage_name)
    record.status = result.status
    record.parameters = _coerce_parameters(parameters)
    record.notes = result.message
    now = datetime.now(tz=timezone.utc)
    if record.started_at is None:
        record.started_at = now
    record.completed_at = now
    outputs: Dict[str, ArtefactRef] = {}
    for key, value in result.outputs.items():
        if isinstance(value, Path):
            outputs[key] = ArtefactRef.from_path(value)
    record.outputs = outputs
    log.touch()


def process_functional_session(
    *,
    animal: AnimalMetadata,
    session: FunctionalSession,
    cfg: ProjectConfig,
    ops_template: dict,
    fps_config: Optional[
        float
        | Mapping[str, float]
        | Callable[[AnimalMetadata, FunctionalSession], Optional[float]]
    ] = None,
    fast_disk: Optional[Path] = None,
    reprocess: bool = False,
    reprocess_motion: bool = False,
    enable_preprocessing: bool = True,
    enable_motion: bool = True,
) -> SessionResult:
    functional_cfg = cfg.functional_preprocessing
    motion_cfg = cfg.motion_correction
    context = {"animal_id": animal.animal_id, "session_id": session.session_id}

    result = SessionResult(
        animal_id=animal.animal_id,
        session_id=session.session_id,
        session_type=session.session_type,
        status="failed",
    )

    if not enable_preprocessing and not enable_motion:
        result.status = "skipped"
        result.message = "functional pipeline disabled"
        return result

    settings = session.session_data.preprocessing_two_photon
    if enable_preprocessing and settings is None:
        result.message = "missing two-photon preprocessing settings"
        return result

    output_root = resolve_output_path(
        animal.animal_id,
        functional_cfg.root_subdir,
        cfg=cfg,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    plane_dir = output_root / functional_cfg.planes_subdir
    plane_dir.mkdir(parents=True, exist_ok=True)

    metadata_name = functional_cfg.metadata_filename_template.format(**context)
    metadata_path = plane_dir / metadata_name
    preproc_generated = False
    notes: list[str] = []

    if enable_preprocessing:
        try:
            raw_dir = _resolve_session_raw_path(
                animal=animal, relative_path=Path(session.session_data.raw_path), cfg=cfg
            )
        except Exception as exc:
            result.message = f"failed to resolve raw path: {exc}"
            return result

        preprocessing_needed = reprocess or not metadata_path.exists()
        try:
            if preprocessing_needed:
                # Plane splitting mirrors the single-animal notebook but now feeds the batch driver.
                preproc_outputs = functional_preproc.run(
                    animal_id=animal.animal_id,
                    session_id=session.session_id,
                    raw_dir=raw_dir,
                    output_root=output_root,
                    settings=settings,
                    planes_subdir=functional_cfg.planes_subdir,
                    plane_filename_template=functional_cfg.plane_filename_template,
                    metadata_filename=functional_cfg.metadata_filename_template,
                )
                result.outputs.update(
                    {f"preprocess_{key}": Path(value) for key, value in preproc_outputs.items()}
                )
                preproc_generated = True
            else:
                result.outputs["preprocess_metadata"] = metadata_path
        except Exception as exc:  # pragma: no cover - pipeline side effects
            logger.exception("Functional preprocessing failed", exc_info=exc)
            result.message = f"functional preprocessing failed: {exc}"
            return result
    else:
        if metadata_path.exists():
            result.outputs["preprocess_metadata"] = metadata_path

    plane_pattern = functional_cfg.plane_filename_template.format(
        animal_id=animal.animal_id,
        session_id=session.session_id,
        plane_index="*",
    )
    plane_paths = (
        sorted(plane_dir.glob(plane_pattern))
        if enable_motion or preproc_generated
        else []
    )

    motion_generated = False
    if enable_motion:
        if not plane_paths:
            result.message = f"no plane TIFFs found in {plane_dir}"
            return result

        fps = _resolve_fps(fps_config, animal, session)
        motion_outputs_collected = True

        motion_subdir = motion_cfg.motion_output_subdir
        suite2p_subdir = motion_cfg.suite2p_output_subdir

        for plane_path in plane_paths:
            plane_idx = _extract_plane_index(plane_path)
            try:
                # Run Suite2p plane-by-plane so outputs match the legacy project layout.
                motion_outputs = motion_correction.run_motion_correction(
                    animal=animal,
                    plane_idx=plane_idx,
                    plane_tiff=plane_path,
                    session_id=session.session_id,
                    ops_template=ops_template,
                    fps=fps,
                    output_root=output_root,
                    fast_disk=fast_disk,
                    reprocess=reprocess_motion,
                    motion_output_subdir=motion_subdir,
                    suite2p_output_subdir=suite2p_subdir,
                    plane_folder_template=motion_cfg.plane_folder_template,
                    segmentation_folder_template=motion_cfg.segmentation_folder_template,
                    motion_filename_template=motion_cfg.motion_filename_template,
                    metadata_filename=motion_cfg.metadata_filename,
                )
                for key, value in motion_outputs.items():
                    result.outputs[f"plane{plane_idx}_{key}"] = Path(value)
                if {"motion_tiff", "segmentation_folder"}.intersection(motion_outputs.keys()):
                    motion_generated = True
            except Exception as exc:  # pragma: no cover - external tool failure
                logger.exception("Motion correction failed", exc_info=exc)
                result.message = f"motion correction failed for plane {plane_idx}: {exc}"
                motion_outputs_collected = False
                break

        if not motion_outputs_collected:
            return result
    else:
        notes.append("motion correction disabled")

    if enable_preprocessing:
        notes.append(
            "plane splitting ran" if preproc_generated else "plane splitting reused existing outputs"
        )
    else:
        notes.append("plane splitting disabled")

    if enable_motion:
        notes.append(
            "motion correction ran" if motion_generated else "motion correction reused existing outputs"
        )

    result.status = "success" if preproc_generated or motion_generated else "skipped"
    if notes:
        result.message = "; ".join(notes)
    return result


def process_anatomy_session(
    *,
    animal: AnimalMetadata,
    session: AnatomySession,
    cfg: ProjectConfig,
    fireants_config: Optional[FireANTsRegistrationConfig] = None,
    fireants_overrides: Optional[dict] = None,
    reprocess_preprocessing: bool = False,
    reprocess_registration: bool = False,
    enable_preprocessing: bool = True,
    enable_registration: bool = True,
) -> SessionResult:
    anatomy_cfg = cfg.anatomy_preprocessing
    fireants_stage_cfg = cfg.fireants_registration
    context = {"animal_id": animal.animal_id, "session_id": session.session_id}

    result = SessionResult(
        animal_id=animal.animal_id,
        session_id=session.session_id,
        session_type=session.session_type,
        status="failed",
    )

    if not enable_preprocessing and not enable_registration:
        result.status = "skipped"
        result.message = "anatomy pipeline disabled"
        return result

    preprocess_root = resolve_output_path(
        animal.animal_id,
        anatomy_cfg.root_subdir,
        cfg=cfg,
    )
    preprocess_root.mkdir(parents=True, exist_ok=True)

    metadata_name = anatomy_cfg.metadata_filename_template.format(**context)
    preprocess_metadata = preprocess_root / metadata_name
    preproc_generated = False
    stack_path: Optional[Path] = None

    if enable_preprocessing:
        try:
            raw_path = _resolve_session_raw_path(
                animal=animal, relative_path=Path(session.session_data.raw_path), cfg=cfg
            )
        except Exception as exc:
            result.message = f"failed to resolve raw anatomy path: {exc}"
            return result

        preprocessing_needed = reprocess_preprocessing or not preprocess_metadata.exists()

        try:
            if preprocessing_needed:
                preproc_outputs = anatomy_preproc.run(
                    animal_id=animal.animal_id,
                    session_id=session.session_id,
                    raw_dir=
                    (
                        raw_path.parent
                        if raw_path.is_file()
                        or raw_path.suffix.lower() in {".tif", ".tiff"}
                        else raw_path
                    ),
                    output_root=preprocess_root,
                    settings=session.session_data.preprocessing_two_photon,
                    stack_filename=anatomy_cfg.stack_filename_template,
                    metadata_filename=anatomy_cfg.metadata_filename_template,
                )
                default_stack = preprocess_root / anatomy_cfg.stack_filename_template.format(**context)
                stack_path = Path(preproc_outputs.get("stack", default_stack))
                result.outputs.update(
                    {f"anatomy_preprocess_{key}": Path(value) for key, value in preproc_outputs.items()}
                )
                preproc_generated = True
            else:
                stack_name = anatomy_cfg.stack_filename_template.format(**context)
                stack_path = preprocess_root / stack_name
                if preprocess_metadata.exists():
                    result.outputs["anatomy_preprocess_metadata"] = preprocess_metadata
        except Exception as exc:  # pragma: no cover - external IO
            logger.exception("Anatomy preprocessing failed", exc_info=exc)
            result.message = f"anatomy preprocessing failed: {exc}"
            return result
    else:
        stack_candidate = preprocess_root / anatomy_cfg.stack_filename_template.format(**context)
        if stack_candidate.exists():
            stack_path = stack_candidate
            if preprocess_metadata.exists():
                result.outputs["anatomy_preprocess_metadata"] = preprocess_metadata

    if stack_path is None or not stack_path.exists():
        result.message = f"anatomy stack not found at {stack_path}"
        return result

    registration_root = resolve_output_path(
        animal.animal_id,
        fireants_stage_cfg.output_subdir,
        cfg=cfg,
    )
    registration_root.mkdir(parents=True, exist_ok=True)

    warped_name = fireants_stage_cfg.warped_stack_template.format(**context)
    warped_stack = registration_root / warped_name
    registration_generated = False

    if enable_registration:
        registration_needed = reprocess_registration or not warped_stack.exists()

        if registration_needed:
            config = _prepare_fireants_config(fireants_config, fireants_overrides)
            reference_path = resolve_reference_brain(cfg=cfg)
            print(
                f"[{animal.animal_id}::{session.session_id}] FireANTs reference brain: {reference_path}"
            )
            print(
                f"[{animal.animal_id}::{session.session_id}] FireANTs reference exists: {reference_path.exists()}"
            )

            try:
                # Delegate to the validated FireANTs pipeline; collect artefact paths for reporting.
                outputs = register_two_photon_anatomy(
                    animal_id=animal.animal_id,
                    session_id=session.session_id,
                    stack_path=stack_path,
                    reference_path=reference_path,
                    output_root=registration_root,
                    animal_metadata=animal,
                    session_metadata=session,
                    config=config,
                )

                warped_from_runner = Path(outputs.get("warped_stack", warped_stack))
                if warped_from_runner != warped_stack and warped_from_runner.exists():
                    if warped_stack.exists():
                        warped_stack.unlink()
                    warped_from_runner.rename(warped_stack)
                    outputs["warped_stack"] = warped_stack

                result.outputs.update(
                    {f"fireants_{key}": Path(value) if isinstance(value, Path) else value for key, value in outputs.items()}
                )
                registration_generated = True
            except Exception as exc:  # pragma: no cover - GPU pipeline failure
                logger.exception("FireANTs registration failed", exc_info=exc)
                result.message = f"fireants registration failed: {exc}"
                return result
        else:
            result.outputs["fireants_warped"] = warped_stack
    else:
        if warped_stack.exists():
            result.outputs["fireants_warped"] = warped_stack

    notes: list[str] = []
    if enable_preprocessing:
        notes.append(
            "anatomy preprocessing ran"
            if preproc_generated
            else "anatomy preprocessing reused existing outputs"
        )
    else:
        notes.append("anatomy preprocessing disabled")

    if enable_registration:
        notes.append(
            "fireants registration ran"
            if registration_generated
            else "fireants registration reused existing outputs"
        )
    else:
        notes.append("fireants registration disabled")

    result.status = "success" if preproc_generated or registration_generated else "skipped"
    if notes:
        result.message = "; ".join(notes)
    return result


def process_functional_to_anatomy_registration(
    *,
    animal: AnimalMetadata,
    functional_session: FunctionalSession,
    anatomy_session: AnatomySession,
    cfg: ProjectConfig,
) -> SessionResult:
    """Register functional projections to the anatomy stack for one animal."""

    stage_cfg = cfg.functional_to_anatomy_registration
    functional_cfg = cfg.functional_preprocessing
    anatomy_cfg = cfg.anatomy_preprocessing
    result = SessionResult(
        animal_id=animal.animal_id,
        session_id=functional_session.session_id,
        session_type="functional_to_anatomy_registration",
        status="failed",
    )

    if not stage_cfg.enabled:
        result.status = "skipped"
        result.message = "functional-to-anatomy registration disabled"
        return result

    context = {
        "animal_id": animal.animal_id,
        "functional_session_id": functional_session.session_id,
        "anatomy_session_id": anatomy_session.session_id,
    }

    def _format(template: str) -> str:
        try:
            return template.format(**context)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing placeholder '{missing}' for template '{template}'")

    functional_root = resolve_output_path(
        animal.animal_id,
        cfg.functional_preprocessing.root_subdir,
        cfg=cfg,
    )
    motion_root = functional_root / cfg.motion_correction.motion_output_subdir
    if not motion_root.exists():
        result.message = f"motion outputs not found: {motion_root}"
        return result

    motion_tiffs = functional_projections.collect_tiff_files(
        motion_root, recursive=True
    )
    if not motion_tiffs:
        result.message = f"no motion-corrected TIFFs discovered under {motion_root}"
        return result

    projections_dir = motion_root / cfg.motion_correction.projections_subdir
    projections_dir.mkdir(parents=True, exist_ok=True)

    max_projection_path = projections_dir / _format(
        stage_cfg.max_projection_filename_template
    )
    avg_projection_path = projections_dir / _format(
        stage_cfg.avg_projection_filename_template
    )

    if stage_cfg.reprocess or not (
        max_projection_path.exists() and avg_projection_path.exists()
    ):
        try:
            functional_projections.save_max_avg_projections(
                motion_root,
                angle_deg=stage_cfg.rotation_angle_deg,
                out_dir=projections_dir,
                animal_name=animal.animal_id,
                recursive=True,
            )
        except Exception as exc:  # pragma: no cover - external IO
            logger.exception("Failed to build functional projections", exc_info=exc)
            result.message = f"failed to compute functional projections: {exc}"
            return result

    if not max_projection_path.exists() or not avg_projection_path.exists():
        result.message = (
            "functional projections missing after generation attempt: "
            f"max={max_projection_path}, avg={avg_projection_path}"
        )
        return result

    result.outputs["projection_max"] = max_projection_path
    result.outputs["projection_avg"] = avg_projection_path

    registration_root = resolve_output_path(
        animal.animal_id,
        stage_cfg.registration_output_subdir,
        cfg=cfg,
    )
    registration_root.mkdir(parents=True, exist_ok=True)
    per_animal_csv = registration_root / _format(stage_cfg.registration_csv_template)

    if per_animal_csv.exists() and not stage_cfg.reprocess:
        result.outputs["registration_csv"] = per_animal_csv
        result.status = "success"
        result.message = "functional-to-anatomy registration reused existing outputs"
        return result

    anatomy_root = resolve_output_path(
        animal.animal_id,
        cfg.anatomy_preprocessing.root_subdir,
        cfg=cfg,
    )
    stack_template = cfg.anatomy_preprocessing.stack_filename_template
    stack_path = anatomy_root / _format(stack_template)
    if not stack_path.exists():
        result.message = f"anatomy stack not found at {stack_path}"
        return result

    try:
        fixed_stack = align_substack.read_good_nrrd_uint8(
            stack_path, flip_horizontal=stage_cfg.flip_fixed_horizontal
        )
    except Exception as exc:  # pragma: no cover - IO heavy
        logger.exception("Failed to load anatomy stack", exc_info=exc)
        result.message = f"failed to load anatomy stack: {exc}"
        return result

    try:
        moving_stack = tifffile.imread(str(avg_projection_path))
    except Exception as exc:  # pragma: no cover - IO heavy
        logger.exception("Failed to load functional projections", exc_info=exc)
        result.message = f"failed to load functional projections: {exc}"
        return result

    functional_root = resolve_output_path(
        animal.animal_id,
        functional_cfg.root_subdir,
        cfg=cfg,
    )
    plane_dir = functional_root / functional_cfg.planes_subdir
    functional_metadata_name = functional_cfg.metadata_filename_template.format(
        animal_id=animal.animal_id,
        session_id=functional_session.session_id,
    )
    functional_metadata_path = plane_dir / functional_metadata_name

    anatomy_metadata_name = anatomy_cfg.metadata_filename_template.format(
        animal_id=animal.animal_id,
        session_id=anatomy_session.session_id,
    )
    anatomy_metadata_path = anatomy_root / anatomy_metadata_name

    try:
        functional_pixel_size = _load_session_pixel_size(functional_metadata_path)
        anatomy_pixel_size = _load_session_pixel_size(anatomy_metadata_path)
    except Exception as exc:
        logger.exception("Failed to read pixel size metadata", exc_info=exc)
        result.message = f"missing pixel size metadata: {exc}"
        return result

    ratio_x = functional_pixel_size[0] / anatomy_pixel_size[0]
    ratio_y = functional_pixel_size[1] / anatomy_pixel_size[1]
    expected_scale = float((ratio_x + ratio_y) / 2.0)

    scale_window = max(0.0, float(stage_cfg.scale_window))
    if stage_cfg.n_scales <= 0:
        raise ValueError("n_scales must be positive")
    if scale_window == 0.0 and stage_cfg.n_scales > 1:
        logger.warning(
            "scale_window=0 but n_scales=%d; scales will be identical", stage_cfg.n_scales
        )

    min_scale = expected_scale * (1.0 - scale_window)
    max_scale = expected_scale * (1.0 + scale_window)
    if stage_cfg.n_scales == 1:
        min_scale = max_scale = expected_scale
    min_scale = max(min_scale, 1e-6)
    max_scale = max(max_scale, min_scale)

    plane_indices = list(range(moving_stack.shape[0]))
    functional_planes = [moving_stack[idx] for idx in plane_indices]
    progress_cb = (lambda m: logger.info("[%s] %s", animal.animal_id, m)) if stage_cfg.progress else None

    try:
        results = register_planes_pass1(
            anatomy_stack=fixed_stack,
            functional_planes=functional_planes,
            plane_indices=plane_indices,
            downscale_if_needed=False,
            scale_range=(float(min_scale), float(max_scale)),
            n_scales=stage_cfg.n_scales,
            z_stride_coarse=stage_cfg.z_stride_coarse,
            z_refine_radius=stage_cfg.z_refine_radius,
            gaussian_sigma=stage_cfg.gaussian_sigma,
            early_stop_score=stage_cfg.early_stop_score,
            progress=progress_cb,
        )
    except Exception as exc:  # pragma: no cover - algorithmic failure
        logger.exception("Functional-to-anatomy registration failed", exc_info=exc)
        result.message = f"functional-to-anatomy registration failed: {exc}"
        return result

    rows: list[dict[str, object]] = []
    for res in results:
        translation = res.transform.GetTranslation()
        matrix = res.transform.GetMatrix()
        scale = float(matrix[0]) if matrix else float("nan")
        rows.append(
            {
                "animal": animal.animal_id,
                "moving_plane": res.plane_index,
                "z_index": res.best_z,
                "z_um": res.best_z * stage_cfg.fixed_z_spacing_um if res.best_z >= 0 else float("nan"),
                "y_px": float(translation[1]) if len(translation) > 1 else float("nan"),
                "x_px": float(translation[0]) if len(translation) > 0 else float("nan"),
                "scale_moving_to_fixed": scale,
                "ncc_score": res.ncc,
                "success": res.success,
                "message": res.message,
                "qc_flag": "",
                "expected_scale": expected_scale,
                "scale_range_min": float(min_scale),
                "scale_range_max": float(max_scale),
            }
        )

    df_results = pd.DataFrame(rows)

    df_results.to_csv(per_animal_csv, index=False)
    result.outputs["registration_csv"] = per_animal_csv
    result.outputs["anatomy_stack"] = stack_path

    n_planes = len(df_results)
    successful = df_results[df_results["success"]]
    median_ncc = float(successful["ncc_score"].median()) if not successful.empty else float("nan")
    result.status = "success"
    result.message = f"registered {n_planes} planes; median NCC={median_ncc:.3f}"
    return result


def run_pipeline(
    animal_ids: Optional[Iterable[str]] = None,
    *,
    metadata_base: Optional[Path] = None,
    cfg: Optional[ProjectConfig] = None,
    ops_path: Optional[Path] = None,
) -> PipelineResult:
    """Execute the batch pipeline, defaulting to parameters supplied by the project config."""

    cfg = cfg or load_project_config()

    global _LOGGING_CONFIGURED
    if getattr(cfg, 'apply_log_settings', False) and not _LOGGING_CONFIGURED:
        level_name = str(getattr(cfg, 'log_level', 'INFO')).upper()
        level = getattr(logging, level_name, None)
        if not isinstance(level, int):
            try:
                level = int(level_name)
            except ValueError:
                level = logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
            force=True,
        )
        logging.getLogger().setLevel(level)
        _LOGGING_CONFIGURED = True
        logger.setLevel(level)

    logger.info("========== Starting social imaging pipeline ==========")

    functional_cfg = cfg.functional_preprocessing
    motion_cfg = cfg.motion_correction
    anatomy_cfg = cfg.anatomy_preprocessing
    fireants_stage_cfg = cfg.fireants_registration
    ftoa_stage_cfg = cfg.functional_to_anatomy_registration
    processing_cfg = cfg.processing_log

    animals = list(iter_animals_with_yaml(metadata_base))
    if animal_ids is not None:
        selected = {animal_id for animal_id in animal_ids}
        animals = [animal for animal in animals if animal.animal_id in selected]
        missing = selected - {animal.animal_id for animal in animals}
        if missing:
            raise KeyError(f"Unknown animal IDs requested: {sorted(missing)}")

    ops_source = ops_path or _default_ops_template_path()
    if not ops_source.exists():
        raise FileNotFoundError(f"Suite2p ops template not found: {ops_source}")
    ops_template = motion_correction.load_global_ops(ops_source)

    fast_disk = cfg.fast_disk_base_dir or cfg.output_base_dir

    fps_value = motion_cfg.default_fps

    fireants_config: Optional[FireANTsRegistrationConfig] = None
    if fireants_stage_cfg.config_path:
        file_overrides = load_yaml_mapping(fireants_stage_cfg.config_path)
        fireants_config = FireANTsRegistrationConfig.with_overrides(file_overrides)

    results: list[AnimalResult] = []

    for animal in animals:
        animal_result = AnimalResult(animal_id=animal.animal_id)

        animal_log: Optional[AnimalProcessingLog] = None
        log_path: Optional[Path] = None
        if processing_cfg.enabled:
            log_base = cfg.output_base_dir or Path.cwd()
            log_path = build_processing_log_path(processing_cfg, animal.animal_id, base_dir=Path(log_base))
            if log_path.exists():
                animal_log = load_processing_log(log_path)
            else:
                animal_log = AnimalProcessingLog(animal_id=animal.animal_id)

        functional_session_ref: Optional[FunctionalSession] = None
        functional_session_result: Optional[SessionResult] = None
        anatomy_session_ref: Optional[AnatomySession] = None
        anatomy_session_result: Optional[SessionResult] = None

        for session in animal.sessions:
            if not session.include_in_analysis:
                skip_result = SessionResult(
                    animal_id=animal.animal_id,
                    session_id=session.session_id,
                    session_type=session.session_type,
                    status="skipped",
                    message="include_in_analysis is false",
                )
                animal_result.sessions.append(skip_result)
                if animal_log is not None and log_path is not None:
                    _update_processing_log_stage(
                        animal_log,
                        _stage_key(skip_result.session_type, skip_result.session_id),
                        skip_result,
                        {"include_in_analysis": False},
                    )
                    save_processing_log(animal_log, log_path)
                continue

            if session.session_type == "functional_stack":
                logger.info(
                    "---------- Functional preprocessing + motion (%s, session %s) ----------",
                    animal.animal_id,
                    session.session_id,
                )
                fps_value = motion_cfg.default_fps
                session_result = process_functional_session(
                    animal=animal,
                    session=session,  # type: ignore[arg-type]
                    cfg=cfg,
                    ops_template=ops_template,
                    fps_config=fps_value,
                    fast_disk=fast_disk,
                    reprocess=functional_cfg.reprocess,
                    reprocess_motion=motion_cfg.reprocess,
                    enable_preprocessing=functional_cfg.enabled,
                    enable_motion=motion_cfg.enabled,
                )
                functional_session_ref = session  # type: ignore[assignment]
                functional_session_result = session_result
                animal_result.sessions.append(session_result)
                logger.info(
                    "---------- Completed functional preprocessing + motion (%s, session %s): %s ----------",
                    animal.animal_id,
                    session.session_id,
                    session_result.status,
                )
                if animal_log is not None and log_path is not None:
                    metadata_path = session_result.outputs.get("preprocess_metadata")
                    pixel_size_xy = None
                    if metadata_path is not None:
                        pixel_size_xy = _load_session_pixel_size(metadata_path)
                    elif session_result.status != "skipped":
                        raise ValueError(
                            f"Preprocessing metadata missing for session {session.session_id}"
                        )

                    parameters = {
                        "functional_enabled": functional_cfg.enabled,
                        "functional_reprocess": functional_cfg.reprocess,
                        "motion_enabled": motion_cfg.enabled,
                        "motion_reprocess": motion_cfg.reprocess,
                        "fast_disk": fast_disk,
                        "fps_hz": fps_value,
                    }
                    if pixel_size_xy is not None:
                        parameters["session_pixel_size_xy_um"] = [float(pixel_size_xy[0]), float(pixel_size_xy[1])]

                    _update_processing_log_stage(
                        animal_log,
                        _stage_key(session_result.session_type, session_result.session_id),
                        session_result,
                        parameters,
                    )
                    save_processing_log(animal_log, log_path)
            elif (
                session.session_type == "anatomy_stack"
                and getattr(session.session_data, "stack_type", "") == "two_photon"
            ):
                logger.info(
                    "---------- Anatomy preprocessing + FireANTs (%s, session %s) ----------",
                    animal.animal_id,
                    session.session_id,
                )
                session_result = process_anatomy_session(
                    animal=animal,
                    session=session,  # type: ignore[arg-type]
                    cfg=cfg,
                    fireants_config=fireants_config,
                    fireants_overrides=fireants_stage_cfg.overrides or None,
                    reprocess_preprocessing=anatomy_cfg.reprocess,
                    reprocess_registration=fireants_stage_cfg.reprocess,
                    enable_preprocessing=anatomy_cfg.enabled,
                    enable_registration=fireants_stage_cfg.enabled,
                )
                anatomy_session_ref = session  # type: ignore[assignment]
                anatomy_session_result = session_result
                animal_result.sessions.append(session_result)
                logger.info(
                    "---------- Completed anatomy preprocessing + FireANTs (%s, session %s): %s ----------",
                    animal.animal_id,
                    session.session_id,
                    session_result.status,
                )
                if animal_log is not None and log_path is not None:
                    metadata_path = session_result.outputs.get("anatomy_preprocess_metadata")
                    pixel_size_xy = None
                    if metadata_path is not None:
                        pixel_size_xy = _load_session_pixel_size(metadata_path)
                    elif session_result.status != "skipped":
                        raise ValueError(
                            f"Anatomy preprocessing metadata missing for session {session.session_id}"
                        )

                    parameters = {
                        "anatomy_enabled": anatomy_cfg.enabled,
                        "anatomy_reprocess": anatomy_cfg.reprocess,
                        "fireants_enabled": fireants_stage_cfg.enabled,
                        "fireants_reprocess": fireants_stage_cfg.reprocess,
                    }
                    if pixel_size_xy is not None:
                        parameters["session_pixel_size_xy_um"] = [float(pixel_size_xy[0]), float(pixel_size_xy[1])]

                    _update_processing_log_stage(
                        animal_log,
                        _stage_key(session_result.session_type, session_result.session_id),
                        session_result,
                        parameters,
                    )
                    save_processing_log(animal_log, log_path)
            else:
                unsupported_result = SessionResult(
                    animal_id=animal.animal_id,
                    session_id=session.session_id,
                    session_type=session.session_type,
                    status="skipped",
                    message="unsupported session type",
                )
                animal_result.sessions.append(unsupported_result)
                if animal_log is not None and log_path is not None:
                    _update_processing_log_stage(
                        animal_log,
                        _stage_key(unsupported_result.session_type, unsupported_result.session_id),
                        unsupported_result,
                        {"reason": "unsupported session type"},
                    )
                    save_processing_log(animal_log, log_path)

        if functional_session_ref and anatomy_session_ref:
            if functional_session_result and functional_session_result.status == "failed":
                stage_result = SessionResult(
                    animal_id=animal.animal_id,
                    session_id=functional_session_ref.session_id,
                    session_type="functional_to_anatomy_registration",
                    status="skipped",
                    message="functional preprocessing failed; cannot register functional to anatomy",
                )
            elif anatomy_session_result and anatomy_session_result.status == "failed":
                stage_result = SessionResult(
                    animal_id=animal.animal_id,
                    session_id=anatomy_session_ref.session_id,
                    session_type="functional_to_anatomy_registration",
                    status="skipped",
                    message="anatomy preprocessing failed; cannot register functional to anatomy",
                )
            else:
                logger.info(
                    "---------- Functional-to-anatomy registration (%s) ----------",
                    animal.animal_id,
                )
                stage_result = process_functional_to_anatomy_registration(
                    animal=animal,
                    functional_session=functional_session_ref,  # type: ignore[arg-type]
                    anatomy_session=anatomy_session_ref,  # type: ignore[arg-type]
                    cfg=cfg,
                )
            animal_result.sessions.append(stage_result)
            logger.info(
                "---------- Completed functional-to-anatomy registration (%s): %s ----------",
                animal.animal_id,
                stage_result.status,
            )

            if animal_log is not None and log_path is not None:
                source_pixel_sizes = {}
                expected_scale_ratio: Optional[float] = None
                try:
                    functional_root = resolve_output_path(
                        animal.animal_id,
                        functional_cfg.root_subdir,
                        cfg=cfg,
                    )
                    plane_dir = functional_root / functional_cfg.planes_subdir
                    functional_metadata_name = functional_cfg.metadata_filename_template.format(
                        animal_id=animal.animal_id,
                        session_id=functional_session_ref.session_id,
                    )
                    functional_metadata_path = plane_dir / functional_metadata_name
                    functional_pixel_size = _load_session_pixel_size(functional_metadata_path)

                    anatomy_root = resolve_output_path(
                        animal.animal_id,
                        anatomy_cfg.root_subdir,
                        cfg=cfg,
                    )
                    anatomy_metadata_name = anatomy_cfg.metadata_filename_template.format(
                        animal_id=animal.animal_id,
                        session_id=anatomy_session_ref.session_id,
                    )
                    anatomy_metadata_path = anatomy_root / anatomy_metadata_name
                    anatomy_pixel_size = _load_session_pixel_size(anatomy_metadata_path)

                    source_pixel_sizes = {
                        functional_session_ref.session_id: [
                            float(functional_pixel_size[0]),
                            float(functional_pixel_size[1]),
                        ],
                        anatomy_session_ref.session_id: [
                            float(anatomy_pixel_size[0]),
                            float(anatomy_pixel_size[1]),
                        ],
                    }
                    ratio_x = functional_pixel_size[0] / anatomy_pixel_size[0]
                    ratio_y = functional_pixel_size[1] / anatomy_pixel_size[1]
                    expected_scale_ratio = float((ratio_x + ratio_y) / 2.0)
                except Exception as exc:
                    if stage_result.status != "skipped":
                        raise

                parameters = {
                    "functional_to_anatomy_enabled": ftoa_stage_cfg.enabled,
                    "functional_to_anatomy_reprocess": ftoa_stage_cfg.reprocess,
                    "scale_window": ftoa_stage_cfg.scale_window,
                    "n_scales": ftoa_stage_cfg.n_scales,
                    "z_stride_coarse": ftoa_stage_cfg.z_stride_coarse,
                    "z_refine_radius": ftoa_stage_cfg.z_refine_radius,
                    "gaussian_sigma": ftoa_stage_cfg.gaussian_sigma,
                    "early_stop_score": ftoa_stage_cfg.early_stop_score,
                    "progress": ftoa_stage_cfg.progress,
                }
                if source_pixel_sizes:
                    parameters["source_session_pixel_sizes_xy_um"] = source_pixel_sizes
                if expected_scale_ratio is not None:
                    parameters["expected_scale_ratio"] = expected_scale_ratio
                    scale_window = max(0.0, float(ftoa_stage_cfg.scale_window))
                    min_scale = expected_scale_ratio * (1.0 - scale_window)
                    max_scale = expected_scale_ratio * (1.0 + scale_window)
                    if ftoa_stage_cfg.n_scales == 1:
                        min_scale = max_scale = expected_scale_ratio
                    min_scale = max(min_scale, 1e-6)
                    max_scale = max(max_scale, min_scale)
                    parameters["scale_range_min"] = float(min_scale)
                    parameters["scale_range_max"] = float(max_scale)

                _update_processing_log_stage(
                    animal_log,
                    _stage_key(stage_result.session_type, stage_result.session_id),
                    stage_result,
                    parameters,
                )
                save_processing_log(animal_log, log_path)
        elif ftoa_stage_cfg.enabled:
            session_id = (
                functional_session_ref.session_id
                if functional_session_ref
                else (
                    anatomy_session_ref.session_id
                    if anatomy_session_ref
                    else "functional_to_anatomy"
                )
            )
            missing_result = SessionResult(
                animal_id=animal.animal_id,
                session_id=session_id,
                session_type="functional_to_anatomy_registration",
                status="skipped",
                message="functional-to-anatomy registration skipped (missing required sessions)",
            )
            animal_result.sessions.append(missing_result)
            if animal_log is not None and log_path is not None:
                _update_processing_log_stage(
                    animal_log,
                    _stage_key("functional_to_anatomy_registration", session_id),
                    missing_result,
                    {
                        "functional_to_anatomy_enabled": ftoa_stage_cfg.enabled,
                        "functional_to_anatomy_reprocess": ftoa_stage_cfg.reprocess,
                    },
                )
                save_processing_log(animal_log, log_path)

        results.append(animal_result)

        if animal_log is not None and log_path is not None:
            save_processing_log(animal_log, log_path)

    if ftoa_stage_cfg.enabled and cfg.output_base_dir:
        # Persist combined registration quality logs so downstream analysis can reference a single table.
        try:
            base_folder = Path(cfg.output_base_dir)
            summary_csv_path = resolve_output_path(
                ftoa_stage_cfg.summary_csv, cfg=cfg
            )
            summary_full_path = resolve_output_path(
                ftoa_stage_cfg.summary_full_csv, cfg=cfg
            )
            summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
            summary_full_path.parent.mkdir(parents=True, exist_ok=True)
            align_substack.write_combined_log(
                base_folder,
                out_csv=summary_csv_path,
                out_full_csv=summary_full_path,
            )
        except Exception as exc:  # pragma: no cover - aggregation best effort
            logger.warning(
                "Failed to write functional-to-anatomy combined registration log: %s",
                exc,
            )

    return PipelineResult(animals=results)
