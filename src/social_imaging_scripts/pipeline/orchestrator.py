"""High-level orchestration helpers for multi-animal processing runs."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Mapping, Optional

from ..metadata.config import (
    ProjectConfig,
    load_project_config,
    resolve_output_path,
    resolve_raw_path,
    resolve_reference_brain,
)
from ..metadata.loader import load_animals
from ..metadata.models import AnatomySession, AnimalMetadata, FunctionalSession
from ..preprocessing.two_photon import anatomy as anatomy_preproc
from ..preprocessing.two_photon import functional as functional_preproc
from ..preprocessing.two_photon import motion as motion_correction
from ..registration.fireants_pipeline import FireANTsRegistrationConfig, register_two_photon_anatomy

logger = logging.getLogger(__name__)

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
    fps_config: Optional[
        float
        | Mapping[str, float]
        | Callable[[AnimalMetadata, FunctionalSession], Optional[float]]
    ],
    animal: AnimalMetadata,
    session: FunctionalSession,
) -> float:
    if fps_config is None:
        return 2.0
    if isinstance(fps_config, Mapping):
        value = fps_config.get(session.session_id)
        if value is not None:
            return value
        value = fps_config.get(animal.animal_id)
        if value is not None:
            return value
        if "default" in fps_config:
            return fps_config["default"]
        raise KeyError(
            f"No frame-rate entry for session '{session.session_id}' or animal '{animal.animal_id}'"
        )
    if callable(fps_config):
        value = fps_config(animal, session)
        if value is None:
            raise ValueError(
                f"Frame-rate callback returned None for session {session.session_id}"
            )
        return float(value)
    return float(fps_config)


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
        "02_reg",
        "00_preprocessing",
        "2p_functional",
        session.session_id,
        cfg=cfg,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = output_root / "01_individualPlanes" / f"{animal.animal_id}_preprocessing_metadata.json"
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

    plane_dir = output_root / "01_individualPlanes"
    plane_paths = (
        sorted(plane_dir.glob(f"{animal.animal_id}_plane*.tif"))
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

        for plane_path in plane_paths:
            plane_idx = _extract_plane_index(plane_path)
            try:
                # Run Suite2p plane-by-plane so outputs match the legacy project layout.
                motion_outputs = motion_correction.run_motion_correction(
                    animal=animal,
                    plane_idx=plane_idx,
                    plane_tiff=plane_path,
                    ops_template=ops_template,
                    fps=fps,
                    output_root=output_root,
                    fast_disk=fast_disk,
                    reprocess=reprocess_motion,
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
        "02_reg",
        "00_preprocessing",
        "2p_anatomy",
        session.session_id,
        cfg=cfg,
    )
    preprocess_root.mkdir(parents=True, exist_ok=True)

    preprocess_metadata = preprocess_root / f"{animal.animal_id}_anatomy_metadata.json"
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
                )
                stack_path = Path(preproc_outputs.get("stack", preprocess_root / f"{animal.animal_id}_anatomy_stack.tif"))
                result.outputs.update(
                    {f"anatomy_preprocess_{key}": Path(value) for key, value in preproc_outputs.items()}
                )
                preproc_generated = True
            else:
                stack_path = preprocess_root / f"{animal.animal_id}_anatomy_stack.tif"
                if preprocess_metadata.exists():
                    result.outputs["anatomy_preprocess_metadata"] = preprocess_metadata
        except Exception as exc:  # pragma: no cover - external IO
            logger.exception("Anatomy preprocessing failed", exc_info=exc)
            result.message = f"anatomy preprocessing failed: {exc}"
            return result
    else:
        stack_candidate = preprocess_root / f"{animal.animal_id}_anatomy_stack.tif"
        if stack_candidate.exists():
            stack_path = stack_candidate
            if preprocess_metadata.exists():
                result.outputs["anatomy_preprocess_metadata"] = preprocess_metadata

    if stack_path is None or not stack_path.exists():
        result.message = f"anatomy stack not found at {stack_path}"
        return result

    registration_root = resolve_output_path(
        animal.animal_id,
        "02_reg",
        "02_fireants",
        session.session_id,
        cfg=cfg,
    )
    registration_root.mkdir(parents=True, exist_ok=True)

    warped_stack = registration_root / f"{animal.animal_id}_anatomy_warped_fireants.tif"
    registration_generated = False

    if enable_registration:
        registration_needed = reprocess_registration or not warped_stack.exists()

        if registration_needed:
            config = _prepare_fireants_config(fireants_config, fireants_overrides)
            reference_path = resolve_reference_brain(cfg=cfg)

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


def run_pipeline(
    animal_ids: Optional[Iterable[str]] = None,
    *,
    metadata_base: Optional[Path] = None,
    cfg: Optional[ProjectConfig] = None,
    ops_path: Optional[Path] = None,
    fast_disk: Optional[Path] = None,
    functional_fps: Optional[
        float
        | Mapping[str, float]
        | Callable[[AnimalMetadata, FunctionalSession], Optional[float]]
    ] = None,
    reprocess_functional: bool = False,
    reprocess_motion: bool = False,
    reprocess_anatomy_pre: bool = False,
    reprocess_anatomy_registration: bool = False,
    run_functional_preprocessing: bool = True,
    run_motion_correction: bool = True,
    run_anatomy_preprocessing: bool = True,
    run_fireants_registration: bool = True,
    fireants_config: Optional[FireANTsRegistrationConfig] = None,
    fireants_overrides: Optional[dict] = None,
) -> PipelineResult:
    cfg = cfg or load_project_config()

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

    if fast_disk is None:
        fast_disk = cfg.output_base_dir

    results: list[AnimalResult] = []

    for animal in animals:
        animal_result = AnimalResult(animal_id=animal.animal_id)
        for session in animal.sessions:
            if not session.include_in_analysis:
                animal_result.sessions.append(
                    SessionResult(
                        animal_id=animal.animal_id,
                        session_id=session.session_id,
                        session_type=session.session_type,
                        status="skipped",
                        message="include_in_analysis is false",
                    )
                )
                continue

            if session.session_type == "functional_stack":
                # Functional stacks: plane splitting + Suite2p motion correction.
                session_result = process_functional_session(
                    animal=animal,
                    session=session,  # type: ignore[arg-type]
                    cfg=cfg,
                    ops_template=ops_template,
                    fps_config=functional_fps,
                    fast_disk=fast_disk,
                    reprocess=reprocess_functional,
                    reprocess_motion=reprocess_motion,
                    enable_preprocessing=run_functional_preprocessing,
                    enable_motion=run_motion_correction,
                )
                animal_result.sessions.append(session_result)
            elif (
                session.session_type == "anatomy_stack"
                and getattr(session.session_data, "stack_type", "") == "two_photon"
            ):
                # Two-photon anatomy stacks: preprocess then register via FireANTs.
                session_result = process_anatomy_session(
                    animal=animal,
                    session=session,  # type: ignore[arg-type]
                    cfg=cfg,
                    fireants_config=fireants_config,
                    fireants_overrides=fireants_overrides,
                    reprocess_preprocessing=reprocess_anatomy_pre,
                    reprocess_registration=reprocess_anatomy_registration,
                    enable_preprocessing=run_anatomy_preprocessing,
                    enable_registration=run_fireants_registration,
                )
                animal_result.sessions.append(session_result)
            else:
                animal_result.sessions.append(
                    SessionResult(
                        animal_id=animal.animal_id,
                        session_id=session.session_id,
                        session_type=session.session_type,
                        status="skipped",
                        message="unsupported session type",
                    )
                )
        results.append(animal_result)

    return PipelineResult(animals=results)
