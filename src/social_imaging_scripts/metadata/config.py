from __future__ import annotations

import os
import platform
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..registration.fireants_pipeline import FireANTsRegistrationConfig


_DEFAULT_CONFIG_FILENAMES = (
    "metadata/pipeline_defaults.yaml",
    "metadata/pipeline_defaults.yml",
)
_PROJECT_OVERRIDE_FILENAMES = (
    "metadata/project.yaml",
    "metadata/project.yml",
    "project.yaml",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _running_in_wsl() -> bool:
    if os.name == "nt":
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8")
    except OSError:
        release = platform.uname().release
    return "microsoft" in release.lower()


def normalise_pathlike(value: str | Path | None) -> Optional[Path]:
    """Convert Windows-style paths to POSIX when running on WSL."""

    if value is None:
        return None

    text = str(value).strip()
    if text == "":
        return Path(text)

    is_wsl = _running_in_wsl()

    # Windows drive letter (e.g., D:\data)
    if is_wsl and len(text) >= 2 and text[1] == ":":
        drive = text[0].lower()
        remainder = text[2:].lstrip("\\/")
        base = Path("/mnt") / drive
        if remainder:
            for part in remainder.replace("\\", "/").split("/"):
                if part:
                    base /= part
        return base

    # UNC share (e.g., \\SERVER\share)
    if is_wsl and text.startswith("\\\\"):
        trimmed = text.lstrip("\\")
        parts = [seg for seg in trimmed.split("\\") if seg]
        if len(parts) >= 2:
            server, *rest = parts
            override = os.getenv(f"SOCIAL_IMG_UNC_{server.upper()}")
            candidates: list[Path] = []
            if override:
                candidates.append(Path(override))
            candidates.extend(
                [
                    Path("/mnt") / server,
                    Path("/mnt") / server.lower(),
                    Path("/mnt") / server.upper(),
                ]
            )
            for root in candidates:
                if root.exists():
                    base = root
                    break
            else:
                base = Path(f"//{server}")
            for part in rest:
                base /= part
            return base
        return Path("//" + trimmed.replace("\\", "/"))

    if "\\" in text:
        text = text.replace("\\", "/")

    return Path(text)


def _resolve_candidate_path(name: str | Path) -> Optional[Path]:
    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    repo_candidate = _repo_root() / candidate
    if repo_candidate.exists():
        return repo_candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return None


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping document."""

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    data = yaml.load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping at {path}, found {type(data).__name__}")
    return dict(data)


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], Mapping)
            and isinstance(value, Mapping)
        ):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = value
    return base


class StageMode(str, Enum):
    """Controls whether a pipeline stage runs, reuses, or forces outputs."""

    SKIP = "skip"
    REUSE = "reuse"
    FORCE = "force"


class FunctionalPreprocessingConfig(BaseModel):
    """Configuration for functional plane splitting and basic preprocessing."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    root_subdir: Path = Field(
        ...,
        description="Relative output path for functional preprocessing artefacts.",
    )
    planes_subdir: Path = Field(
        ...,
        description="Subdirectory containing per-plane TIFF outputs.",
    )
    plane_filename_template: str = Field(
        ...,
        description="Template for preprocessed plane TIFF filenames.",
    )
    metadata_filename_template: str = Field(
        ...,
        description="Template for preprocessing metadata filenames.",
    )

    @field_validator("root_subdir", "planes_subdir", mode="before")
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class MotionCorrectionConfig(BaseModel):
    """Configuration for Suite2p motion correction and projections."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    planes_subdir: Path = Field(
        ...,
        description="Location of per-plane TIFFs relative to the functional root.",
    )
    motion_output_subdir: Path = Field(
        ...,
        description="Subdirectory for Suite2p motion-corrected outputs.",
    )
    plane_folder_template: str = Field(
        ...,
        description="Template for per-plane folder names under motion outputs.",
    )
    projections_subdir: Path = Field(
        ...,
        description="Subdirectory used to store max/avg projection stacks.",
    )
    suite2p_output_subdir: Path = Field(
        ...,
        description="Subdirectory containing Suite2p segmentation artefacts.",
    )
    segmentation_folder_template: str = Field(
        ...,
        description="Template for per-plane Suite2p segmentation folders.",
    )
    motion_filename_template: str = Field(
        ...,
        description="Filename template for motion-corrected TIFF outputs.",
    )
    metadata_filename: str = Field(
        ...,
        description="Filename for per-plane motion metadata.",
    )
    default_fps: float = Field(
        ...,
        description="Frame rate (Hz) used for Suite2p runs.",
    )

    @field_validator(
        "planes_subdir",
        "motion_output_subdir",
        "projections_subdir",
        "suite2p_output_subdir",
        mode="before",
    )
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class AnatomyPreprocessingConfig(BaseModel):
    """Configuration for two-photon anatomy preprocessing."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    root_subdir: Path = Field(
        ...,
        description="Relative output path for anatomy preprocessing artefacts.",
    )
    metadata_filename_template: str = Field(
        ...,
        description="Template for anatomy preprocessing metadata filenames.",
    )
    stack_filename_template: str = Field(
        ...,
        description="Template for the generated anatomy stack filename.",
    )

    @field_validator("root_subdir", mode="before")
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class ConfocalPreprocessingConfig(BaseModel):
    """Configuration for confocal stack preprocessing."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    root_subdir: Path = Field(
        ...,
        description="Relative output directory for confocal preprocessing artefacts.",
    )
    channel_filename_template: str = Field(
        ...,
        description="Filename pattern for per-channel confocal stacks.",
    )
    metadata_filename_template: str = Field(
        ...,
        description="Filename template for confocal preprocessing metadata.",
    )
    flip_horizontal: bool = Field(
        ...,
        description="Flip confocal stacks left-right during preprocessing.",
    )
    flip_z: bool = Field(
        ...,
        description="Reverse confocal stack order along the axial (Z) dimension.",
    )

    @field_validator("root_subdir", mode="before")
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class FireantsRegistrationStageConfig(BaseModel):
    """Configuration for FireANTs anatomy registration stage."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    output_subdir: Path = Field(
        ...,
        description="Relative output path for FireANTs outputs.",
    )
    warped_stack_template: str = Field(
        ...,
        description="Filename template for the warped anatomy stack.",
    )
    fireants: FireANTsRegistrationConfig = Field(
        ...,
        description="FireANTs configuration for this stage.",
    )

    @field_validator("output_subdir", mode="before")
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class ConfocalToAnatomyRegistrationConfig(BaseModel):
    """Configuration for registering confocal stacks to two-photon anatomy."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    output_root_subdir: Path = Field(
        ...,
        description="Base output directory for confocal registration artefacts.",
    )
    registration_subdir_template: str = Field(
        ...,
        description="Template for per-session registration subdirectories.",
    )
    reference_channel_name: str = Field(
        ...,
        description="Channel name used as the moving image during registration.",
    )
    warped_channel_template: str = Field(
        ...,
        description="Filename template for warped channel outputs.",
    )
    metadata_filename_template: str = Field(
        ...,
        description="Filename template for registration metadata JSON.",
    )
    transforms_subdir: Path = Field(
        ...,
        description="Subdirectory name for saved FireANTs transforms.",
    )
    qc_subdir: Path = Field(
        ...,
        description="Subdirectory name for QC artefacts.",
    )
    mask_margin_xy: float = Field(
        default=0.08,
        description="Fraction (<=1) or absolute voxel margin to crop laterally before registration.",
    )
    mask_margin_z: float = Field(
        default=0.05,
        description="Fraction (<=1) or absolute voxel margin to crop axially before registration.",
    )
    mask_soft_edges: bool = Field(
        default=True,
        description="Use a smooth taper around mask margins instead of a hard cutoff.",
    )
    histogram_match: bool = Field(
        default=True,
        description="Apply histogram matching between confocal and anatomy reference volumes.",
    )
    histogram_levels: int = Field(
        default=64,
        description="Histogram levels used during intensity histogram matching.",
    )
    histogram_match_points: int = Field(
        default=12,
        description="Number of match points used for histogram matching.",
    )
    histogram_threshold_at_mean: bool = Field(
        default=True,
        description="Threshold histogram matching at the mean intensity (SimpleITK behaviour).",
    )
    initial_translation_mode: Literal["none", "crop", "centroid"] = Field(
        default="centroid",
        description="Seed affine translation using the crop offsets or centroid difference (none|crop|centroid).",
    )
    crop_to_extent: bool = Field(
        default=True,
        description="Crop confocal XY extent to approximate the anatomy field of view.",
    )
    crop_padding_um: float = Field(
        default=0.0,
        description="Extra padding (µm) preserved around the cropped confocal volume.",
    )
    fireants: FireANTsRegistrationConfig = Field(
        ...,
        description="FireANTs configuration for this stage.",
    )

    @field_validator(
        "output_root_subdir",
        "transforms_subdir",
        "qc_subdir",
        mode="before",
    )
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class FunctionalToAnatomyRegistrationConfig(BaseModel):
    """Configuration for registering functional projections to anatomy stacks."""

    model_config = ConfigDict(extra="ignore")

    mode: StageMode = Field(..., description="Stage execution mode.")
    registration_output_subdir: Path = Field(
        ...,
        description="Relative output path for functional-to-anatomy registration artefacts.",
    )
    projections_subdir: Path = Field(
        ...,
        description="Subdirectory under the motion outputs for projection stacks.",
    )
    summary_csv: Path = Field(
        ...,
        description="Relative path for the aggregated registration summary CSV.",
    )
    summary_full_csv: Path = Field(
        ...,
        description="Relative path for the detailed registration log CSV.",
    )
    registration_csv_template: str = Field(
        ...,
        description="Filename template for per-animal registration CSV outputs.",
    )
    max_projection_filename_template: str = Field(
        ...,
        description="Filename template for saved maximum projection stacks.",
    )
    avg_projection_filename_template: str = Field(
        ...,
        description="Filename template for saved average projection stacks.",
    )
    fixed_z_spacing_um: float = Field(
        ...,
        description="Axial spacing (um) for the fixed anatomy stack.",
    )
    scale_window: float = Field(
        ...,
        description="Fractional window around the expected scale (0 ⇒ single scale).",
    )
    n_scales: int = Field(
        ...,
        description="Number of discrete scales evaluated during search.",
    )
    z_stride_coarse: int = Field(
        ...,
        description="Coarse stride (planes) for initial z search.",
    )
    z_refine_radius: int = Field(
        ...,
        description="Number of planes on each side sampled during z refinement.",
    )
    gaussian_sigma: float = Field(
        ...,
        description="Gaussian sigma used during preprocessing.",
    )
    early_stop_score: Optional[float] = Field(
        default=None,
        description="Optional NCC threshold for terminating the search early.",
    )

    @field_validator(
        "registration_output_subdir",
        "projections_subdir",
        "summary_csv",
        "summary_full_csv",
        mode="before",
    )
    @classmethod
    def _normalise_paths(cls, value):
        return normalise_pathlike(value)


class ProcessingLogConfig(BaseModel):
    """Configuration for generated per-animal processing artefact logs."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(..., description="Write per-animal processing logs.")
    directory: Path = Field(
        ...,
        description="Directory where per-animal processing logs are written.",
    )
    filename_template: str = Field(
        ...,
        description="Filename template for per-animal processing logs.",
    )

    @field_validator("directory", mode="before")
    @classmethod
    def _normalise_path(cls, value):
        return normalise_pathlike(value)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    apply_log_settings: bool = Field(
        default=False,
        description="Configure logging automatically when running the pipeline.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level to apply when configuring logging.",
    )
    raw_base_dir: Optional[Path] = Field(
        default=None, description="Base path for raw inputs (e.g., network share)."
    )
    output_base_dir: Optional[Path] = Field(
        default=None, description="Base path for processed outputs."
    )
    fast_disk_base_dir: Optional[Path] = Field(
        default=None,
        description="Optional fast disk scratch path for Suite2p outputs.",
    )
    ref_base_dir: Optional[Path] = Field(
        default=None,
        description="Base path containing reference volumes.",
    )
    reference_brain_filename: Path = Field(
        default=Path("reference brains/ref_05_LB_Perrino_2p/average_2p_noRot_orig.nrrd"),
        description="Path relative to ref_base_dir for the reference brain NRRD.",
    )

    functional_preprocessing: FunctionalPreprocessingConfig = Field(...)
    motion_correction: MotionCorrectionConfig = Field(...)
    anatomy_preprocessing: AnatomyPreprocessingConfig = Field(...)
    confocal_preprocessing: ConfocalPreprocessingConfig = Field(...)
    fireants_registration: FireantsRegistrationStageConfig = Field(...)
    confocal_to_anatomy_registration: ConfocalToAnatomyRegistrationConfig = Field(...)
    functional_to_anatomy_registration: FunctionalToAnatomyRegistrationConfig = Field(
        ...
    )
    processing_log: ProcessingLogConfig = Field(...)

    @field_validator("raw_base_dir", "output_base_dir", "ref_base_dir", mode="before")
    @classmethod
    def _normalise_base(cls, value):
        return normalise_pathlike(value)

    @field_validator("fast_disk_base_dir", mode="before")
    @classmethod
    def _normalise_fast_disk(cls, value):
        return normalise_pathlike(value)

    @staticmethod
    def default_locations() -> list[Path]:
        return [Path(name) for name in _PROJECT_OVERRIDE_FILENAMES]


def load_project_config(path: Optional[Path] = None) -> ProjectConfig:
    """Load project configuration from defaults, overrides, and environment."""

    data: Dict[str, Any] = {}

    default_loaded = False
    for filename in _DEFAULT_CONFIG_FILENAMES:
        candidate = _resolve_candidate_path(filename)
        if candidate:
            data = _deep_update(data, load_yaml_mapping(candidate))
            default_loaded = True
            break
    if not default_loaded:
        raise FileNotFoundError(
            "Unable to locate pipeline defaults YAML. Expected one of: "
            + ", ".join(str(_repo_root() / name) for name in _DEFAULT_CONFIG_FILENAMES)
        )

    override_path: Optional[Path] = None
    if path is not None:
        override_path = Path(path)
    else:
        for candidate in ProjectConfig.default_locations():
            resolved = _resolve_candidate_path(candidate)
            if resolved:
                override_path = resolved
                break

    if override_path is not None and override_path.exists():
        data = _deep_update(data, load_yaml_mapping(override_path))

    raw_env = os.getenv("SOCIAL_IMG_RAW_BASE")
    out_env = os.getenv("SOCIAL_IMG_OUT_BASE")

    if raw_env:
        data["raw_base_dir"] = raw_env
    if out_env:
        data["output_base_dir"] = out_env

    return ProjectConfig.model_validate(data)



def resolve_raw_path(p: Path, cfg: Optional[ProjectConfig] = None) -> Path:
    p = normalise_pathlike(p) or Path(p)
    if p.is_absolute():
        return p
    cfg = cfg or load_project_config()
    if cfg.raw_base_dir is not None:
        return Path(cfg.raw_base_dir) / p
    return p


def resolve_output_path(*parts: str | Path, cfg: Optional[ProjectConfig] = None) -> Path:
    cfg = cfg or load_project_config()
    base = cfg.output_base_dir or Path.cwd()
    base = Path(base)
    cleaned_parts: list[Path] = []
    for part in parts:
        if part is None:
            continue
        candidate = normalise_pathlike(part)
        cleaned_parts.append(candidate if candidate is not None else Path(part))
    return base.joinpath(*cleaned_parts)


def resolve_raw_subpath(*parts: str | Path, cfg: Optional[ProjectConfig] = None) -> Path:
    relative = Path(".")
    for part in parts:
        if part is None:
            continue
        candidate = normalise_pathlike(part)
        candidate = candidate if candidate is not None else Path(part)
        if candidate.is_absolute():
            relative = candidate
        else:
            relative = relative.joinpath(candidate)
    return resolve_raw_path(relative, cfg=cfg)


def resolve_reference_brain(cfg: Optional[ProjectConfig] = None) -> Path:
    cfg = cfg or load_project_config()
    base = cfg.ref_base_dir or cfg.raw_base_dir or Path.cwd()
    base = normalise_pathlike(base) or Path(base)
    relative = cfg.reference_brain_filename
    relative = normalise_pathlike(relative) or Path(relative)
    return Path(base) / relative
