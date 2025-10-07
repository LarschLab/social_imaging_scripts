from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


REFERENCE_BRAIN_RELATIVE = Path(
    "03_Common_Use/reference brains/ref_05_LB_Perrino_2p/average_2p_noRot_orig.nrrd"
)


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


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Base directories
    raw_base_dir: Optional[Path] = Field(default=None, description="Base path for raw inputs (e.g., network share)")
    output_base_dir: Optional[Path] = Field(default=None, description="Base path for large outputs (local fast disk)")

    @field_validator("raw_base_dir", "output_base_dir", mode="before")
    @classmethod
    def _normalise_base(cls, value):
        return normalise_pathlike(value)

    @staticmethod
    def default_locations() -> list[Path]:
        return [
            Path("metadata/project.yaml"),
            Path("metadata/project.yml"),
            Path("project.yaml"),
        ]


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


def _default_base_dirs() -> tuple[Optional[Path], Optional[Path]]:
    if os.name == "nt":
        return Path("Y:/"), Path(r"D:\\social_imaging_outputs")
    if _running_in_wsl():
        return Path("/mnt/nas_jlarsch"), Path("/mnt/f/johannes/testoutput")
    return None, None


def load_project_config(path: Optional[Path] = None) -> ProjectConfig:
    """Load project config from YAML if present, overridden by env vars.

    Env overrides:
      - SOCIAL_IMG_RAW_BASE
      - SOCIAL_IMG_OUT_BASE
    """
    raw = os.getenv("SOCIAL_IMG_RAW_BASE")
    out = os.getenv("SOCIAL_IMG_OUT_BASE")

    data = {}
    if path is None:
        for cand in ProjectConfig.default_locations():
            if cand.exists():
                path = cand
                break

    if path is not None and Path(path).exists():
        try:
            from ruamel.yaml import YAML

            y = YAML(typ="safe")
            data = y.load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    if raw:
        data["raw_base_dir"] = raw
    if out:
        data["output_base_dir"] = out

    defaults = _default_base_dirs()
    if defaults[0] is not None and not data.get("raw_base_dir"):
        data["raw_base_dir"] = defaults[0]
    if defaults[1] is not None and not data.get("output_base_dir"):
        data["output_base_dir"] = defaults[1]

    return ProjectConfig.model_validate(data)


def resolve_raw_path(p: Path, cfg: Optional[ProjectConfig] = None) -> Path:
    p = normalise_pathlike(p) or Path(p)
    if p.is_absolute():
        return p
    cfg = cfg or load_project_config()
    if cfg.raw_base_dir is None:
        return p
    return Path(cfg.raw_base_dir) / p


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
    return resolve_raw_path(REFERENCE_BRAIN_RELATIVE, cfg=cfg)
