from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Base directories
    raw_base_dir: Optional[Path] = Field(default=None, description="Base path for raw inputs (e.g., network share)")
    output_base_dir: Optional[Path] = Field(default=None, description="Base path for large outputs (local fast disk)")

    @staticmethod
    def default_locations() -> list[Path]:
        return [
            Path("metadata/project.yaml"),
            Path("metadata/project.yml"),
            Path("project.yaml"),
        ]


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

    return ProjectConfig.model_validate(data)


def resolve_raw_path(p: Path, cfg: Optional[ProjectConfig] = None) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    cfg = cfg or load_project_config()
    if cfg.raw_base_dir is None:
        return p
    return Path(cfg.raw_base_dir) / p


def resolve_output_path(*parts: str | Path, cfg: Optional[ProjectConfig] = None) -> Path:
    cfg = cfg or load_project_config()
    base = cfg.output_base_dir or Path.cwd()
    return Path(base).joinpath(*parts)

