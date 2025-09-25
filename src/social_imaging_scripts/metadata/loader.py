from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ruamel.yaml import YAML

from .models import AnimalMetadata, MetadataCollection

_yaml = YAML(typ="safe")


def load_animal_file(path: Path) -> AnimalMetadata:
    data = _yaml.load(path.read_text(encoding="utf-8"))
    return AnimalMetadata.model_validate(data)


def load_animals(paths: Iterable[Path] | None = None, base_dir: Path | None = None) -> MetadataCollection:
    if paths is None:
        if base_dir is None:
            base_dir = Path("metadata/animals")
        paths = sorted(base_dir.glob("*.yaml"))
    animals = [load_animal_file(Path(p)) for p in paths]
    return MetadataCollection(animals=animals)
