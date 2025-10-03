"""Metadata domain models for social imaging pipelines.

The models describe the declarative metadata we expect to collect before
running the processing pipeline. The schema stays concise so users can edit it
in spreadsheets, YAML, or internal tools while retaining strict validation for
code.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .config import normalise_pathlike


class MetaSource(BaseModel):
    """Pointer to an external metadata source such as a lab log or spreadsheet."""

    type: Literal["lab_record", "experiment_log", "other"] = "other"
    path: str


class ChannelInfo(BaseModel):
    """Describes one imaging channel (microscope wavelength / marker pairing)."""

    channel_id: int
    name: str
    marker: Optional[str] = None
    wavelength_nm: Optional[float] = None
    round_id: Optional[int] = None


class TwoPhotonPreprocessing(BaseModel):
    """Acquisition parameters required to preprocess two-photon stacks."""

    mode: Literal["resonant", "linear"]
    n_planes: Optional[int] = None
    frames_per_plane: int
    flyback_frames: int = 0
    remove_first_frame: bool = False
    blocks: Optional[list[int]] = None


class FunctionalSessionData(BaseModel):
    """Functional two-photon acquisition exported by Suite2p preprocessing."""

    raw_path: Path
    stimulus_name: Optional[str] = None
    stimulus_metadata_path: Optional[Path] = None
    zoom_factor: Optional[float] = None
    channels: list[ChannelInfo] = Field(default_factory=list)
    preprocessing_two_photon: Optional[TwoPhotonPreprocessing] = None

    @field_validator("raw_path", mode="before")
    @classmethod
    def _normalise_raw_path(cls, value):
        return normalise_pathlike(value)

    @field_validator("stimulus_metadata_path", mode="before")
    @classmethod
    def _normalise_stimulus_metadata(cls, value):
        return normalise_pathlike(value)


class AnatomySessionData(BaseModel):
    """Any anatomy stack (two-photon or confocal) captured for a session."""

    raw_path: Path
    round_id: Optional[int] = None
    plane_spacing: Optional[float] = None
    stack_type: Literal["two_photon", "confocal"] = "confocal"
    channels: list[ChannelInfo] = Field(default_factory=list)
    preprocessing_two_photon: Optional[TwoPhotonPreprocessing] = None
    pixel_size_xy_um: Optional[tuple[float, float]] = None
    z_step_um: Optional[float] = None

    @field_validator("raw_path", mode="before")
    @classmethod
    def _normalise_raw_path(cls, value):
        return normalise_pathlike(value)


class SessionCommon(BaseModel):
    """Fields shared by every session irrespective of imaging modality."""

    session_id: str
    date: date
    condition: Optional[str] = None
    experimenter: Optional[str] = None
    include_in_analysis: bool = True
    image_quality: Optional[str] = None
    notes: Optional[str] = None


class FunctionalSession(SessionCommon):
    """Structured session whose payload is a functional stack."""

    session_type: Literal["functional_stack"]
    session_data: FunctionalSessionData


class AnatomySession(SessionCommon):
    """Structured session whose payload is an anatomy stack (2p or confocal)."""

    session_type: Literal["anatomy_stack"]
    session_data: AnatomySessionData


class BehaviorSession(SessionCommon):
    """Placeholder session for behaviour experiments tracked alongside imaging."""

    session_type: Literal["behavior"]
    session_data: dict = Field(default_factory=dict)


class OtherSession(SessionCommon):
    """Fallback session for modalities we do not yet model explicitly."""

    session_type: Literal["other"]
    session_data: dict = Field(default_factory=dict)


Session = Annotated[
    Union[FunctionalSession, AnatomySession, BehaviorSession, OtherSession],
    Field(discriminator="session_type"),
]


class AvailableModalities(BaseModel):
    """Flags summarising which types of data exist for an animal."""

    functional_2p: bool = False
    anatomy_2p: bool = False
    confocal_rounds: int = 0
    behavior: bool = False


class AnimalMetadata(BaseModel):
    """All metadata that can be known before running the processing pipeline."""

    animal_id: str
    genotype: Optional[str] = None
    owner: Optional[str] = None
    tank: Optional[str] = None
    date_of_birth: Optional[date] = None
    meta_sources: list[MetaSource] = Field(default_factory=list)
    root_dir: Optional[Path] = None
    available_modalities: AvailableModalities = Field(default_factory=AvailableModalities)
    sessions: list[Session] = Field(default_factory=list)

    @field_validator("root_dir", mode="before")
    @classmethod
    def _normalise_root_dir(cls, value):
        return normalise_pathlike(value)


class MetadataCollection(BaseModel):
    """Utility wrapper for a group of animals loaded from disk."""

    animals: list[AnimalMetadata]

    def by_id(self, animal_id: str) -> AnimalMetadata:
        """Return the metadata for one animal or raise if the ID is unknown."""

        for item in self.animals:
            if item.animal_id == animal_id:
                return item
        raise KeyError(f"Animal {animal_id} not found")
