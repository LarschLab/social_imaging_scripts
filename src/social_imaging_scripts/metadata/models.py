from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class MetaSource(BaseModel):
    type: Literal["lab_record", "experiment_log", "other"] = "other"
    path: str


class ChannelInfo(BaseModel):
    channel_id: int
    name: str
    marker: Optional[str] = None
    wavelength_nm: Optional[float] = None
    round_id: Optional[int] = None


class FunctionalSessionData(BaseModel):
    raw_path: Path
    stimulus_name: Optional[str] = None
    stimulus_metadata_path: Optional[Path] = None
    microscope_settings_path: Optional[Path] = None
    num_planes: Optional[int] = None
    zoom_factor: Optional[float] = None
    channels: list[ChannelInfo] = Field(default_factory=list)


class AnatomySessionData(BaseModel):
    raw_path: Path
    round_id: Optional[int] = None
    plane_spacing: Optional[float] = None
    microscope_settings_path: Optional[Path] = None
    channels: list[ChannelInfo] = Field(default_factory=list)


class SessionCommon(BaseModel):
    session_id: str
    date: date
    condition: Optional[str] = None
    experimenter: Optional[str] = None
    include_in_analysis: bool = True
    image_quality: Optional[str] = None
    notes: Optional[str] = None


class FunctionalSession(SessionCommon):
    session_type: Literal["functional_stack"]
    session_data: FunctionalSessionData


class AnatomySession(SessionCommon):
    session_type: Literal["anatomy_stack"]
    session_data: AnatomySessionData


class BehaviorSession(SessionCommon):
    session_type: Literal["behavior"]
    session_data: dict = Field(default_factory=dict)


class OtherSession(SessionCommon):
    session_type: Literal["other"]
    session_data: dict = Field(default_factory=dict)


Session = Annotated[
    Union[FunctionalSession, AnatomySession, BehaviorSession, OtherSession],
    Field(discriminator="session_type"),
]


class AvailableModalities(BaseModel):
    functional_2p: bool = False
    anatomy_2p: bool = False
    confocal_rounds: int = 0
    behavior: bool = False


class AnimalMetadata(BaseModel):
    animal_id: str
    genotype: Optional[str] = None
    owner: Optional[str] = None
    tank: Optional[str] = None
    date_of_birth: Optional[date] = None
    meta_sources: list[MetaSource] = Field(default_factory=list)
    root_dir: Optional[Path] = None
    available_modalities: AvailableModalities = Field(default_factory=AvailableModalities)
    sessions: list[Session] = Field(default_factory=list)


class MetadataCollection(BaseModel):
    animals: list[AnimalMetadata]

    def by_id(self, animal_id: str) -> AnimalMetadata:
        for item in self.animals:
            if item.animal_id == animal_id:
                return item
        raise KeyError(f"Animal {animal_id} not found")
