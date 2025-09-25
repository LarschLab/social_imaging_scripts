from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class ChannelInfo(BaseModel):
    channel_id: int
    name: str
    wavelength_nm: Optional[float] = None


class BaseStack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stack_id: str
    date: date
    condition: Optional[str] = None
    experimenter: Optional[str] = None
    include_in_analysis: bool = True
    image_quality: Optional[str] = None
    notes: Optional[str] = None
    # IO
    raw_path: Path
    microscope_settings_path: Optional[Path] = None
    # channels
    reference_channel_index: Optional[int] = None  # 1-based index into channels
    channels: list[ChannelInfo] = Field(default_factory=list)


class FunctionalStack(BaseStack):
    stack_type: Literal["functional_stack"]
    # optional but constrained to this type
    round_id: Optional[int] = None
    stimulus_name: Optional[str] = None
    stimulus_metadata_path: Optional[Path] = None
    num_planes: Optional[int] = None
    zoom_factor: Optional[float] = None


class AnatomyStack(BaseStack):
    stack_type: Literal["anatomy_stack"]
    round_id: Optional[int] = None
    plane_spacing: Optional[float] = None


Stack = Annotated[
    Union[FunctionalStack, AnatomyStack],
    Field(discriminator="stack_type"),
]


class AnimalMetadata(BaseModel):
    animal_id: str
    stacks: list[Stack] = Field(default_factory=list)


class MetadataCollection(BaseModel):
    animals: list[AnimalMetadata]

    def by_id(self, animal_id: str) -> AnimalMetadata:
        for item in self.animals:
            if item.animal_id == animal_id:
                return item
        raise KeyError(f"Animal {animal_id} not found")
