from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from ruamel.yaml import YAML

from social_imaging_scripts.metadata.models import (
    AnimalMetadata,
    ChannelInfo,
    FunctionalStack,
    AnatomyStack,
)

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def _maybe_path(value: Optional[str]) -> Optional[Path]:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return None
    return Path(str(value))


def _maybe_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _maybe_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _bool_from_cell(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no"}


def _collect_channels(row) -> list[ChannelInfo]:
    channels: list[ChannelInfo] = []
    for idx in (1, 2, 3):
        name = _clean_str(row.get(f"channel{idx}_name"))
        wl = _maybe_float(row.get(f"channel{idx}_wavelength_nm"))
        if name:
            channels.append(
                ChannelInfo(
                    channel_id=idx,
                    name=name,
                    wavelength_nm=wl,
                )
            )
    return channels


def build_metadata_from_excel(excel_path: Path, output_dir: Path) -> None:
    excel_path = Path(excel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    # stacks is the only required sheet in the simplified design
    stacks_df = xls.parse("stacks").fillna("")

    # group by animal_id
    for animal_id, animal_stacks in stacks_df.groupby(stacks_df["animal_id"].astype(str).str.strip()):
        if not animal_id:
            continue
        stacks: list = []

        for _, srow in animal_stacks.iterrows():
            stack_id = _clean_str(srow.get("stack_id"))
            if not stack_id:
                continue
            stack_type = _clean_str(srow.get("stack_type")) or "anatomy_stack"
            date_value = srow.get("date")
            if date_value in (None, "") or (isinstance(date_value, float) and pd.isna(date_value)):
                raise ValueError(f"Stack {stack_id} for {animal_id} is missing a date")
            date_obj = pd.to_datetime(date_value).date()

            raw_path = _maybe_path(srow.get("raw_path"))
            if raw_path is None:
                raise ValueError(f"Stack {stack_id} for {animal_id} is missing raw_path")

            base_kwargs = dict(
                stack_id=stack_id,
                date=date_obj,
                condition=_clean_str(srow.get("condition")),
                experimenter=_clean_str(srow.get("experimenter")),
                include_in_analysis=_bool_from_cell(srow.get("include_in_analysis")),
                image_quality=_clean_str(srow.get("image_quality")),
                notes=_clean_str(srow.get("notes")),
                raw_path=raw_path,
                microscope_settings_path=_maybe_path(srow.get("microscope_settings_path")),
                reference_channel_index=_maybe_int(srow.get("reference_channel_index")),
            )

            channels = _collect_channels(srow)
            ref_idx = base_kwargs["reference_channel_index"]
            if ref_idx is not None:
                if not (1 <= ref_idx <= len(channels)):
                    raise ValueError(f"reference_channel_index {ref_idx} invalid for stack {stack_id}")

            if stack_type == "functional_stack":
                stack = FunctionalStack(
                    channels=channels,
                    round_id=_maybe_int(srow.get("round_id")),
                    stimulus_name=_clean_str(srow.get("stimulus_name")),
                    stimulus_metadata_path=_maybe_path(srow.get("stimulus_metadata_path")),
                    num_planes=_maybe_int(srow.get("num_planes")),
                    zoom_factor=_maybe_float(srow.get("zoom_factor")),
                    stack_type="functional_stack",
                    **base_kwargs,
                )
            else:
                stack = AnatomyStack(
                    channels=channels,
                    round_id=_maybe_int(srow.get("round_id")),
                    plane_spacing=_maybe_float(srow.get("plane_spacing")),
                    stack_type="anatomy_stack",
                    **base_kwargs,
                )

            stacks.append(stack)

        animal_meta = AnimalMetadata(animal_id=animal_id, stacks=stacks)

        out_path = output_dir / f"{animal_id}.yaml"
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.dump(animal_meta.model_dump(mode="json"), fh)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build animal metadata YAML from spreadsheet")
    parser.add_argument("excel", type=Path, help="Path to animals workbook")
    parser.add_argument("--out", type=Path, default=Path("metadata/animals"), help="Output directory")
    args = parser.parse_args()

    build_metadata_from_excel(args.excel, args.out)


if __name__ == "__main__":
    main()
