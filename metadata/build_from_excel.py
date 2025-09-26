"""Build per-animal YAML metadata files from the spreadsheet entry form."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from ruamel.yaml import YAML

from social_imaging_scripts.metadata.models import (
    AnatomySession,
    AnatomySessionData,
    AnimalMetadata,
    AvailableModalities,
    BehaviorSession,
    ChannelInfo,
    FunctionalSession,
    FunctionalSessionData,
    MetaSource,
    OtherSession,
)

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def _maybe_path(value: Optional[str]) -> Optional[Path]:
    """Return a ``Path`` if the string is non-empty, otherwise ``None``."""

    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return None
    return Path(str(value))


def _maybe_int(value: Optional[str]) -> Optional[int]:
    """Parse an optional integer cell from the spreadsheet."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _maybe_float(value: Optional[str]) -> Optional[float]:
    """Parse an optional float cell from the spreadsheet."""

    if value in (None, ""):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _clean_str(value: Optional[str]) -> Optional[str]:
    """Strip whitespace and normalise empty spreadsheet cells to ``None``."""

    if value in (None, ""):
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _bool_from_cell(value) -> bool:
    """Interpret spreadsheet flags that may be blank, 0/1, yes/no, or booleans."""

    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no"}


def build_metadata_from_excel(excel_path: Path, output_dir: Path) -> None:
    """Convert the multi-sheet workbook into validated per-animal YAML files."""

    excel_path = Path(excel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    animals_df = xls.parse("animals").fillna("")
    sessions_df = xls.parse("sessions").fillna("")
    functional_df = xls.parse("functional_stack").fillna("")
    anatomy_df = xls.parse("anatomy_stack").fillna("")
    channels_df = xls.parse("channels").fillna("")

    functional_map: dict[str, FunctionalSessionData] = {}
    for _, row in functional_df.iterrows():
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            continue
        raw_path = _maybe_path(row.get("raw_path"))
        if raw_path is None:
            raise ValueError(f"functional_stack entry for {session_id} is missing raw_path")
        functional_map[session_id] = FunctionalSessionData(
            raw_path=raw_path,
            stimulus_name=_clean_str(row.get("stimulus_name")),
            stimulus_metadata_path=_maybe_path(row.get("stimulus_metadata_path")),
            microscope_settings_path=_maybe_path(row.get("microscope_settings_path")),
            num_planes=_maybe_int(row.get("num_planes")),
            zoom_factor=_maybe_float(row.get("zoom_factor")),
        )

    anatomy_map: dict[str, AnatomySessionData] = {}
    for _, row in anatomy_df.iterrows():
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            continue
        raw_path = _maybe_path(row.get("raw_path"))
        if raw_path is None:
            raise ValueError(f"anatomy_stack entry for {session_id} is missing raw_path")
        if session_id in anatomy_map:
            raise ValueError(f"Duplicate anatomy_stack entry for session {session_id}")
        anatomy_map[session_id] = AnatomySessionData(
            raw_path=raw_path,
            round_id=_maybe_int(row.get("round_id")),
            plane_spacing=_maybe_float(row.get("plane_spacing")),
            microscope_settings_path=_maybe_path(row.get("microscope_settings_path")),
        )

    channels_map: dict[str, list[ChannelInfo]] = {}
    for _, row in channels_df.iterrows():
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            continue
        channel_id = _maybe_int(row.get("channel_id"))
        if channel_id is None:
            raise ValueError(f"Channel entry for session {session_id} is missing channel_id")
        info = ChannelInfo(
            channel_id=channel_id,
            name=_clean_str(row.get("name")) or f"channel_{channel_id}",
            marker=_clean_str(row.get("marker")),
            wavelength_nm=_maybe_float(row.get("wavelength_nm")),
            round_id=_maybe_int(row.get("round_id")),
        )
        channels_map.setdefault(session_id, []).append(info)

    for _, row in animals_df.iterrows():
        animal_id = str(row.get("animal_id", "")).strip()
        if not animal_id:
            continue

        meta_sources = []
        meta_path = _clean_str(row.get("meta_source_path"))
        if meta_path:
            meta_sources.append(
                MetaSource(
                    type=_clean_str(row.get("meta_source_type")) or "other",
                    path=meta_path,
                )
            )

        available = AvailableModalities(
            functional_2p=_bool_from_cell(row.get("functional_2p")),
            anatomy_2p=_bool_from_cell(row.get("anatomy_2p")),
            confocal_rounds=int(row.get("confocal_rounds") or 0),
            behavior=_bool_from_cell(row.get("behavior")),
        )

        sessions: list = []
        animal_sessions = sessions_df[
            sessions_df["animal_id"].astype(str).str.strip() == animal_id
        ]
        for _, sess in animal_sessions.iterrows():
            session_id = str(sess.get("session_id", "")).strip()
            if not session_id:
                continue
            session_type = _clean_str(sess.get("session_type")) or "other"

            date_value = sess.get("date")
            if date_value in (None, "") or (isinstance(date_value, float) and pd.isna(date_value)):
                raise ValueError(f"Session {session_id} for {animal_id} is missing a date")
            date_obj = pd.to_datetime(date_value).date()

            base_kwargs = dict(
                session_id=session_id,
                date=date_obj,
                condition=_clean_str(sess.get("condition")),
                experimenter=_clean_str(sess.get("experimenter")),
                include_in_analysis=_bool_from_cell(sess.get("include_in_analysis")),
                image_quality=_clean_str(sess.get("image_quality")),
                notes=_clean_str(sess.get("notes")),
            )

            if session_type == "functional_stack":
                data = functional_map.get(session_id)
                if data is None:
                    raise ValueError(
                        f"Session {session_id} marked functional_stack but missing functional_stack entry"
                    )
                if session_id in channels_map:
                    data.channels = channels_map[session_id]
                session = FunctionalSession(session_type="functional_stack", session_data=data, **base_kwargs)
            elif session_type == "anatomy_stack":
                data = anatomy_map.get(session_id)
                if data is None:
                    raise ValueError(
                        f"Session {session_id} marked anatomy_stack but missing anatomy_stack entry"
                    )
                if session_id in channels_map:
                    data.channels = channels_map[session_id]
                session = AnatomySession(session_type="anatomy_stack", session_data=data, **base_kwargs)
            elif session_type == "behavior":
                session = BehaviorSession(session_type="behavior", session_data={}, **base_kwargs)
            else:
                session = OtherSession(session_type="other", session_data={}, **base_kwargs)

            sessions.append(session)

        animal_meta = AnimalMetadata(
            animal_id=animal_id,
            genotype=_clean_str(row.get("genotype")),
            owner=_clean_str(row.get("owner")),
            tank=_clean_str(row.get("tank")),
            date_of_birth=pd.to_datetime(row.get("date_of_birth")).date()
            if _clean_str(row.get("date_of_birth"))
            else None,
            meta_sources=meta_sources,
            root_dir=_maybe_path(row.get("root_dir")),
            available_modalities=available,
            sessions=sessions,
        )

        out_path = output_dir / f"{animal_id}.yaml"
        with out_path.open("w", encoding="utf-8") as fh:
            yaml.dump(animal_meta.model_dump(mode="json"), fh)


def main():
    """Entry point so the script can be invoked from the command line."""

    import argparse

    parser = argparse.ArgumentParser(description="Build animal metadata YAML from spreadsheet")
    parser.add_argument("excel", type=Path, help="Path to animals workbook")
    parser.add_argument("--out", type=Path, default=Path("metadata/animals"), help="Output directory")
    args = parser.parse_args()

    build_metadata_from_excel(args.excel, args.out)


if __name__ == "__main__":
    main()