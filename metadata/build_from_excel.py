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
    TwoPhotonPreprocessing,
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


def _parse_blocks(value) -> Optional[list[int]]:
    """Parse block identifiers from a delimited cell."""

    text = _clean_str(value)
    if not text:
        return None
    blocks = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            blocks.append(int(part))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid block identifier '{part}'") from exc
    return blocks or None


def build_metadata_from_excel(excel_path: Path, output_dir: Path) -> None:
    """Convert the multi-sheet workbook into validated per-animal YAML files."""

    excel_path = Path(excel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(excel_path)
    animals_df = xls.parse("animals").fillna("")
    stacks_df = xls.parse("stacks").fillna("")
    stacks_df = stacks_df.copy()
    stacks_df["session_type"] = stacks_df.get("stack_type")
    try:
        tp_df = xls.parse("two_photon_settings").fillna("")
    except ValueError:
        tp_df = pd.DataFrame(columns=[
            "session_id",
            "mode",
            "n_planes",
            "frames_per_plane",
            "flyback_frames",
            "remove_first_frame",
            "blocks",
        ])

    sessions_df = stacks_df[[
        "animal_id",
        "stack_id",
        "stack_type",
        "date",
        "condition",
        "experimenter",
        "include_in_analysis",
        "image_quality",
        "notes",
    ]].rename(columns={
        "stack_id": "session_id",
        "stack_type": "session_type",
    })

    functional_df = stacks_df[stacks_df["session_type"] == "functional_stack"][[
        "stack_id",
        "raw_path",
        "stimulus_name",
        "stimulus_metadata_path",
        "zoom_factor",
    ]].rename(columns={"stack_id": "session_id"})

    anatomy_df = stacks_df[stacks_df["session_type"] == "anatomy_stack"][[
        "stack_id",
        "round_id",
        "raw_path",
        "plane_spacing",
    ]].rename(columns={"stack_id": "session_id"})

    anatomy_df["stack_type"] = anatomy_df["raw_path"].astype(str).apply(
        lambda p: "two_photon" if "\\2p\\" in p or "/2p/" in p else "confocal"
    )

    channel_records = []
    channel_cols = [
        (1, "channel1_name", "channel1_wavelength_nm"),
        (2, "channel2_name", "channel2_wavelength_nm"),
        (3, "channel3_name", "channel3_wavelength_nm"),
    ]
    for _, row in stacks_df.iterrows():
        session_id = row.get("session_id", row.get("stack_id"))
        round_id = row.get("round_id")
        for channel_id, name_col, wavelength_col in channel_cols:
            name = row.get(name_col)
            if not isinstance(name, str) or not name.strip():
                continue
            marker = str(name).strip()
            wavelength = row.get(wavelength_col)
            channel_records.append(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "channel_id": channel_id,
                    "name": marker,
                    "marker": marker,
                    "wavelength_nm": wavelength if pd.notna(wavelength) else None,
                }
            )
    channels_df = pd.DataFrame(channel_records, columns=[
        "session_id",
        "round_id",
        "channel_id",
        "name",
        "marker",
        "wavelength_nm",
    ])

    tp_settings_map: dict[str, TwoPhotonPreprocessing] = {}
    for _, row in tp_df.iterrows():
        session_id = str(row.get("session_id", "")).strip()
        if not session_id:
            continue
        mode = _clean_str(row.get("mode"))
        if not mode:
            continue
        frames_per_plane = _maybe_int(row.get("frames_per_plane"))
        if frames_per_plane is None:
            raise ValueError(f"Two-photon session {session_id} missing frames_per_plane")
        tp_settings_map[session_id] = TwoPhotonPreprocessing(
            mode=mode.lower(),
            n_planes=_maybe_int(row.get("n_planes")),
            frames_per_plane=frames_per_plane,
            flyback_frames=_maybe_int(row.get("flyback_frames")) or 0,
            remove_first_frame=_bool_from_cell(row.get("remove_first_frame")),
            blocks=_parse_blocks(row.get("blocks")),
        )

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
                data_row = functional_df[functional_df["session_id"] == session_id]
                if data_row.empty:
                    raise ValueError(
                        f"Session {session_id} marked functional_stack but missing functional_stack entry"
                    )
                record = data_row.iloc[0]
                session_data = FunctionalSessionData(
                    raw_path=_maybe_path(record.get("raw_path")),
                    stimulus_name=_clean_str(record.get("stimulus_name")),
                    stimulus_metadata_path=_maybe_path(record.get("stimulus_metadata_path")),
                    zoom_factor=_maybe_float(record.get("zoom_factor")),
                )
                if session_data.raw_path is None:
                    raise ValueError(f"Functional session {session_id} missing raw_path")
                if session_id in tp_settings_map:
                    session_data.preprocessing_two_photon = tp_settings_map[session_id]
                if session_id in channels_df["session_id"].values:
                    session_data.channels = [
                        ChannelInfo(
                            channel_id=int(row_c["channel_id"]),
                            name=row_c["name"],
                            marker=row_c["marker"],
                            wavelength_nm=_maybe_float(row_c.get("wavelength_nm")),
                            round_id=_maybe_int(row_c.get("round_id")),
                        )
                        for _, row_c in channels_df[channels_df["session_id"] == session_id].iterrows()
                    ]
                session = FunctionalSession(session_type="functional_stack", session_data=session_data, **base_kwargs)
            elif session_type == "anatomy_stack":
                data_row = anatomy_df[anatomy_df["session_id"] == session_id]
                if data_row.empty:
                    raise ValueError(
                        f"Session {session_id} marked anatomy_stack but missing anatomy_stack entry"
                    )
                record = data_row.iloc[0]
                session_data = AnatomySessionData(
                    raw_path=_maybe_path(record.get("raw_path")),
                    round_id=_maybe_int(record.get("round_id")),
                    plane_spacing=_maybe_float(record.get("plane_spacing")),
                    stack_type=record.get("stack_type", "confocal"),
                )
                if session_data.raw_path is None:
                    raise ValueError(f"Anatomy session {session_id} missing raw_path")
                if session_id in tp_settings_map:
                    session_data.preprocessing_two_photon = tp_settings_map[session_id]
                if session_id in channels_df["session_id"].values:
                    session_data.channels = [
                        ChannelInfo(
                            channel_id=int(row_c["channel_id"]),
                            name=row_c["name"],
                            marker=row_c["marker"],
                            wavelength_nm=_maybe_float(row_c.get("wavelength_nm")),
                            round_id=_maybe_int(row_c.get("round_id")),
                        )
                        for _, row_c in channels_df[channels_df["session_id"] == session_id].iterrows()
                    ]
                session = AnatomySession(session_type="anatomy_stack", session_data=session_data, **base_kwargs)
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
            yaml.dump(animal_meta.model_dump(mode="json", exclude_none=True), fh)


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
