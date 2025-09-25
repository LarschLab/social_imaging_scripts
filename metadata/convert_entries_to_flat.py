from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def build_flat_workbook(old_path: Path, new_path: Path) -> None:
    xls = pd.ExcelFile(old_path)
    animals = xls.parse("animals").fillna("")
    sessions = xls.parse("sessions").fillna("")
    func = xls.parse("functional_stack").fillna("")
    anat = xls.parse("anatomy_stack").fillna("")
    chans = xls.parse("channels").fillna("")

    # Map session -> functional/anatomy rows
    func_map = {str(r["session_id"]).strip(): r for _, r in func.iterrows() if str(r.get("session_id", "")).strip()}
    anat_map = {str(r["session_id"]).strip(): r for _, r in anat.iterrows() if str(r.get("session_id", "")).strip()}

    # Group channels per session
    chan_group: Dict[str, List[dict]] = {}
    for _, r in chans.iterrows():
        sid = str(r.get("session_id", "")).strip()
        if not sid:
            continue
        chan_group.setdefault(sid, []).append(r)

    cols = [
        "animal_id",
        "stack_id",
        "stack_type",
        "date",
        "condition",
        "experimenter",
        "include_in_analysis",
        "image_quality",
        "notes",
        "raw_path",
        "microscope_settings_path",
        "stimulus_name",
        "stimulus_metadata_path",
        "num_planes",
        "zoom_factor",
        "round_id",
        "plane_spacing",
        "reference_channel_index",
        "channel1_name",
        "channel1_wavelength_nm",
        "channel2_name",
        "channel2_wavelength_nm",
        "channel3_name",
        "channel3_wavelength_nm",
    ]
    rows: List[dict] = []

    for _, s in sessions.iterrows():
        animal_id = str(s.get("animal_id", "")).strip()
        sid = str(s.get("session_id", "")).strip()
        if not sid:
            continue
        stype = str(s.get("session_type", "")).strip()
        base = {
            "animal_id": animal_id,
            "stack_id": sid,
            "stack_type": stype,
            "date": s.get("date"),
            "condition": s.get("condition"),
            "experimenter": s.get("experimenter"),
            "include_in_analysis": s.get("include_in_analysis"),
            "image_quality": s.get("image_quality"),
            "notes": s.get("notes"),
            "raw_path": "",
            "microscope_settings_path": "",
            "stimulus_name": "",
            "stimulus_metadata_path": "",
            "num_planes": "",
            "zoom_factor": "",
            "round_id": "",
            "plane_spacing": "",
            "reference_channel_index": "",
            "channel1_name": "",
            "channel1_wavelength_nm": "",
            "channel2_name": "",
            "channel2_wavelength_nm": "",
            "channel3_name": "",
            "channel3_wavelength_nm": "",
        }

        if stype == "functional_stack":
            fr = func_map.get(sid)
            if fr is not None:
                base["raw_path"] = fr.get("raw_path", "")
                base["microscope_settings_path"] = fr.get("microscope_settings_path", "")
                base["stimulus_name"] = fr.get("stimulus_name", "")
                base["stimulus_metadata_path"] = fr.get("stimulus_metadata_path", "")
                base["num_planes"] = fr.get("num_planes", "")
                base["zoom_factor"] = fr.get("zoom_factor", "")
        elif stype == "anatomy_stack":
            ar = anat_map.get(sid)
            if ar is not None:
                base["raw_path"] = ar.get("raw_path", "")
                base["microscope_settings_path"] = ar.get("microscope_settings_path", "")
                base["round_id"] = ar.get("round_id", "")
                base["plane_spacing"] = ar.get("plane_spacing", "")

        # Channels -> flatten up to 3, set reference to gcamp if present
        chs = sorted(chan_group.get(sid, []), key=lambda r: int(r.get("channel_id", 9999)))
        ref_index = ""
        for i, rr in enumerate(chs[:3], start=1):
            base[f"channel{i}_name"] = rr.get("name", "")
            base[f"channel{i}_wavelength_nm"] = rr.get("wavelength_nm", "")
            if str(rr.get("name", "")).strip().lower() == "gcamp" and ref_index == "":
                ref_index = i
        base["reference_channel_index"] = ref_index

        rows.append(base)

    stacks_df = pd.DataFrame(rows, columns=cols)

    # Write new workbook
    with pd.ExcelWriter(new_path, engine="openpyxl") as writer:
        animals.to_excel(writer, sheet_name="animals", index=False)
        stacks_df.to_excel(writer, sheet_name="stacks", index=False)


def main():
    old = Path("metadata/animals_entries.xlsx")
    new = Path("metadata/animals_entries.xlsx")
    build_flat_workbook(old, new)


if __name__ == "__main__":
    main()

