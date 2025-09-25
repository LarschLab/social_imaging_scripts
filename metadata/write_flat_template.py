from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_template(path: Path) -> None:
    stacks_cols = [
        "animal_id",
        "stack_id",
        "stack_type",  # functional_stack | anatomy_stack
        "date",
        "condition",
        "experimenter",
        "include_in_analysis",
        "image_quality",
        "notes",
        "raw_path",
        "microscope_settings_path",
        # functional-only
        "stimulus_name",
        "stimulus_metadata_path",
        "num_planes",
        "zoom_factor",
        # anatomy-only
        "round_id",
        "plane_spacing",
        # channels (flattened)
        "reference_channel_index",
        "channel1_name",
        "channel1_wavelength_nm",
        "channel2_name",
        "channel2_wavelength_nm",
        "channel3_name",
        "channel3_wavelength_nm",
    ]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(columns=stacks_cols).to_excel(writer, sheet_name="stacks", index=False)


def main():
    write_template(Path("metadata/animals_template.xlsx"))


if __name__ == "__main__":
    main()
