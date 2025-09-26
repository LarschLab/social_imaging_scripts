# Metadata Entry Workflow

This document explains how to record new animals and imaging sessions so the
pipeline can discover raw data automatically. The metadata lives in CSV/YAML
form inside this repository, but the day-to-day data entry happens in the Excel
workbook `metadata/animals_entries.xlsx` (a blank scaffold is available as
`metadata/animals_template.xlsx`).

## 1. Duplicate the entry workbook for your batch

1. Copy `metadata/animals_template.xlsx` to a new workbook (for example
   `metadata/animals_entries_2025Q4.xlsx`).
2. Keep the sheet names exactly as provided: `animals`, `sessions`,
   `functional_stack`, `anatomy_stack`, and `channels`.

Each sheet captures a specific layer of the metadata schema:

- **animals**: static per-animal information (genotype, owner, root directory,
  etc.).
- **sessions**: one row per imaging or behavioural session. The
  `session_type` column determines which additional sheet must also contain a
  matching row.
- **functional_stack**: details that only apply to functional two-photon
  sessions (stimulus metadata, microscope configuration, number of planes).
- **anatomy_stack**: details for any anatomy stack (two-photon or confocal),
  including round IDs and acquisition settings.
- **channels**: optional rows describing the channels present in a given
  session (`round_id` is useful when multiple confocal rounds exist).

## 2. Add or update animals

Fill one row per animal in the **animals** tab:

- `animal_id`: canonical identifier (`L395_f11`, etc.).
- `root_dir`: location of the raw data root (e.g. `D:/pipelineTest/L395_f11`).
- `meta_source_type` / `meta_source_path`: reference back to the lab record that
  justifies the metadata entry (spreadsheet, lab notebook, etc.).
- `available_modalities`: the boolean columns (`functional_2p`, `anatomy_2p`,
  `behavior`) plus `confocal_rounds` describe what data you expect to exist for
  the animal. These flags help downstream scripts decide which pipeline stages
  to run.

The remaining columns (`genotype`, `owner`, `tank`, `date_of_birth`) capture
context that is often needed in figures or QC reports. Leave a cell blank if
unknown.

## 3. Record sessions as experiments finish

As soon as an acquisition completes, add a row in the **sessions** tab:

- Choose a unique `session_id` (`<animal>_<modality>_<date>` works well).
- Set `session_type` to `functional_stack`, `anatomy_stack`, `behavior`, or
  `other`.
- Fill in the date, condition (live/fixed), experimenter, and any QC notes.
- Toggle `include_in_analysis` to `0` if the session should be skipped by
  default.

For sessions with imaging payloads, also add supporting rows:

- **functional sessions** need one row in the `functional_stack` sheet with the
  session ID, the raw data folder, stimulus metadata locations, and microscope
  settings. Optional fields (planes, zoom) can be filled later.
- **anatomy sessions** need one row in the `anatomy_stack` sheet with the raw
  stack path, an optional round ID, and optional acquisition settings.
- If the imaging session has multiple channels, enumerate them in the
  `channels` sheet. Re-use the same `session_id` and `round_id`, then specify
  `channel_id`, `name`, `marker`, and the wavelength in nanometres. Use channel
  names consistent with acquisition software (e.g. `gcamp`, `sst1_1`).

It is fine if a session initially lists only the raw file path: spacing and
other settings can be filled in once they are known.

## 4. Regenerate YAML files

When the spreadsheet has been updated, run the converter from the repository
root (activate your environment first):

```powershell
python metadata/build_from_excel.py metadata/animals_entries.xlsx --out metadata/animals
```

This validates every row and writes one YAML file per animal under
`metadata/animals/`. Existing files with the same name are overwritten, so the
YAML directory always reflects the latest state of the workbook.

## 5. Loading metadata in code

Notebooks and scripts should stay within the repository root and call:

```python
from social_imaging_scripts.metadata.loader import load_animals

collection = load_animals()
fish = collection.by_id("L395_f11")
for session in fish.sessions:
    print(session.session_type, session.session_data.raw_path)
```

`load_animals()` discovers every YAML file in `metadata/animals/`, validates it
with the schema, and returns a collection that downstream modules can iterate.

## 6. Tips and good practices

- Use relative paths where possible if the raw data lives beneath a shared
  mount. Absolute drive letters are acceptable for local testing but make sure
  they match what the processing machine expects.
- Keep the workbook under version control so changes across experiments are
  reviewed. If multiple users edit simultaneously, prefer saving their own copy
  and merging periodically with the converter script.
- Add brief QC notes whenever you adjust `include_in_analysis`. This context
  becomes invaluable when investigating pipeline failures or comparing animals.
- When new modalities appear, they can be recorded with `session_type = other`
  until the schema is extended.

Following these steps keeps metadata consistent without forcing the team to edit
YAML by hand, while giving the pipeline deterministic input about available
stacks and channels.