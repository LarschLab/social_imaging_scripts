# Manual Confocal-to-Anatomy Prematch Workflow

## Overview

This document describes the manual prematching system for confocal→2p anatomy registration. It allows users to manually specify initial translation (x, y) and rotation values through an interactive GUI before FireANTs registration runs.

## Storage Strategy (Hybrid Approach)

The system uses the existing hybrid logging approach:

1. **Processing Log (Primary)** - `metadata/processed/{animal_id}.yaml`
   - Stores manual prematch parameters under `confocal_to_anatomy_registration` stage
   - Structure:
     ```yaml
     stages:
       confocal_to_anatomy_registration:
         parameters:
           manual_prematch:
             "{session_id}":
               translation_x_px: -45.2
               translation_y_px: 23.8
               rotation_deg: 12.5
               created_at: "2025-10-15T14:30:00Z"
               method: "manual_gui"
     ```

2. **Stage-specific JSON** - `{output_dir}/{animal}_confocal_to_anatomy_metadata.json`
   - Keeps existing detailed registration metadata
   - Redundant but useful for quick inspection next to outputs

## Implementation Files

### 1. GUI Notebook: `exampleNotebooks/manual_confocal_prematch.ipynb`

**Features:**
- Interactive ipywidgets-based GUI with matplotlib overlay
- Fixed image (2p anatomy) shown in magenta
- Moving image (confocal) shown in green
- Three sliders: X translation, Y translation, Rotation
- Alpha blending slider for visibility adjustment
- Buttons: Save, Skip, Reset
- Automatically loads existing prematch values if present
- Can be run standalone or called programmatically

**Key Functions:**
- `collect_sessions_needing_prematch()` - Scans all animals and finds sessions without prematch
- `ManualPrematchGUI` - Widget-based interactive interface
- `run_manual_prematch_gui()` - Main entry point, processes all sessions sequentially
- `save_prematch_to_log()` - Writes to processing log YAML
- `get_existing_prematch()` - Reads from processing log YAML

**Standalone Usage:**
```python
# In the notebook, simply run all cells
# It will automatically find and process all sessions needing prematch
```

**Programmatic Usage:**
```python
from pathlib import Path
from social_imaging_scripts.metadata.config import load_project_config
from social_imaging_scripts.metadata.loader import load_animals

# Import the function (would need to export from notebook or convert to .py)
# from manual_prematch import run_manual_prematch_gui

cfg = load_project_config()
animals = list(load_animals(...).animals)

stats = run_manual_prematch_gui(
    animals=animals,
    cfg=cfg,
    force=False  # Set True to redo existing prematches
)
```

## Pipeline Integration

### ✅ Phase 1: Manual Standalone Use (COMPLETE)
- Run `manual_confocal_prematch.ipynb` before running the batch pipeline
- Saves prematch values to processing logs in `{output_base_dir}/metadata/processed/`
- GUI automatically loads existing prematch values for re-adjustment

### ✅ Phase 2: Pipeline Reading (COMPLETE)
The pipeline now automatically reads and uses manual prematch values:

1. **Modified `confocal_to_anatomy.py`**:
   - Added `_load_manual_prematch_from_log()` - Reads prematch from processing log
   - Added `_manual_prematch_to_result()` - Converts pixel values to XYMIPPrematchResult
   - Modified `register_confocal_to_anatomy()` to accept `output_base_dir` and `processing_log_config`
   - **Priority**: Manual prematch is checked FIRST, automated prematch only runs if manual is missing
   
2. **Modified `orchestrator.py`**:
   - Updated `process_confocal_to_anatomy_registration()` to pass processing log parameters
   - Passes `cfg.output_base_dir` and `cfg.processing_log` to registration function

**How It Works:**
```python
# In confocal_to_anatomy.py
manual_prematch_data = _load_manual_prematch_from_log(
    animal_id, confocal_session_id, output_base_dir, processing_log_config
)

if manual_prematch_data:
    logger.info("Using manual prematch from processing log (overrides automated prematch)")
    prematch_result = _manual_prematch_to_result(manual_prematch_data, spacing)
else:
    # Fall back to automated prematch if enabled
    if prematch_settings.enabled:
        prematch_result = run_xy_mip_prematch(...)
```

**Transform Conversion:**
- Manual GUI saves translations in **anatomy pixel space** (as displayed to user)
- Registration code converts to **moving (confocal) voxel space** for FireANTs
- Rotation angle applied around image center
- Prematch result passed to FireANTs as initial seed via affine matrix

### Phase 3: Orchestrator Pre-flight (FUTURE)
Add optional pre-flight check in orchestrator batch processing:
   ```

2. **Add config flag** in `pipeline_defaults.yaml`:
   ```yaml
   confocal_to_anatomy_registration:
     # ... existing config ...
     use_manual_prematch: true  # Whether to check for manual prematch values
     manual_prematch_priority: true  # If true, manual prematch overrides automated
   ```

### Phase 3: Integrated GUI Launch (Future)
Modify `orchestrator.py` to optionally launch GUI:

1. **Add manual_prematch section** to `pipeline_defaults.yaml`:
   ```yaml
   manual_prematch:
     mode: reuse  # skip/reuse/force
     enabled: true  # Whether to check for manual prematch
     interactive: true  # If true, launch GUI for missing prematches
   ```

2. **Pre-flight check in orchestrator**:
   ```python
   def run_pipeline(animal_ids, cfg):
       # ... existing setup ...
       
       if cfg.manual_prematch.enabled:
           # Collect sessions needing prematch
           sessions_needing_prematch = []
           for animal_id in animal_ids:
               # ... check logic ...
               
           if sessions_needing_prematch and cfg.manual_prematch.interactive:
               print(f"⚠ {len(sessions_needing_prematch)} sessions need manual prematch")
               user_choice = input("Launch GUI now? (y/n): ")
               if user_choice.lower() == 'y':
                   run_manual_prematch_gui(animals, cfg, force=False)
       
       # ... continue with normal pipeline ...
   ```

## Workflow Examples

### Scenario 1: First-time setup
```python
# 1. Run manual prematch notebook
# Opens GUI for each session, user adjusts sliders, clicks Save

# 2. Run batch pipeline
# Pipeline reads manual prematch values and uses them as seeds
```

### Scenario 2: Redo specific animal
```python
# In manual_prematch.ipynb:
animals = [a for a in animals if a.animal_id == "L395_f10"]
stats = run_manual_prematch_gui(animals=animals, cfg=cfg, force=True)

# Then rerun pipeline with FORCE mode for that animal
```

### Scenario 3: Pipeline-integrated (future)
```python
# In batch_two_photon_pipeline.ipynb:
cfg_slice.manual_prematch.mode = StageMode.REUSE  # Check for manual prematch
cfg_slice.manual_prematch.interactive = True  # Launch GUI if missing

result = run_pipeline(animal_ids=selected_animals, cfg=cfg_slice)
# Pipeline automatically launches GUI if prematches missing, then continues
```

## Technical Details

### Transform Application
- **Order**: Rotation first (around center), then translation
- **Interpolation**: Bilinear (order=1) for smooth results
- **Padding**: Images padded to same size before overlay
- **Coordinate system**: Pixel-based (not yet converted to physical units)

### Data Flow
```
User adjusts sliders
  ↓
apply_transform(moving_mip, x, y, rot)
  ↓
create_overlay(fixed_mip, transformed_moving)
  ↓
matplotlib display updates
  ↓
User clicks "Save"
  ↓
save_prematch_to_log()
  ↓
metadata/processed/{animal_id}.yaml updated
  ↓
(Future) Pipeline reads values during registration
```

### Unit Conversion (Future Enhancement)
Currently stores pixel values. For FireANTs integration, need to:
1. Load voxel spacing from confocal/anatomy preprocessing metadata
2. Convert pixel translation to physical units (μm)
3. Build affine matrix with rotation + translation
4. Pass matrix to FireANTs as initial transform

```python
# Pseudocode for conversion:
def prematch_to_affine_matrix(x_px, y_px, rot_deg, voxel_spacing_um):
    # Convert pixels to physical units
    x_um = x_px * voxel_spacing_um[2]  # X is dimension 2
    y_um = y_px * voxel_spacing_um[1]  # Y is dimension 1
    
    # Build 2D rotation matrix
    theta = np.deg2rad(rot_deg)
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Build 3D affine (rotation in XY plane)
    affine = np.eye(4)
    affine[1:3, 1:3] = rot_matrix  # Y-X rotation
    affine[1, 3] = y_um  # Y translation
    affine[2, 3] = x_um  # X translation
    
    return affine
```

## Known Limitations & TODOs

### Current Limitations:
1. ❌ Pipeline doesn't read manual prematch values yet
2. ❌ No physical unit conversion (pixels only)
3. ❌ Sequential GUI (one session at a time, blocks execution)
4. ❌ No undo/history
5. ❌ No side-by-side before/after comparison

### Future Enhancements:
1. ✅ **Pipeline integration** (read values during registration) - HIGH PRIORITY
2. ✅ **Physical unit conversion** - HIGH PRIORITY
3. ⚠ **Batch GUI** (show multiple sessions in tabs)
4. ⚠ **Quality metrics** (correlation score as user adjusts)
5. ⚠ **Export/import** prematch values as CSV
6. ⚠ **Keyboard shortcuts** (arrow keys for fine adjustment)
7. ⚠ **Checkerboard overlay** mode (alternative to alpha blend)

## Testing Plan

### Unit Tests:
- [ ] `test_save_prematch_to_log()` - Verify YAML writing
- [ ] `test_load_existing_prematch()` - Verify YAML reading
- [ ] `test_apply_transform()` - Verify rotation + translation math
- [ ] `test_collect_sessions_needing_prematch()` - Verify session discovery

### Integration Tests:
- [ ] Run GUI for known animal with confocal session
- [ ] Verify processing log updated correctly
- [ ] Run pipeline and verify prematch values are used (once integrated)

### Manual QA:
- [ ] Test with L331_f01 (known good alignment)
- [ ] Test with L395_f10 (challenging case)
- [ ] Verify overlay colors are distinguishable
- [ ] Test slider responsiveness
- [ ] Test Save/Skip/Reset buttons

## Questions for User

1. **Color scheme**: Is magenta (fixed) + green (moving) acceptable, or prefer red/cyan?
2. **Slider ranges**: Current ranges based on image size. Should we have absolute max limits?
3. **Pipeline integration timeline**: Should we integrate into orchestrator immediately, or test standalone first?
4. **Unit conversion**: Should we convert to physical units now, or wait until pipeline integration?
5. **Interactive mode**: For batch pipeline, should GUI launch automatically for missing prematches, or require explicit flag?

## Next Steps

1. ✅ **DONE**: Create GUI notebook with basic functionality
2. **TODO**: Test GUI with one animal (L331_f01 or L395_f06)
3. **TODO**: Verify processing log writes correctly
4. **TODO**: Add unit conversion functions
5. **TODO**: Integrate with `confocal_to_anatomy.py` to read manual prematch
6. **TODO**: Add config flags to `pipeline_defaults.yaml`
7. **TODO**: Update orchestrator to check for manual prematch
8. **TODO**: Test full pipeline with manual prematch seed
