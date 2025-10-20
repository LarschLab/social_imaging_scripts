# Manual Prematch Coordinate System Issue

**Status**: UNRESOLVED as of 2025-10-20  
**Priority**: HIGH (blocks reliable manual prematch workflow)

## Problem Summary

The manual prematch GUI (`exampleNotebooks/manual_confocal_prematch.ipynb`) and the code that applies the prematch transformation (`src/social_imaging_scripts/registration/confocal_to_anatomy.py`) use incompatible coordinate systems. This manifests as:

1. **Rotation mismatch**: GUI shows angle θ, but code must apply θ+90° for correct orientation
2. **Translation mismatch**: GUI shows confocal centered in anatomy, but QC plots show it shifted (typically top-left)

## Technical Details

### What Works
- **Without +90° offset**: Test scripts show perfect match to GUI when using:
  - Rotation: `theta = GUI_angle` (no offset)
  - Translation: `[y, x]` ordering (array coordinates)
- **Spacing fix**: Now correctly converts GUI pixels → microns using anatomy spacing (1.0 µm/px) instead of confocal spacing (0.328 µm/px)

### What's Broken
- **With +90° offset**: Required for rotation to look correct in actual registration, but breaks translation:
  - Rotation: `theta = GUI_angle + 90.0` (empirically necessary)
  - Translation: No simple formula (`[x,y]`, `[-x,y]`, `[y,x]`, `[y,-x]`, etc.) produces correct result
  - Best attempt `[-x, y]` still has 9.49 px error in test

### Current Code State
```python
# In _build_prematch_affine() (lines ~235-290):
theta = math.radians(float(prematch.rotation_deg) + 90.0)  # +90° offset
translation = np.array([translation_gui[0], -translation_gui[1], 0.0])  # Latest attempt: [x, -y]
```

## Root Cause Theories

1. **Coordinate system flip**: GUI/FIJI might use different axis conventions than FireANTs
2. **Handedness**: Left-handed vs right-handed coordinate system
3. **Rotation direction**: CCW in one system, CW in another
4. **Matrix application**: Issue in how prematch gets left-multiplied: `base @ prematch_matrix`
5. **Hidden transform**: FireANTs might apply additional coordinate transform internally

## Investigation Done

### Test Scripts Created
- `test_gui_vs_qc.py`: Compares GUI transformation vs our affine approach
- `test_find_correct_translation.py`: Systematically tests 8 translation formulas with +90° offset
- `test_rotate_translation.py`: Tests rotating translation by full rotation matrix

### Key Findings
1. GUI does: `rotate(img, θ)` then `shift(img, [dy, dx])`
2. Test proves this can be replicated with standard affine math (rotation matrix + translation)
3. Adding +90° to rotation breaks the mathematical equivalence
4. Rotating translation vector by ±90° doesn't fix it (still 9-28 px error)

## Files Affected

```
src/social_imaging_scripts/registration/confocal_to_anatomy.py:
  - _manual_prematch_to_result() [lines ~97-140]: Converts GUI px → µm (FIXED)
  - _build_prematch_affine() [lines ~235-290]: Builds affine matrix (BROKEN)
  - QC plot generation [lines ~598-680]: Visualizes initialization
  
exampleNotebooks/manual_confocal_prematch.ipynb:
  - GUI using scipy.ndimage.rotate + shift
  - Saves to processing log: manual_prematch[session_id]
  
Test scripts:
  - test_gui_vs_qc.py
  - test_find_correct_translation.py  
  - test_rotate_translation.py
```

## Attempted Solutions (All Failed)

1. ❌ Remove +90° offset → Breaks rotation orientation
2. ❌ Swap x/y: `[y, x]` → Still shifted with +90° offset
3. ❌ Negate: `[-x, -y]`, `[x, -y]`, `[-x, y]` → All produce 9-33 px errors
4. ❌ Rotate by +90°: `[-y, x]` or `[y, -x]` → Still wrong (20-28 px errors)
5. ❌ Rotate by full matrix: `R_inv @ translation` → 28.64 px error
6. ❌ Rotate around different center → Still wrong

## Questions for Future Investigation

1. Why does removing +90° make rotation wrong? What is it compensating for?
2. Is there documentation on FireANTs coordinate conventions vs ImageJ/FIJI?
3. Should the GUI display θ+90° to match what it's actually doing geometrically?
4. Is `_apply_prematch_to_affine` doing `base @ prematch` correct, or should it be `prematch @ base`?
5. Does the QC plot visualization use the same coordinate system as the actual registration?
6. Could this be related to how FireANTs interprets the affine matrix internally?

## Current Workaround

Users can still use manual prematch, with caveats:
- ✓ Better than no prematch initialization
- ✓ Rotation approximately correct (~10° typical error acceptable for FireANTs to refine)
- ✗ Translation persistently wrong (10-30 µm offset)
- ⚠️ FireANTs may recover if offset isn't too large
- ⚠️ May need multiple iterations: run → check QC → adjust GUI → re-run

For critical sessions:
- Prefer automated XY MIP prematch when score >0.6 (avoids coordinate issues)
- Or manually adjust, run, check, iterate

## Next Steps

1. **Investigate FireANTs source**: Look for coordinate system conventions, axis ordering
2. **Check similar issues**: Search FireANTs GitHub for coordinate system discussions
3. **Compare with ANTs**: Check if this is inherited from ITK/ANTs conventions
4. **Test with simple case**: Create minimal test with known rotation/translation
5. **Consider removing feature**: If unfixable, disable manual prematch until coordinate systems aligned
6. **Document thoroughly**: If we find a working formula, document WHY it works

## Impact

- **Blocks**: Reliable manual prematch workflow for challenging sessions
- **Affects**: L395_f11 and potentially other animals with poor automated prematch scores
- **Priority**: High - manual intervention should be more reliable than automated, currently it's worse
