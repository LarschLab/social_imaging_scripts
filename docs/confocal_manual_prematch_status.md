# Confocal→Anatomy Manual Prematch: Current Understanding (2025‑10‑20)

## Background
We rely on the manual prematch GUI (`manual_confocal_prematch.ipynb`) whenever the automated XY MIP heuristic fails. The GUI lets users dial in a rotation and translation on rescaled MIPs (confocal resampled to anatomy pixel spacing) and stores the slider values in the processing log. The goal is to seed FireANTs with the exact same rigid transform that the user saw in the GUI and to visualise that seed in the initialization QC plot.

Historically, `_build_prematch_affine` in `confocal_to_anatomy.py` applied two empirical adjustments:

1. Add +90° to the GUI rotation so the confocal subjectively “looked correct”.
2. Flip the Y translation (`[x, -y]`) before injecting it into the affine.

This produced acceptable results for a handful of datasets but introduced systematic rotation and translation errors (QC overlays shifted/rotated, FireANTs starting far from the GUI pose).

## What We Tested

Over the past debugging session we ran a series of experiments:

1. **Literal GUI reconstruction** – rebuild the exact GUI transform (rescale → pad → rotate about the canvas centre → translate → unpad) and convert the resulting affine into FireANTs physical axes.
2. **SimpleITK vs SciPy** – verify how resampling back into the anatomy grid should convert the FireANTs affine into voxel space (SITK needs the inverse transform; SciPy needs the inverse matrix + offset).
3. **Centre mapping checks** – log where the confocal centre lands after the prematch affine to confirm whether mismatches come from rotation or translation.
4. **QC parity** – ensure the initialization QC preview uses the exact same affine as FireANTs so any misalignment in QC reflects the actual seed.

## Current Implementation (2025‑10‑20)

- `_build_prematch_affine` now:
  - Still **adds +90°** to the GUI rotation (we have not removed this hack yet).
  - Computes translation by following the GUI pipeline: rescale, pad, rotate about the canvas centre, apply the stored translation, remove anatomy padding, then convert the resulting pixel transform to physical space. This replaced the previous `[x, -y]` shortcut.
- The QC image path in `register_confocal_to_anatomy` uses the same affine that is sent to FireANTs (no axis swaps, only a clean matrix inverse), so QC overlays match the FireANTs seed exactly.

## What Works

- Rotation in the current QC screenshots matches the GUI (the “diamond” overlay has the same orientation as the manual prematch).
- Translation now comes directly from the GUI transform instead of the empirical `[x, -y]` sign flip, so the centre mapping log is consistent with the GUI to first order.
- QC plots and FireANTs now see the *same* affine, making QC a reliable diagnostic again.

## What Still Fails

- The **+90° rotation bias** remains: we are still implicitly assuming a 90° mismatch between the GUI and FireANTs coordinate frames. The latest QC overlay shows rotation visually correct; however, this is due to the GUI value + our +90° offset. We need to confirm the actual coordinate conventions and remove this hack entirely.
- Translation is still off by a few microns in real datasets (see L395_f11). The derived translation is closer but still differs slightly from what the user expected.
- SITK/SciPy conversions can still be confusing. Small mistakes (swapping axes or using the wrong inverse) reintroduce 90°/180° flips. We need a cleaner utility to map FireANTs affines to array transforms.

## Lessons Learned

1. **Always reconstruct transforms from first principles** (rescaling, padding, rotation, translation). Shortcuts lead to silent errors.
2. **QC must mirror FireANTs exactly**; otherwise, we’re debugging two coordinate systems at once.
3. **SimpleITK takes physical-space transforms** and internally applies the inverse; SciPy expects an explicit inverse matrix + offset in array index space.
4. Our assumptions about the GUI vs FireANTs axes are still incomplete; rotations appear to be correct visually with the +90° offset, but the exact source of that 90° gap is unresolved.

## Open Questions & Next Steps

1. **Remove the +90° hack**: Derive the true rotation mapping by tracing the orientation of the confocal preprocessing (flip horizontal + Z) and the GUI’s display axes. Validate on at least one real dataset (e.g., L395_f11).
2. **Quantify translation accuracy**: Once rotation is correct, confirm translation numerically across multiple sessions. Post logs showing `centre -> centre` to check for residual bias.
3. **Unit-test the transform**: Add small unit tests (or a standalone script) that feed known slider values into `_build_prematch_affine`, then assert that the centre/basis points end up exactly where the GUI pipeline predicts.
4. **Document coordinate conventions**: Clean up this doc once the fix lands and keep the link updated in `imagingPipelineAgentInstruct.txt`.

For quick reference, see `test_gui_vs_qc.py` in the repo, which now exercises `_build_prematch_affine` end-to-end.
