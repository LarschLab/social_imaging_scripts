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
  - Still **adds +90°** to the GUI rotation (legacy hack; see open questions below).
  - Translates so that the confocal centre coincides with the anatomy centre, then applies the rotation about that shared centre (implemented as `T_anat @ R @ T_conf⁻¹`). No GUI X/Y translation is currently applied.
- The QC resample uses the same 4×4 affine (converted via a homogeneous voxel transform), so the overlay now shows the rotated confocal centred inside the anatomy volume—matching exactly what FireANTs receives.

## What Works

- QC overlays finally mirror the manual prematch seed: the confocal block is centred in X/Y/Z and rotated as seen in the GUI.
- FireANTs seeds from the same affine used to draw the QC plot, so debugging the seed is now straightforward.
- Verbose logging reports the GUI angle, applied angle, confocal/anatomy centres, and the mapped centre as a sanity check.

## What Still Fails / Open Questions

1. **Rotation bias** – We still rely on the empirical `+90°` offset. We need to trace the GUI vs FireANTs coordinate conventions so that we can remove the hack and trust the GUI angle verbatim.
2. **Optional GUI translations** – At the moment we ignore the GUI’s XY translation sliders. Once rotation is finalised we can reintroduce the GUI translation path (or formally decide to leave it at zero and document the behaviour).
3. **Unit tests** – Codify the synthetic checks (centres/basis points) so regressions are caught automatically.
4. **Documentation refresh** – Update this note once the rotation bias is gone and the translation story is final; keep the link in `imagingPipelineAgentInstruct.txt` current.

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

---

## 2025‑10‑20 Update: Centering Seed Verified (axes clarified)

What we did to get the confocal to land at the centre of the anatomy in all three views (XY/XZ/YZ):

- Compute both centres in physical (LPS) units from image headers (sizes × spacing):
  - Confocal centre and anatomy centre built from voxel counts and spacings; see `src/social_imaging_scripts/registration/confocal_to_anatomy.py:647`.
- Build a pure translation seed in physical units with identity rotation:
  - We pass `init_translation = centre_moving − centre_fixed` (µm) to FireANTs; we do not permute axes or convert units ourselves. See `src/social_imaging_scripts/registration/confocal_to_anatomy.py:859`.
  - Rotation is disabled for this diagnostic (`init_moment=None`); see `src/social_imaging_scripts/registration/confocal_to_anatomy.py:860`.
- Freeze optimisation to verify the seed exactly (no solver drift):
  - We skip optimisation and evaluate directly; log confirms “Rigid optimisation skipped; prematch seed is frozen.” (`src/social_imaging_scripts/registration/confocal_to_anatomy.py:872`).

Key lessons about axes/units (FireANTs vs. GUI):

- FireANTs initialisers (rigid/affine) take translations in physical units, not voxel indices and not pre‑normalised torch coordinates. Its internal `torch2phy/phy2torch` handles conversion; external pre‑conversion caused off‑by‑plane rotations and translations out of FOV.
- A translation that centres volumes must be `centre_moving − centre_fixed` because the rigid mapping is y = R x + t with x in fixed space and y in moving space.
- For this translation‑only seed, no axis permutation is required. Earlier attempts to “swap to [z,y,x]” or transpose the 3×3 produced shears and flips in the YZ view.

Next steps (now that centring is correct):

- Reintroduce rotation seeding from the GUI while keeping translation as above. With `around_center=True`, FireANTs internally adjusts the learnable translation by t′ = t − c + A c; we therefore keep t = `centre_moving − centre_fixed` and set `init_moment` to the GUI rotation about Z. Validate on L395_f11 and a second animal.
- Restore mask soft edges and greedy only after the rigid seed + affine behave as expected.
- Update unit tests to assert that the synthetic centre‑only seed maps the moving centre exactly onto the fixed centre within 1 voxel in all three axes.

## 2025‑10‑21 Update: ND grid mismatch still unresolved

- Rebuilt the synthetic harness to compare the world transform against FireANTs plus PyTorch’s `affine_grid`/`grid_sample`.
- Added full physical→torch conversion, `around_center=False`, and logged FireANTs’ torch-space affine.
- Switched the harness to `padding_mode='border'` + `mode='nearest'` and reduced the translation (Δ=20 µm) to keep the bright voxel inside the volume.
- Composed ND transforms for both forward (`I2N_m @ A_torch @ N2I_f`) and inverse (`I2N_m @ A_torch⁻¹ @ N2I_f`) cases.
- **Observation:** the bright voxel still collapses to indices `(0,0,0)`—the sampler sees mostly zero because we’re still using the wrong ND matrix.

### Next diagnostic steps

1. Double‑check the index↔ND conversion for the chosen `align_corners` setting (PyTorch default is `False`; FireANTs currently uses `True`).
2. Probe forward vs inverse explicitly and keep the one that lands the bright voxel at the analytic target.
3. Once the harness reports `delta vox = (0,0,0)` (Δ=0) and the expected offset for Δ≠0, reapply the same logic inside `_build_prematch_affine` before re-enabling the optimiser.
