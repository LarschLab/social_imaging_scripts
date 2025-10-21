# Rigid Seed Debug Notes (2025-10-20)

## Goal
Seed FireANTs’ rigid stage with a world-space transform that applies:

1. A 45° clockwise rotation about the shared centre of confocal and anatomy.
2. A pure +100 µm translation along the anatomy X axis (viewer’s right).

The warped confocal should remain centred in Y/Z and move only in +X after rotation.

## Attempts & Findings

- **Baseline translation-only seed**
  - Translation `t = centre_conf - centre_anat` in physical units works as expected (confocal recentres in all axes).

- **Add 45° rotation + translation (world space)**
  - Analytical seed: `t_base = centre_conf - R · centre_anat`
  - Added Δ = [100, 0, 0] µm (LPS +X).
  - QC shows maths consistent: mapped centre matches expectation.
  - **Observed issue:** warped stack drifts diagonally (top-left).
  - Likely cause: translation interpreted in rotated frame (FireANTs’ internal `around_center` adjustment).

- **Synthetic unit test (`scripts/test_fireants_rigid_seed_vs_numpy.py`)**
  - Single bright voxel at analytically predicted location.
  - Warped argmax lands at origin ⇒ translation still applied in wrong basis.
  - Logged `fireants affine (torch space)` reveals R/t scaled by spacing; translation ≈ 0 instead of expected shift.

- **Investigated FireANTs internals**
  - `RigidRegistration` with `around_center=True` performs `t' = t - c + R c`.
  - `get_warp_parameters` returns `moving_phy2torch @ ( rigid @ fixed_torch2phy )`.
  - Need to supply translation in FireANTs’ torch basis and disable `around_center`.

- **Helper `_convert_phys_to_torch_affine`**
  - Currently permutes using `fixed_batch.get_torch2phy()` and `moving_batch.get_phy2torch()`.
  - Resulting `R_torch` still scaled (anisotropic), translation ≈ 0.66.
  - Applying to fixed-centre torch coord yields physical `(251, 151, 90)` as expected, but warped output still at origin ⇒ FireANTs ignores physical values when array is entirely zero.

- **Next steps**
  - Rebuild 4×4 in `[z, y, x]` order to match FireANTs tensor layout.
  - Ensure conversion includes homogeneous row/column permutations.
  - Re-run synthetic unit test until `delta vox=(0,0,0)`.
  - Only then integrate rotation + GUI translation back into `_build_prematch_affine`.

