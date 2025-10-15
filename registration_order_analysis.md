# Confocal-to-Anatomy Registration: Order of Operations Analysis

## Current Order (PROBLEMATIC)

1. **Load images** (line ~366-367)
   - `moving_array` = confocal
   - `fixed_array` = 2p anatomy

2. **CROP first** (line ~371-387) ⚠️
   ```python
   if crop_to_extent:
       crop_slices, crop_info = _compute_extent_crop_slices(...)
       moving_array = moving_array[crop_slices]  # Cropped to anatomy extent
   ```

3. **Prematch calculation** (line ~389-435)
   - Uses CROPPED moving_array
   - Calculates rotation and translation
   - **Problem**: Prematch works on already-cropped data

4. **MASK** (line ~437-444)
   ```python
   moving_mask = _build_central_mask(moving_array.shape, ...)  # Mask on cropped shape
   moving_array *= moving_mask
   ```

5. **Convert to SimpleITK** (line ~446-447)
   
6. **FireANTs registration** (line ~506+)
   - Moments initialization
   - Apply prematch affine (rotation + translation)
   - **Problem**: Rotation applied AFTER crop creates black corners!
   - Affine registration
   - Greedy deformable registration

## The Problem

When you:
1. Crop confocal to anatomy extent
2. Then apply rotation via prematch

**Result**: Black triangular corners appear in the rotated moving image because:
- The crop removed pixels that would be needed after rotation
- Rotation around center brings in empty (black) regions at corners
- These black corners hurt registration quality (treated as real data)

## Example Scenario

```
Original confocal: 800x800 pixels
After crop: 600x600 pixels (to match anatomy extent)
After 54° rotation: Has large black corners where rotated edges extend beyond crop
```

For L395_f06_confocal_round1 with 54° rotation, this is especially problematic!

## Recommended Order

**Option A: Rotate BEFORE Crop**
1. Load images
2. Calculate prematch (on full images)
3. **Apply prematch rotation to moving_array BEFORE any cropping**
4. Crop rotated image to anatomy extent
5. Apply mask
6. Pass to FireANTs (with translation-only prematch, since rotation already applied)

**Option B: Expand Crop to Account for Rotation**
1. Load images
2. Calculate prematch
3. **Expand crop boundaries to accommodate max rotation**
   - Calculate bounding box needed after rotation
   - Add padding: `max_extent = sqrt(2) * original_extent` (for 45° case)
4. Crop with expanded boundaries
5. Apply mask
6. Pass to FireANTs with full prematch (rotation + translation)

**Option C: Skip Crop When Rotation is Large**
1. Load images
2. Calculate prematch
3. If rotation > threshold (e.g., 30°):
   - Skip cropping
   - Apply mask only
4. Else:
   - Apply crop
   - Apply mask
5. Pass to FireANTs

## Recommended Solution: Option A

**Why**: 
- Cleanest approach
- No black corners
- Crop still reduces computation for FireANTs
- Manual prematch rotation already applied, FireANTs just refines

**Implementation**:
1. Apply prematch rotation to raw moving_array using scipy
2. Then crop to anatomy extent
3. Pass translation-only prematch to FireANTs (or let it refine)

## Code Changes Needed

### In `register_confocal_to_anatomy()`:

```python
# Current flow:
moving_array = load_image()
if crop_to_extent:
    moving_array = crop(moving_array)  # ❌ BAD: crop first
prematch = calculate_prematch(moving_array)  # Works on cropped
mask(moving_array)
fireants_register(moving_array, prematch)  # Rotation creates black corners

# Proposed flow:
moving_array = load_image()
prematch = calculate_prematch(moving_array)  # Works on full image
if prematch has rotation:
    moving_array = apply_rotation(moving_array, prematch.rotation)  # ✅ GOOD: rotate first
if crop_to_extent:
    moving_array = crop(moving_array)  # Crop after rotation
mask(moving_array)
fireants_register(moving_array, translation_only_prematch)  # No black corners!
```

### Advantages:
1. ✅ No black corners from rotation
2. ✅ Cropping still reduces computation
3. ✅ Manual prematch rotation already applied before registration
4. ✅ FireANTs just refines translation and small rotation adjustments
5. ✅ Masking works on properly oriented and cropped data

### Considerations:
- Need to add scipy rotation application before crop
- Need to adjust prematch affine to only include translation (rotation already applied)
- May need to adjust crop calculation if center shifts after rotation
- Should log that rotation was pre-applied

## Impact on Current Data

**L395_f06 with 54° and 59° rotations**:
- Currently has significant black corners hurting registration
- Would benefit greatly from this fix

**L331_f01 with 0° rotation**:
- No impact (no rotation to apply)
- Still benefits from correct ordering
