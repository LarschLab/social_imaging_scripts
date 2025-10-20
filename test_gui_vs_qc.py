"""Test to compare GUI transformation vs QC plot transformation."""
import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, affine_transform
import math

# Simulate the parameters from the GUI
x_shift_px = 12.0  # GUI x_shift in anatomy pixels
y_shift_px = 4.0   # GUI y_shift in anatomy pixels
rotation_deg = 46.0  # GUI rotation

# Spacing (use matching XY spacing to keep the toy example simple)
anatomy_spacing = (1.0, 1.0, 2.0)  # µm (x, y, z)
confocal_spacing = anatomy_spacing

# Convert GUI pixels to microns using ANATOMY spacing (since GUI operates in anatomy pixel space)
x_shift_um = x_shift_px * anatomy_spacing[0]
y_shift_um = y_shift_px * anatomy_spacing[1]

print("="*80)
print("PARAMETERS")
print("="*80)
print(f"GUI translation: x={x_shift_px} px, y={y_shift_px} px")
print(f"GUI rotation: {rotation_deg}°")
print(f"Anatomy spacing: {anatomy_spacing} µm")
print(f"Confocal spacing: {confocal_spacing} µm")
print(f"Translation in microns: x={x_shift_um} µm, y={y_shift_um} µm")
print()

# ============================================================================
# PART 1: Simulate what the GUI does
# ============================================================================
print("="*80)
print("GUI TRANSFORMATION (what you see in the manual prematch GUI)")
print("="*80)

# Create a simple test image with a marker at a known position
img_size = 100
test_img = np.zeros((img_size, img_size), dtype=np.float32)
# Put a bright spot at (30, 40) to track
test_img[30:35, 40:45] = 1.0

print(f"Original marker position: row=30-35, col=40-45")

# GUI applies: rotate first, then translate
# Step 1: Rotate around center
img_rotated = nd_rotate(test_img, rotation_deg, reshape=False, order=1, mode='constant', cval=0)

# Step 2: Translate by (y_shift, x_shift) - scipy uses (row, col) = (y, x) order
img_gui = nd_shift(img_rotated, shift=(y_shift_px, x_shift_px), order=1, mode='constant', cval=0)

# Find where the marker ended up
coords_gui = np.where(img_gui > 0.5)
if len(coords_gui[0]) > 0:
    gui_center_y = coords_gui[0].mean()
    gui_center_x = coords_gui[1].mean()
    print(f"GUI result: marker at row={gui_center_y:.1f}, col={gui_center_x:.1f}")
    print(f"  -> Shift from original: dy={gui_center_y - 32.5:.1f}, dx={gui_center_x - 42.5:.1f}")
else:
    print("GUI result: marker not found (rotated out of bounds?)")

print()

# ============================================================================
# PART 2: Simulate what the QC plot code does
# ============================================================================
print("="*80)
print("QC PLOT TRANSFORMATION (what confocal_to_anatomy.py does)")
print("="*80)

# Build the affine matrix to match what nd_rotate + nd_shift does
# 
# nd_rotate(img, angle) rotates CCW by `angle` degrees
# nd_shift(img, (dy, dx)) shifts by dy rows down, dx cols right
#
# To replicate this with affine_transform:
# - Rotation by angle θ (CCW) around center
# - Then translation by (dx, dy)
#
# Forward transform: output = R(θ) @ (input - center) + center + translation
# Inverse (what scipy needs): input = R(-θ) @ (output - center - translation) + center
#
# Simplifying: input = R(-θ) @ output + (center - R(-θ) @ (center + translation))
#
# So M_inv = R(-θ) and t_inv = center - R(-θ) @ (center + translation)

theta_forward = math.radians(rotation_deg)  # GUI rotation angle
cos_f = math.cos(theta_forward)
sin_f = math.sin(theta_forward)

# Forward rotation matrix (what GUI does)
R_forward = np.array([
    [cos_f, -sin_f],
    [sin_f, cos_f]
], dtype=np.float64)

# Inverse rotation matrix (for scipy)
R_inv = R_forward.T  # Transpose = inverse for rotation

# Center in 2D
center_2d = np.array([49.5, 49.5], dtype=np.float64)

# When building affine matrix in standard (x,y) Cartesian coordinates:
# x = cols (horizontal, right is +)
# y = rows (vertical, in image coords down is +)
# 
# But scipy's affine_transform operates in array index space (row, col)
# We need to be careful about ordering!
#
# Let's use (row, col) = (y, x) ordering to match scipy
# The GUI shift=(y_shift_px, x_shift_px) should become translation [y, x]
trans_2d = np.array([y_shift_um, x_shift_um], dtype=np.float64)  # (y, x) = (row, col) ordering

# Compute inverse offset for scipy
# t_inv = center - R_inv @ (center + translation)
t_inv_test = center_2d - R_inv @ (center_2d + trans_2d)

print(f"TEST COMPUTATION:")
print(f"R_forward (θ={rotation_deg}°):\n{R_forward}")
print(f"R_inv:\n{R_inv}")
print(f"center_2d: {center_2d}")
print(f"trans_2d (x,y): {trans_2d}")
print(f"t_inv computed: {t_inv_test}")
print()

# Now let's use this directly with scipy
test_img2 = np.zeros((img_size, img_size), dtype=np.float32)
test_img2[30:35, 40:45] = 1.0

img_test = affine_transform(
    test_img2,
    R_inv,
    offset=t_inv_test,
    output_shape=(img_size, img_size),
    order=1,
    mode='constant',
    cval=0
)

coords_test = np.where(img_test > 0.5)
if len(coords_test[0]) > 0:
    test_center_y = coords_test[0].mean()
    test_center_x = coords_test[1].mean()
    print(f"TEST (direct formula): marker at row={test_center_y:.1f}, col={test_center_x:.1f}")
    print(f"  -> Shift from original: dy={test_center_y - 32.5:.1f}, dx={test_center_x - 42.5:.1f}")
    print(f"  -> Difference from GUI: dy={test_center_y - gui_center_y:.1f}, dx={test_center_x - gui_center_x:.1f}")
    print()

# ============================================================================
# PIPELINE AFFINE CHECK
# ============================================================================
print("=" * 80)
print("PIPELINE AFFINE (matrix comparison)")
print("=" * 80)

from social_imaging_scripts.registration.confocal_to_anatomy import _build_prematch_affine
from social_imaging_scripts.registration.prematch import XYMIPPrematchResult

prematch = XYMIPPrematchResult(
    rotation_deg=rotation_deg,
    translation_vox=np.array([x_shift_px, y_shift_px, 0.0]),
    translation_um=np.array([x_shift_um, y_shift_um, 0.0]),
    score=1.0,
    delta_pixels=np.zeros(2),
    matched_centre_pixels=np.zeros(2),
    peak_index=(0, 0),
    downsample_scale=1.0,
    resample_factors=(1.0, 1.0),
    angle_records=[],
    applied=True,
)

fireants_affine = _build_prematch_affine(
    prematch,
    moving_shape=(1, img_size, img_size),
    moving_spacing=confocal_spacing,
    fixed_shape=(1, img_size, img_size),
    fixed_spacing=anatomy_spacing,
)

A_pipeline = fireants_affine[:2, :2]
t_pipeline = fireants_affine[:2, 3]

expected_rotation = np.array(
    [
        [math.cos(math.radians(rotation_deg)), -math.sin(math.radians(rotation_deg))],
        [math.sin(math.radians(rotation_deg)), math.cos(math.radians(rotation_deg))],
    ],
    dtype=np.float64,
)
print("Pipeline rotation:\n", A_pipeline)
print("Expected rotation:\n", expected_rotation)
print("Pipeline translation (µm):", t_pipeline)

def _simulate_gui_mapping(points_phys):
    scale = confocal_spacing[0] / anatomy_spacing[0]
    rescaled_w = int(round(img_size * scale))
    rescaled_h = int(round(img_size * scale))
    canvas_w = max(rescaled_w, img_size)
    canvas_h = max(rescaled_h, img_size)
    pad_x_conf = (canvas_w - rescaled_w) // 2
    pad_y_conf = (canvas_h - rescaled_h) // 2
    pad_x_anat = (canvas_w - img_size) // 2
    pad_y_anat = (canvas_h - img_size) // 2

    outputs = []
    for x_phys, y_phys in points_phys:
        col = x_phys / confocal_spacing[0]
        row = y_phys / confocal_spacing[1]
        col_scaled = col * scale + pad_x_conf
        row_scaled = row * scale + pad_y_conf
        c_x = (canvas_w - 1) / 2
        c_y = (canvas_h - 1) / 2
        theta = math.radians(rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        col_shift = col_scaled - c_x
        row_shift = row_scaled - c_y
        col_rot = cos_t * col_shift - sin_t * row_shift + c_x
        row_rot = sin_t * col_shift + cos_t * row_shift + c_y
        col_trans = col_rot + x_shift_px
        row_trans = row_rot + y_shift_px
        col_final = col_trans - pad_x_anat
        row_final = row_trans - pad_y_anat
        outputs.append([col_final * anatomy_spacing[0], row_final * anatomy_spacing[1]])
    return np.array(outputs)

center = np.array([(img_size - 1) / 2 * confocal_spacing[0], (img_size - 1) / 2 * confocal_spacing[1]])
dx_point = center + np.array([confocal_spacing[0], 0.0])
dy_point = center + np.array([0.0, confocal_spacing[1]])
test_points = np.stack([center, dx_point, dy_point])

pipeline_outputs = (A_pipeline @ test_points.T + t_pipeline[:, None]).T
gui_outputs = _simulate_gui_mapping(test_points)

print("Pipeline mapped points:\n", pipeline_outputs)
print("GUI mapped points:\n", gui_outputs)

rot_error = np.max(np.abs(A_pipeline - expected_rotation))
point_error = np.max(np.abs(pipeline_outputs - gui_outputs))

print()
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Max rotation diff: {rot_error:.3e}")
print(f"Max point diff (µm): {point_error:.3e}")
if rot_error < 1e-6 and point_error < 1e-6:
    print("✓ MATCH! Pipeline affine reproduces GUI transform exactly.")
else:
    print("✗ MISMATCH! Transform still diverges.")
