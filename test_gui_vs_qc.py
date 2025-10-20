"""Test to compare GUI transformation vs QC plot transformation."""
import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, affine_transform
import math

# Simulate the parameters from the GUI
x_shift_px = 12.0  # GUI x_shift in anatomy pixels
y_shift_px = 4.0   # GUI y_shift in anatomy pixels
rotation_deg = 46.0  # GUI rotation

# Spacing (both confocal and anatomy have same XY spacing after rescaling in GUI)
anatomy_spacing = (1.0, 1.0, 2.0)  # µm (x, y, z)
confocal_spacing = (0.328, 0.328, 3.0)  # µm (x, y, z)

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
# NOW DO IT THE WAY THE CODE ACTUALLY DOES IT
# ============================================================================
print("="*80)
print("QC PLOT TRANSFORMATION (current code approach)")
print("="*80)

theta_qc = math.radians(rotation_deg + 90.0)  # Current code adds +90°
cos_t = math.cos(theta_qc)
sin_t = math.sin(theta_qc)
rotation_matrix = np.array([
    [cos_t, -sin_t, 0],
    [sin_t, cos_t, 0],
    [0, 0, 1]
], dtype=np.float64)

# Center point (using moving/confocal shape for rotation center)
# In reality this would be the confocal stack center
center = np.array([
    0.5 * (img_size - 1) * anatomy_spacing[0],  # x in µm
    0.5 * (img_size - 1) * anatomy_spacing[1],  # y in µm
    0.0
], dtype=np.float64)

# Translation from GUI - current code approach
translation = np.array([y_shift_um, -x_shift_um, 0.0], dtype=np.float64)

# Total translation for rotation around center
total_translation = center - rotation_matrix @ center + translation

print(f"Rotation matrix (θ={rotation_deg + 90.0}°):")
print(rotation_matrix[:2, :2])
print(f"Center: {center}")
print(f"Translation: {translation}")
print(f"Total translation: {total_translation}")
print()

# Build 4x4 affine for scipy
affine_4x4 = np.eye(4, dtype=np.float64)
affine_4x4[:3, :3] = rotation_matrix
affine_4x4[:3, 3] = total_translation

print("Physical space affine (4x4):")
print(affine_4x4)
print()

# Convert to voxel space (for 2D test, spacing is just (x, y))
S_moving = np.diag([anatomy_spacing[0], anatomy_spacing[1], 1.0])
S_fixed = np.diag([anatomy_spacing[0], anatomy_spacing[1], 1.0])
S_moving_inv = np.linalg.inv(S_moving)
S_fixed_inv = np.linalg.inv(S_fixed)

M_phys = affine_4x4[:3, :3]
t_phys = affine_4x4[:3, 3]

M_vox_xyz = S_fixed_inv @ M_phys @ S_moving
t_vox_xyz = S_fixed_inv @ t_phys

print("Voxel space transform (XYZ):")
print(f"M_vox_xyz:\n{M_vox_xyz}")
print(f"t_vox_xyz: {t_vox_xyz}")
print()

# For 2D, we just need the 2x2 rotation and 2D translation
M_vox_2d = M_vox_xyz[:2, :2]
t_vox_2d = t_vox_xyz[:2]

# Scipy affine_transform uses INVERSE
M_inv = np.linalg.inv(M_vox_2d)
t_inv = -M_inv @ t_vox_2d

print("Scipy inverse transform (2D):")
print(f"M_inv:\n{M_inv}")
print(f"t_inv: {t_inv}")
print()

# Apply with scipy affine_transform
# Reset test image
test_img = np.zeros((img_size, img_size), dtype=np.float32)
test_img[30:35, 40:45] = 1.0

img_qc = affine_transform(
    test_img,
    M_inv,
    offset=t_inv,
    output_shape=(img_size, img_size),
    order=1,
    mode='constant',
    cval=0
)

# Find where the marker ended up
coords_qc = np.where(img_qc > 0.5)
if len(coords_qc[0]) > 0:
    qc_center_y = coords_qc[0].mean()
    qc_center_x = coords_qc[1].mean()
    print(f"QC result: marker at row={qc_center_y:.1f}, col={qc_center_x:.1f}")
    print(f"  -> Shift from original: dy={qc_center_y - 32.5:.1f}, dx={qc_center_x - 42.5:.1f}")
else:
    print("QC result: marker not found")

print()

# ============================================================================
# PART 3: Comparison
# ============================================================================
print("="*80)
print("COMPARISON")
print("="*80)

if len(coords_gui[0]) > 0 and len(coords_qc[0]) > 0:
    diff_y = qc_center_y - gui_center_y
    diff_x = qc_center_x - gui_center_x
    print(f"Difference (QC - GUI):")
    print(f"  dy = {diff_y:.1f} px")
    print(f"  dx = {diff_x:.1f} px")
    print()
    
    if abs(diff_y) < 1.0 and abs(diff_x) < 1.0:
        print("✓ MATCH! The transformations are equivalent.")
    else:
        print("✗ MISMATCH! The transformations differ significantly.")
        print()
        print("This means the QC plot is NOT showing what you saw in the GUI.")
        print("The bug is in how we're constructing the transformation matrix.")
else:
    print("Cannot compare - marker not found in one or both results")
