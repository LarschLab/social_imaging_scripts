"""Test rotating the translation vector by the rotation matrix."""
import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, affine_transform
import math

# Parameters
x_shift_px = 12.0
y_shift_px = 4.0
rotation_deg = 46.0
anatomy_spacing = (1.0, 1.0, 2.0)

x_shift_um = x_shift_px * anatomy_spacing[0]
y_shift_um = y_shift_px * anatomy_spacing[1]

print("Testing: rotate translation vector by the +90° offset rotation")
print(f"GUI: x={x_shift_px} px, y={y_shift_px} px, rotation={rotation_deg}°")
print()

# Create test image
img_size = 100
test_img = np.zeros((img_size, img_size), dtype=np.float32)
test_img[30:35, 40:45] = 1.0

# GUI transformation (ground truth)
img_rotated = nd_rotate(test_img, rotation_deg, reshape=False, order=1, mode='constant', cval=0)
img_gui = nd_shift(img_rotated, shift=(y_shift_px, x_shift_px), order=1, mode='constant', cval=0)
coords_gui = np.where(img_gui > 0.5)
gui_center_y = coords_gui[0].mean()
gui_center_x = coords_gui[1].mean()
print(f"GUI result: row={gui_center_y:.1f}, col={gui_center_x:.1f}")
print()

# Approach: The GUI applies translation in the ORIGINAL coordinate frame (before rotation).
# But we apply a +90° rotation to the coordinate system.
# So we need to express the translation in the ROTATED coordinate frame.
#
# If the original translation is T, and we rotate coords by R(+90°),
# then the translation in the new coords is R(-90°) @ T = R_inv @ T

# Build rotation matrix with +90° offset
theta_with_offset = math.radians(rotation_deg + 90.0)
cos_t = math.cos(theta_with_offset)
sin_t = math.sin(theta_with_offset)
R_offset = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

# Translation in GUI frame (using y, x ordering for array coords)
trans_gui_2d = np.array([y_shift_um, x_shift_um], dtype=np.float64)

# Apply INVERSE of the +90° rotation to the translation
# This converts from GUI frame to the +90° rotated frame
R_inv_offset = R_offset.T  # Inverse of rotation is transpose

trans_rotated = R_inv_offset @ trans_gui_2d

print(f"Translation in GUI frame: {trans_gui_2d}")
print(f"R_inv (+90° offset inverse):\n{R_inv_offset}")
print(f"Translation after rotating by R_inv: {trans_rotated}")
print()

# Now build the full transformation
center = np.array([49.5, 49.5], dtype=np.float64)
R_full = R_offset  # Full rotation matrix (GUI angle + 90°)

# Translation: rotate around center, then add the rotated translation
total_translation = center - R_full @ center + trans_rotated

M_inv = np.linalg.inv(R_full)
t_inv = -M_inv @ total_translation

# Apply
test_img2 = np.zeros((img_size, img_size), dtype=np.float32)
test_img2[30:35, 40:45] = 1.0
img_result = affine_transform(test_img2, M_inv, offset=t_inv,
                               output_shape=(img_size, img_size),
                               order=1, mode='constant', cval=0)

coords = np.where(img_result > 0.5)
if len(coords[0]) > 0:
    center_y = coords[0].mean()
    center_x = coords[1].mean()
    diff_y = center_y - gui_center_y
    diff_x = center_x - gui_center_x
    error = np.sqrt(diff_y**2 + diff_x**2)
    
    print(f"Result: row={center_y:.1f}, col={center_x:.1f}")
    print(f"Difference from GUI: dy={diff_y:+.1f}, dx={diff_x:+.1f}")
    print(f"Error: {error:.2f} px")
    
    if error < 1.0:
        print("\n✓ EXCELLENT MATCH!")
    elif error < 5.0:
        print("\n✓ GOOD MATCH!")
    else:
        print("\n✗ Still significant error")
else:
    print("Result: marker not found")
