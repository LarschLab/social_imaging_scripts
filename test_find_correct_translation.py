"""Systematically test different translation formulas with +90° rotation offset."""
import numpy as np
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, affine_transform
import math

# Parameters from GUI
x_shift_px = 12.0
y_shift_px = 4.0
rotation_deg = 46.0
anatomy_spacing = (1.0, 1.0, 2.0)

x_shift_um = x_shift_px * anatomy_spacing[0]
y_shift_um = y_shift_px * anatomy_spacing[1]

print("Testing different translation formulas with +90° rotation offset")
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

# Test different translation formulas
formulas = [
    ("[-x, -y]", lambda x, y: np.array([-x, -y, 0.0])),
    ("[-x, y]", lambda x, y: np.array([-x, y, 0.0])),
    ("[x, -y]", lambda x, y: np.array([x, -y, 0.0])),
    ("[x, y]", lambda x, y: np.array([x, y, 0.0])),
    ("[-y, -x]", lambda x, y: np.array([-y, -x, 0.0])),
    ("[-y, x]", lambda x, y: np.array([-y, x, 0.0])),
    ("[y, -x]", lambda x, y: np.array([y, -x, 0.0])),
    ("[y, x]", lambda x, y: np.array([y, x, 0.0])),
]

def test_formula(name, trans_func):
    """Test a translation formula with +90° rotation."""
    theta = math.radians(rotation_deg + 90.0)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    R = np.array([[cos_t, -sin_t, 0],
                   [sin_t, cos_t, 0],
                   [0, 0, 1]], dtype=np.float64)
    
    center = np.array([49.5, 49.5, 0.0], dtype=np.float64)
    translation = trans_func(x_shift_um, y_shift_um)
    total_translation = center - R @ center + translation
    
    # Build affine for scipy
    M_vox = R[:2, :2]
    t_vox = total_translation[:2]
    M_inv = np.linalg.inv(M_vox)
    t_inv = -M_inv @ t_vox
    
    # Apply transformation
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
        return (center_y, center_x, diff_y, diff_x, error)
    else:
        return (None, None, None, None, 9999)

print("Formula           Result (row, col)    Diff from GUI (dy, dx)    Error")
print("-" * 75)

results = []
for name, func in formulas:
    row, col, dy, dx, error = test_formula(name, func)
    if row is not None:
        print(f"{name:15s}   ({row:4.1f}, {col:4.1f})        ({dy:+5.1f}, {dx:+5.1f})         {error:5.2f}")
        results.append((name, error))
    else:
        print(f"{name:15s}   (not found)                                    {error:5.2f}")

print()
best = min(results, key=lambda x: x[1])
print(f"✓ BEST MATCH: {best[0]} with error {best[1]:.2f} px")
