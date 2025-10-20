"""Test the QC plot transformation to debug why confocal is black."""

import numpy as np
import torch
from scipy.ndimage import affine_transform

# Simulate the data sizes from L395_f11
moving_shape = (61, 922, 922)  # Confocal (Z, Y, X)
fixed_shape = (216, 922, 922)  # Anatomy (Z, Y, X)

spacing = (0.33, 0.33, 3.0)  # Confocal spacing (X, Y, Z) in µm
fixed_spacing_um = (0.33, 0.33, 2.0)  # Anatomy spacing (X, Y, Z) in µm

# Create dummy data
moving_array = np.random.rand(*moving_shape).astype(np.float32) * 100
fixed_array = np.random.rand(*fixed_shape).astype(np.float32) * 100

# Simulate the init_affine from ACTUAL prematch (from the logs: 46° rotation)
# From logs:
# [[  0.6946584  -0.7193398   0.        145.8998   ]
#  [  0.7193398   0.6946584   0.        -68.75632  ]
#  [  0.          0.          1.        125.       ]]
init_affine = torch.tensor([
    [0.6946584, -0.7193398, 0.0, 145.8998],
    [0.7193398,  0.6946584, 0.0, -68.75632],
    [0.0,        0.0,       1.0, 125.0]
], dtype=torch.float32).unsqueeze(0)

print("Initial affine matrix (46° rotation + translation + Z-offset):")
print(init_affine[0].numpy())
print()

# Now apply the transformation code from confocal_to_anatomy.py
init_affine_np = init_affine[0].cpu().numpy()

# Build spacing scaling matrices (diagonal)
S_moving = np.diag([spacing[0], spacing[1], spacing[2]])
S_fixed = np.diag([fixed_spacing_um[0], fixed_spacing_um[1], fixed_spacing_um[2]])
S_fixed_inv = np.diag([1.0/fixed_spacing_um[0], 1.0/fixed_spacing_um[1], 1.0/fixed_spacing_um[2]])

# Extract physical space transform
M_phys = init_affine_np[:, :3]
t_phys = init_affine_np[:, 3]

print("Physical transform:")
print("M_phys:", M_phys)
print("t_phys:", t_phys)
print()

# Convert to voxel space
M_vox = S_fixed_inv @ M_phys @ S_moving
t_vox = S_fixed_inv @ t_phys

print("Voxel transform (XYZ ordering):")
print("M_vox:", M_vox)
print("t_vox:", t_vox)
print()

# Convert to ZYX ordering
M_vox_zyx = M_vox[[2, 1, 0], :][:, [2, 1, 0]]
t_vox_zyx = t_vox[[2, 1, 0]]

print("Voxel transform (ZYX ordering):")
print("M_vox_zyx:", M_vox_zyx)
print("t_vox_zyx:", t_vox_zyx)
print()

# Scipy uses inverse
M_inv = np.linalg.inv(M_vox_zyx)
t_inv = -M_inv @ t_vox_zyx

print("Inverse transform for scipy:")
print("M_inv:", M_inv)
print("t_inv:", t_inv)
print()

# Apply transform
moving_init_np = affine_transform(
    moving_array,
    M_inv,
    offset=t_inv,
    output_shape=fixed_array.shape,
    order=1,
    mode='constant',
    cval=0.0
)

print("Results:")
print(f"Moving original: min={moving_array.min():.2f}, max={moving_array.max():.2f}, mean={moving_array.mean():.2f}")
print(f"Moving transformed: min={moving_init_np.min():.2f}, max={moving_init_np.max():.2f}, mean={moving_init_np.mean():.2f}")
print(f"Non-zero voxels: {np.count_nonzero(moving_init_np)}/{moving_init_np.size}")
print()

# Check where the data ended up
nonzero_coords = np.nonzero(moving_init_np)
if len(nonzero_coords[0]) > 0:
    print("Non-zero data location:")
    print(f"  Z range: {nonzero_coords[0].min()}-{nonzero_coords[0].max()} (out of {fixed_shape[0]})")
    print(f"  Y range: {nonzero_coords[1].min()}-{nonzero_coords[1].max()} (out of {fixed_shape[1]})")
    print(f"  X range: {nonzero_coords[2].min()}-{nonzero_coords[2].max()} (out of {fixed_shape[2]})")
else:
    print("ERROR: No non-zero voxels in transformed data!")
    print()
    print("Expected Z center of moving data (in voxels):")
    moving_z_center_vox = (moving_shape[0] - 1) / 2.0
    print(f"  Moving Z center: {moving_z_center_vox:.1f} voxels")
    print()
    z_offset_um = init_affine[0, 2, 3].item()
    print("With Z offset of {:.1f} µm and spacing {:.2f} µm/voxel:".format(z_offset_um, fixed_spacing_um[2]))
    print(f"  Offset in voxels: {z_offset_um / fixed_spacing_um[2]:.1f}")
    expected_z = moving_z_center_vox * (spacing[2] / fixed_spacing_um[2]) + (z_offset_um / fixed_spacing_um[2])
    print(f"  Expected Z location in fixed: {expected_z:.1f} voxels")

# Now test the actual code from confocal_to_anatomy.py
print("\n" + "="*60)
print("Testing new coordinate transform code:")
print("="*60 + "\n")

# New approach with proper permutation
init_affine_np = init_affine[0].cpu().numpy()
M_phys = init_affine_np[:, :3]
t_phys = init_affine_np[:, 3]

S_moving = np.diag([spacing[0], spacing[1], spacing[2]])
S_fixed_inv = np.diag([1.0/fixed_spacing_um[0], 1.0/fixed_spacing_um[1], 1.0/fixed_spacing_um[2]])

M_vox_xyz = S_fixed_inv @ M_phys @ S_moving
t_vox_xyz = S_fixed_inv @ t_phys

print("Voxel transform (XYZ):")
print("M_vox_xyz:\n", M_vox_xyz)
print("t_vox_xyz:", t_vox_xyz)
print()

# Check that rotation is preserved
angle_xyz = np.arctan2(M_vox_xyz[1, 0], M_vox_xyz[0, 0]) * 180 / np.pi
print(f"Rotation angle in XY plane (XYZ coords): {angle_xyz:.1f}°")
print()

# Permutation matrix
P_zyx_to_xyz = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]], dtype=np.float32)

M_vox_zyx = P_zyx_to_xyz.T @ M_vox_xyz @ P_zyx_to_xyz
t_vox_zyx = P_zyx_to_xyz.T @ t_vox_xyz

print("Voxel transform (ZYX):")
print("M_vox_zyx:\n", M_vox_zyx)
print("t_vox_zyx:", t_vox_zyx)
print()

# Check rotation in ZYX - should be in the YX plane (indices [1,2])
angle_zyx = np.arctan2(M_vox_zyx[2, 1], M_vox_zyx[1, 1]) * 180 / np.pi
print(f"Rotation angle in YX plane (ZYX coords): {angle_zyx:.1f}°")
print("(Should match the 46° rotation from prematch)")
print()

M_inv = np.linalg.inv(M_vox_zyx)
t_inv = -M_inv @ t_vox_zyx

# Apply NEW transform
moving_init_new = affine_transform(
    moving_array,
    M_inv,
    offset=t_inv,
    output_shape=fixed_array.shape,
    order=1,
    mode='constant',
    cval=0.0
)

print("Results with NEW transform:")
print(f"Moving transformed: min={moving_init_new.min():.2f}, max={moving_init_new.max():.2f}, mean={moving_init_new.mean():.2f}")
print(f"Non-zero voxels: {np.count_nonzero(moving_init_new)}/{moving_init_new.size}")

nonzero_coords = np.nonzero(moving_init_new)
if len(nonzero_coords[0]) > 0:
    print("Non-zero data location:")
    print(f"  Z range: {nonzero_coords[0].min()}-{nonzero_coords[0].max()} (out of {fixed_shape[0]})")
    print(f"  Y range: {nonzero_coords[1].min()}-{nonzero_coords[1].max()} (out of {fixed_shape[1]})")
    print(f"  X range: {nonzero_coords[2].min()}-{nonzero_coords[2].max()} (out of {fixed_shape[2]})")

