#!/usr/bin/env python
"""Compare a hand-constructed rigid seed (rotation+translation) applied via
FireANTs vs. a simple NumPy/SciPy expectation and a direct point check.

Goal: Verify that seeding FireANTs with R (about Z) and t = c_m - R c_f + Δ
produces exactly the same effect as sampling the moving image at
    y = R x + t
for each fixed-space coordinate x, for the special case where we only check
the fixed-space centre point and place a bright voxel at the analytically
predicted moving-space coordinate y.

If this is correct, the warped output should have its maximum at the fixed
centre voxel. Any deviation indicates a mismatch in axis order or the seed
parameterisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
import torch


@dataclass
class VoxelGeom:
    shape: tuple[int, int, int]  # (Z, Y, X)
    spacing: tuple[float, float, float]  # (X, Y, Z) microns

    def centre_phys(self) -> np.ndarray:
        z, y, x = self.shape
        sx, sy, sz = self.spacing
        return np.array([
            0.5 * (x - 1) * sx,
            0.5 * (y - 1) * sy,
            0.5 * (z - 1) * sz,
        ], dtype=np.float64)

    def phys_to_idx(self, p: np.ndarray) -> tuple[int, int, int]:
        sx, sy, sz = self.spacing
        x = int(round(p[0] / sx))
        y = int(round(p[1] / sy))
        z = int(round(p[2] / sz))
        # clamp inside volume
        z = max(0, min(self.shape[0] - 1, z))
        y = max(0, min(self.shape[1] - 1, y))
        x = max(0, min(self.shape[2] - 1, x))
        return (z, y, x)


def make_image_from_array(arr: np.ndarray, spacing: tuple[float, float, float]):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def main():
    try:
        from fireants.io.image import Image as FAImage, BatchedImages
        from fireants.registration.rigid import RigidRegistration
    except Exception as e:  # pragma: no cover
        print("fireants not available:", e)
        return
    from social_imaging_scripts.registration.confocal_to_anatomy import _convert_phys_to_torch_affine

    # Geometry from logs
    moving_geom = VoxelGeom(shape=(61, 922, 922), spacing=(0.32796363114129157, 0.32796363114129157, 3.0))
    fixed_geom = VoxelGeom(shape=(216, 512, 512), spacing=(0.8736364543437957, 0.8736364543437957, 2.0))

    # Build moving with a single bright voxel at the analytically expected location
    moving_arr = np.zeros(moving_geom.shape, dtype=np.float32)
    fixed_arr = np.zeros(fixed_geom.shape, dtype=np.float32)

    # Start with pure translation to determine grid convention (10 µm to the right = -X in LPS).
    R = np.eye(3, dtype=np.float64)
    delta = np.array([-10.0, 0.0, 0.0], dtype=np.float64)

    c_m = moving_geom.centre_phys()
    c_f = fixed_geom.centre_phys()
    t = c_m - (R @ c_f) + delta
    y_expected = c_m + delta
    idx_expected = moving_geom.phys_to_idx(y_expected)
    moving_arr[idx_expected] = 1.0

    moving_img = make_image_from_array(moving_arr, moving_geom.spacing)
    fixed_img = make_image_from_array(fixed_arr, fixed_geom.spacing)

    moving_batch = BatchedImages(FAImage(moving_img, device='cpu', dtype=torch.float32))
    fixed_batch = BatchedImages(FAImage(fixed_img, device='cpu', dtype=torch.float32))

    affine_phys = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)
    affine_phys[:, :, :3] = torch.from_numpy(R.astype(np.float32))
    affine_phys[:, :, 3] = torch.from_numpy(t.astype(np.float32))

    R_torch, t_torch = _convert_phys_to_torch_affine(affine_phys, fixed_batch, moving_batch)
    print("fixed torch2phy:\n", fixed_batch.get_torch2phy()[0].cpu().numpy())
    print("moving phy2torch:\n", moving_batch.get_phy2torch()[0].cpu().numpy())

    # Compose full index-space 4×4
    T_torch = torch.eye(4, dtype=torch.float32)
    T_torch[:3, :3] = R_torch.squeeze(0)
    T_torch[:3, 3] = t_torch.squeeze(0)

    def perm_zyx_to_xyz():
        P = torch.zeros(4, 4, dtype=torch.float32)
        P[0, 2] = 1.0  # x <= idx x
        P[1, 1] = 1.0  # y <= idx y
        P[2, 0] = 1.0  # z <= idx z
        P[3, 3] = 1.0
        return P

    def perm_xyz_to_zyx():
        return perm_zyx_to_xyz().t()

    def make_i2n(shape, align_corners=True):
        z, y, x = shape
        M = torch.eye(4, dtype=torch.float32)
        if align_corners:
            M[0, 0] = 2.0 / (x - 1); M[0, 3] = -1.0
            M[1, 1] = 2.0 / (y - 1); M[1, 3] = -1.0
            M[2, 2] = 2.0 / (z - 1); M[2, 3] = -1.0
        else:
            M[0, 0] = 2.0 / x; M[0, 3] = -1.0 + 1.0 / x
            M[1, 1] = 2.0 / y; M[1, 3] = -1.0 + 1.0 / y
            M[2, 2] = 2.0 / z; M[2, 3] = -1.0 + 1.0 / z
        return M

    def make_n2i(shape, align_corners=True):
        I2N = make_i2n(shape, align_corners)
        return torch.inverse(I2N)

    ALIGN = True
    I2N_mov = make_i2n(moving_geom.shape, ALIGN)
    N2I_mov = make_n2i(moving_geom.shape, ALIGN)
    I2N_fix = make_i2n(fixed_geom.shape, ALIGN)
    N2I_fix = make_n2i(fixed_geom.shape, ALIGN)

    moving_tensor = torch.from_numpy(moving_arr).unsqueeze(0).unsqueeze(0)
    size = (1, 1, *fixed_geom.shape)

    combos = []
    for ALIGN in (True, False):
        I2N_mov = make_i2n(moving_geom.shape, ALIGN)
        N2I_fix = make_n2i(fixed_geom.shape, ALIGN)
        Pzx = perm_zyx_to_xyz()
        Pxz = perm_xyz_to_zyx()
        theta_fwd = (I2N_mov @ Pzx @ T_torch @ Pxz @ N2I_fix)[:3, :].unsqueeze(0)
        theta_inv = (I2N_mov @ Pzx @ torch.inverse(T_torch) @ Pxz @ N2I_fix)[:3, :].unsqueeze(0)
        combos.append((ALIGN, "forward", theta_fwd))
        combos.append((ALIGN, "inverse", theta_inv))

    results = {}

    for ALIGN, tag, theta in combos:
        grid = torch.nn.functional.affine_grid(theta, size, align_corners=ALIGN)
        print(
            f"ALIGN={ALIGN}, {tag} grid ranges:",
            grid[..., 0].min().item(), grid[..., 0].max().item(),
            grid[..., 1].min().item(), grid[..., 1].max().item(),
            grid[..., 2].min().item(), grid[..., 2].max().item(),
        )
        warped = torch.nn.functional.grid_sample(
            moving_tensor,
            grid,
            mode='nearest',
            padding_mode='border',
            align_corners=ALIGN,
        )
        argmax = torch.nonzero(warped == warped.max(), as_tuple=False)
        print(f"ALIGN={ALIGN}, {tag} theta:\n{theta.squeeze(0)}")
        print(f"ALIGN={ALIGN}, {tag} argmax voxels:", argmax[:5])
        results[(ALIGN, tag)] = warped.squeeze().numpy()

    fixed_centre_idx = np.array([
        fixed_geom.shape[0] // 2,
        fixed_geom.shape[1] // 2,
        fixed_geom.shape[2] // 2,
    ])
    print("Expected fixed centre idx:", fixed_centre_idx)
    print("Bright voxel (moving idx):", idx_expected)

    for (ALIGN, tag), vol in results.items():
        argmax = np.unravel_index(np.argmax(vol), vol.shape)
        delta_idx = np.array(argmax) - fixed_centre_idx
        print(f"ALIGN={ALIGN}, {tag} argmax -> {argmax}, delta={delta_idx}")

if __name__ == "__main__":
    main()
