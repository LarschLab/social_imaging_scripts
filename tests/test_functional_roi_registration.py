import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)

import numpy as np
import pandas as pd
import pytest

from social_imaging_scripts.functional import roi_registration
from social_imaging_scripts.metadata.config import FunctionalRoiRegistrationQcConfig


def test_build_transform_sequence_forward_and_inverse(tmp_path: Path):
    transforms_dir = tmp_path / "transforms"
    transforms_dir.mkdir()
    affine = transforms_dir / "affine.mat"
    warp = transforms_dir / "warp.nii.gz"
    inverse = transforms_dir / "inverse.nii.gz"
    for path in (affine, warp, inverse):
        path.touch()

    metadata = {
        "outputs": {
            "affine_transform": str(affine),
            "greedy_transform": str(warp),
            "greedy_inverse_transform": str(inverse),
        }
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    seq_forward = roi_registration.build_transform_sequence(
        metadata_path,
        use_inverse_warp=False,
        invert_forward_warp=True,
        invert_inverse_warp=False,
        invert_affine=False,
    )
    assert seq_forward == [(warp, True), (affine, False)]

    seq_inverse = roi_registration.build_transform_sequence(
        metadata_path,
        use_inverse_warp=True,
        invert_forward_warp=False,
        invert_inverse_warp=True,
        invert_affine=True,
    )
    assert seq_inverse == [(inverse, True), (affine, True)]


@pytest.fixture()
def fake_ants(monkeypatch):
    calls = []

    def _apply_transforms_to_points(dim, df, transformlist=None, whichtoinvert=None):
        calls.append(
            {
                "dim": dim,
                "transformlist": list(transformlist or []),
                "whichtoinvert": list(whichtoinvert or []),
            }
        )
        return pd.DataFrame({"x": df["x"], "y": df["y"], "z": df["z"]})

    class DummyAntsImage:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)
            self.spacing = None
            self.origin = None

        def set_spacing(self, spacing):
            self.spacing = spacing

        def set_origin(self, origin):
            self.origin = origin

        def numpy(self):
            return self._array

    def _from_numpy(array):
        return DummyAntsImage(array)

    def _apply_transforms(fixed, moving, transformlist=None, whichtoinvert=None):
        return DummyAntsImage(moving.numpy())

    monkeypatch.setattr(
        roi_registration.ants, "apply_transforms_to_points", _apply_transforms_to_points
    )
    monkeypatch.setattr(roi_registration.ants, "from_numpy", _from_numpy)
    monkeypatch.setattr(roi_registration.ants, "apply_transforms", _apply_transforms)
    return calls


def _write_nrrd(path: Path, data: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    import nrrd  # type: ignore

    header = {
        "space": "left-posterior-superior",
        "space directions": [
            [spacing[2], 0.0, 0.0],
            [0.0, spacing[1], 0.0],
            [0.0, 0.0, spacing[0]],
        ],
        "space origin": list(origin),
    }
    nrrd.write(str(path), data.astype(np.float32), header)


def test_transform_rois_to_reference(tmp_path: Path, fake_ants):
    # Create dummy registration CSV
    df = pd.DataFrame(
        [
            {
                "animal": "A1",
                "moving_plane": 0,
                "scale_moving_to_fixed": 1.0,
                "y_px": 0.0,
                "x_px": 0.0,
                "z_index": 0,
                "success": True,
            }
        ]
    )

    suite2p_root = tmp_path / "suite2p"
    plane_dir = suite2p_root / "plane0"
    plane_dir.mkdir(parents=True)
    stat = np.array(
        [
            {"med": np.array([1.0, 2.0], dtype=np.float32)},
            {"med": np.array([3.0, 4.0], dtype=np.float32)},
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat, allow_pickle=True)

    anatomy_data = np.zeros((2, 6, 6), dtype=np.float32)
    reference_data = np.zeros((2, 6, 6), dtype=np.float32)
    anatomy_path = tmp_path / "anatomy.nrrd"
    reference_path = tmp_path / "reference.nrrd"
    _write_nrrd(anatomy_path, anatomy_data)
    _write_nrrd(reference_path, reference_data)

    transform_path = tmp_path / "warp.nii.gz"
    transform_path.touch()

    output_csv = tmp_path / "out.csv"

    summary = roi_registration.transform_rois_to_reference(
        animal_id="A1",
        session_id="S1",
        registration_df=df,
        suite2p_root=suite2p_root,
        plane_folder_template="plane{plane_index}",
        anatomy_stack_path=anatomy_path,
        fireants_metadata_path=tmp_path / "metadata.json",
        reference_brain_path=reference_path,
        output_csv=output_csv,
        transform_sequence=[(transform_path, False)],
        flip_anatomy_x=False,
    )

    assert summary.total_rois == 2
    assert summary.processed_planes == 1
    assert output_csv.exists()

    out = pd.read_csv(output_csv)
    assert list(out["roi_id"]) == [0, 1]
    assert pytest.approx(out["z_anat"].tolist()) == [0.0, 0.0]
    assert pytest.approx(out["y_anat"].tolist()) == [1.0, 3.0]
    assert pytest.approx(out["x_anat"].tolist()) == [2.0, 4.0]
    assert pytest.approx(out["z_ref"].tolist()) == [0.0, 0.0]
    assert pytest.approx(out["y_ref"].tolist()) == [1.0, 3.0]
    assert pytest.approx(out["x_ref"].tolist()) == [2.0, 4.0]

    assert fake_ants[0]["dim"] == 3
    assert fake_ants[0]["transformlist"] == [str(transform_path)]


def test_transform_rois_to_reference_with_qc(tmp_path: Path, fake_ants):
    df = pd.DataFrame(
        [
            {
                "animal": "A1",
                "moving_plane": 0,
                "scale_moving_to_fixed": 1.0,
                "y_px": 0.0,
                "x_px": 0.0,
                "z_index": 0,
                "success": True,
            }
        ]
    )

    suite2p_root = tmp_path / "suite2p"
    plane_dir = suite2p_root / "plane0"
    plane_dir.mkdir(parents=True)
    stat = np.array(
        [
            {"med": np.array([1.0, 2.0], dtype=np.float32)},
            {"med": np.array([3.0, 4.0], dtype=np.float32)},
        ],
        dtype=object,
    )
    np.save(plane_dir / "stat.npy", stat, allow_pickle=True)

    anatomy_path = tmp_path / "anatomy.nrrd"
    reference_path = tmp_path / "reference.nrrd"
    _write_nrrd(anatomy_path, np.zeros((2, 6, 6), dtype=np.float32))
    _write_nrrd(reference_path, np.zeros((2, 6, 6), dtype=np.float32))

    transform_path = tmp_path / "warp.nii.gz"
    transform_path.touch()

    output_csv = tmp_path / "out.csv"
    qc_output_path = tmp_path / "qc" / "qc.png"

    qc_settings = FunctionalRoiRegistrationQcConfig(
        enabled=True,
        output_subdir=Path("qc"),
        filename_template="{animal_id}_{session_id}_qc.png",
        max_planes=2,
        max_columns=2,
        z_tolerance=1.0,
        point_size=5.0,
        alpha=0.8,
        figsize=(4.0, 4.0),
        dpi=100,
        percentiles=(1.0, 99.0),
    )

    summary = roi_registration.transform_rois_to_reference(
        animal_id="A1",
        session_id="S1",
        registration_df=df,
        suite2p_root=suite2p_root,
        plane_folder_template="plane{plane_index}",
        anatomy_stack_path=anatomy_path,
        fireants_metadata_path=tmp_path / "metadata.json",
        reference_brain_path=reference_path,
        output_csv=output_csv,
        transform_sequence=[(transform_path, False)],
        flip_anatomy_x=False,
        qc_settings=qc_settings,
        qc_output_path=qc_output_path,
    )

    assert summary.qc_path == qc_output_path
    assert qc_output_path.exists()
