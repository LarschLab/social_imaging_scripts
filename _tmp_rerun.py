from pathlib import Path
from social_imaging_scripts.preprocessing.two_photon import motion
from social_imaging_scripts.metadata import loader

collection = loader.load_animals()
animal = collection.by_id("L395_f10")
session = [s for s in animal.sessions if s.session_type == "functional_stack"][0]
ops_template = motion.load_global_ops(Path(r"D:\pipelineTest\ressources\suite2p_ops_may2025.npy"))
output_root = Path(r"D:\pipelineTestOutput\L395_f10\02_reg\00_preprocessing\2p_functional")
plane_dir = Path(r"D:\pipelineTestOutput\L395_f10\02_reg\00_preprocessing\2p_functional\01_individualPlanes")
for plane_idx in range(5):
    plane_tiff = plane_dir / f"L395_f10_plane{plane_idx}.tif"
    result = motion.run_motion_correction(
        animal=animal,
        plane_idx=plane_idx,
        plane_tiff=plane_tiff,
        ops_template=ops_template,
        fps=5.0,
        output_root=output_root,
        fast_disk=None,
        reprocess=True,
    )
    print(plane_idx, result['motion_tiff'])
