from pathlib import Path
import shutil
from social_imaging_scripts.preprocessing.two_photon import motion
from social_imaging_scripts.metadata import loader

collection = loader.load_animals()
animal = collection.by_id("L395_f10")
session = [s for s in animal.sessions if s.session_type == "functional_stack"][0]
plane_idx = 0
plane_tiff = Path(r"D:\pipelineTestOutput\L395_f10\02_reg\00_preprocessing\2p_functional\01_individualPlanes\L395_f10_plane0.tif")
ops_template = motion.load_global_ops(Path(r"D:\pipelineTest\ressources\suite2p_ops_may2025.npy"))
output_root = Path(r"D:\pipelineTestOutput\tmp_suite2p_debug")
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)
motion.run_suite2p_one_plane(
    plane_tiff=plane_tiff,
    ops_template=ops_template,
    save_path=output_root,
    fps=5.0,
)
print("done")
