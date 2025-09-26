from pathlib import Path
from social_imaging_scripts.preprocessing.two_photon.motion import move_suite2p_outputs
motion_output = Path(r"D:\pipelineTestOutput\tmp_merge_test\motion")
seg_output = Path(r"D:\pipelineTestOutput\tmp_merge_test\suite2p")
results = move_suite2p_outputs(
    animal_id="L395_f10",
    plane_idx=0,
    suite2p_folder=Path(r"D:\pipelineTestOutput\tmp_suite2p_debug"),
    motion_output=motion_output,
    segmentation_output=seg_output,
)
print(results)
