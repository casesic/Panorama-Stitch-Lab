import cv2
import config_script
import os
import lib_pano.local_setting as ls
from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.stitch.cumulative_stitch import stitch_cumulative

# 1. Setup
print("Initializing...")
# Source = cam 2, Target = cam 0
source_pano_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_2.png') 
target_pano_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_0.png')

source_pano_full = cv2.imread(source_pano_path)
target_pano_full = cv2.imread(target_pano_path)

if source_pano_full is None or target_pano_full is None:
    print("Error: Images not found.")
    exit()

# 2. Stitch using the library function
print("Stitching...")
aligner = RAFTAligner()
debug_dir = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result')

# Note: stitch_cumulative adds 'new_shard' (source) onto 'canvas' (target).
final_pano = stitch_cumulative(
    canvas=target_pano_full,
    new_shard=source_pano_full,
    aligner=aligner,
    step_idx=99, # Arbitrary debug index
    use_mesh_warp=True,
    # You can toggle this to True to test bidirectional warp
    distribute_warp=False, 
    debug_dir=debug_dir
)

# 3. Save
final_path = os.path.join(debug_dir, 'final_composed_result.png')
cv2.imwrite(final_path, final_pano)
print(f"Final composed panorama saved to: {final_path}")
print("Done.")
