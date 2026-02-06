
import cv2
import config_script
from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.stitch.cumulative_stitch import stitch_cumulative
import lib_pano.local_setting as ls
import os

def load_projected_image(idx):
    """Load the pre-projected (global warp) images."""
    path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', f'debug_sampled_cam_{idx}.png')
    img = cv2.imread(path)
    if img is None:
        print(f"[Warning] Could not load {path}")
        return None
    return img

if __name__ == "__main__":
    print("=== Cumulative Stitching 'Snowball' Test ===")
    USE_MESH_WARP = True # Global Toggle for Mesh Warping
    # 改为 True 启用双向光流分布（形变分摊），改为 False 使用单向形变（只变新图，画板不动）
    DISTRIBUTE_WARP = True 
    
    # Initialize Aligner
    aligner = RAFTAligner()
    
    # 1. Start with Cam 0
    print("Initializing Canvas with Cam 0...")
    canvas = load_projected_image(0)
    if canvas is None:
        print("Failed to start: Cam 0 missing.")
    else:
        # 2. Iterate through subsequent cameras
        # Adding Cam 1, then Cam 2...
        cam_indices = [1, 2] 
        
        debug_dir = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_steps')

        for i, cam_idx in enumerate(cam_indices):
            print(f"\n--- Adding Cam {cam_idx} ---")
            new_img = load_projected_image(cam_idx)
            
            # UPDATE the canvas by stitching the new image onto it
            canvas = stitch_cumulative(
                canvas, 
                new_img, 
                aligner, 
                step_idx=i+1, 
                use_mesh_warp=USE_MESH_WARP, 
                distribute_warp=DISTRIBUTE_WARP,
                debug_dir=debug_dir
            )
            
            # Save intermediate checkpoint
            out_path = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', f'snowball_result_after_cam{cam_idx}.png')
            cv2.imwrite(out_path, canvas)
            print(f"Saved checkpoint: {out_path}")

    print("\nProcessing Complete.")

