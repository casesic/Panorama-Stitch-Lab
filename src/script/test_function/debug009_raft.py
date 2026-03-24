import cv2
import config_script
import os
import lib_pano.local_setting as ls
from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.stitch.cumulative_stitch import stitch_cumulative


# 固定配置：按需直接改这里
STEP_IDX = 1
USE_MESH_WARP = True
DISTRIBUTE_WARP = True
MASK_PEOPLE = False
DILATE_PIXELS = 20
MASK_THRESHOLD = 0.5
USE_BIDIRECTIONAL = True
BLEND_THRESHOLD = 1
BLEND_ERODE_KERNEL_SIZE = 30
USE_VALID_MASKS = True


def main():
    print("Initializing...")
    source_pano_path = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'preview', 'debug_sampled_cam_1.png')
    target_pano_path = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'preview', 'debug_sampled_cam_0.png')
    source_valid_path = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'preview', 'debug_valid_mask_cam_1.png')
    target_valid_path = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'preview', 'debug_valid_mask_cam_0.png')

    source_pano_full = cv2.imread(source_pano_path)
    target_pano_full = cv2.imread(target_pano_path)
    source_valid_mask = cv2.imread(source_valid_path, cv2.IMREAD_GRAYSCALE)
    target_valid_mask = cv2.imread(target_valid_path, cv2.IMREAD_GRAYSCALE)

    if source_pano_full is None or target_pano_full is None:
        raise FileNotFoundError('Error: Images not found.')
    if USE_VALID_MASKS and (source_valid_mask is None or target_valid_mask is None):
        raise FileNotFoundError('Error: Valid mask images not found.')

    print("Stitching...")
    aligner = RAFTAligner()
    debug_dir = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'test_raft_result')

    final_pano = stitch_cumulative(
        canvas=target_pano_full,
        new_shard=source_pano_full,
        aligner=aligner,
        step_idx=STEP_IDX,
        use_mesh_warp=USE_MESH_WARP,
        distribute_warp=DISTRIBUTE_WARP,
        debug_dir=debug_dir,
        mask_people=MASK_PEOPLE,
        dilate_pixels=DILATE_PIXELS,
        mask_threshold=MASK_THRESHOLD,
        use_bidirectional=USE_BIDIRECTIONAL,
        blend_threshold=BLEND_THRESHOLD,
        blend_erode_kernel_size=BLEND_ERODE_KERNEL_SIZE,
        source_valid_mask=source_valid_mask if USE_VALID_MASKS else None,
        target_valid_mask=target_valid_mask if USE_VALID_MASKS else None,
    )

    final_path = os.path.join(debug_dir, 'final_composed_result.png')
    cv2.imwrite(final_path, final_pano)
    print(f"Final composed panorama saved to: {final_path}")
    print("Done.")


if __name__ == '__main__':
    main()
