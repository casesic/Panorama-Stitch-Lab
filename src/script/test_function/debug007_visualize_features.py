import sys
import os
import config_script

from lib_pano.vis.visualize_features import visualize_feature_points_on_panorama, visualize_feature_matching_errors
import lib_pano.local_setting as ls

def main():
    # 输入路径
    panorama_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'stitched.png')
    npz_ground_paths = [os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam2_cam4_ground_matches_filtered.npz'),
                    os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam2_cam6_ground_matches_filtered.npz'),
                    os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam4_cam6_ground_matches_filtered.npz')]
    calib_opt_path = ls.CALIB_OPT_FILE
    npz_origin_paths = [os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam2_frame_00001_equirect_cam4_matches.npz'),
                    os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam2_frame_00001_equirect_cam6_matches.npz'),
                    os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam4_frame_00001_equirect_cam6_matches.npz')]
    # 输出路径
    output_points = os.path.join(ls.DATA_ROOT,'Vis', "feature_points_visualization.png")
    output_errors = os.path.join(ls.DATA_ROOT, 'Vis', "feature_errors_visualization.png")
    
    print("=" * 60)
    print("特征点可视化测试")
    print("=" * 60)
    
    # 检查全景图是否存在
    if not os.path.exists(panorama_path):
        print(f"警告: 全景图不存在 {panorama_path}")
        print("请先运行拼接生成全景图")
        return
    
    # 1. 可视化特征点位置（不同相机用不同颜色）
    print("\n1. 生成特征点位置可视化...")
    visualize_feature_points_on_panorama(
        panorama_path=panorama_path,
        npz_paths=npz_origin_paths,
        calib_opt_path=calib_opt_path,
        output_path=output_points,
        use_ground_homography=True,
        ground_y_threshold=600,
        ground_params_path=ls.CALIB_GROUND_FILE
    )
    
    # 2. 可视化匹配误差（用连线和颜色表示误差大小）
    print("\n2. 生成匹配误差可视化...")
    visualize_feature_matching_errors(
        panorama_path=panorama_path,
        npz_paths=npz_origin_paths,
        calib_opt_path=calib_opt_path,
        output_path=output_errors,
        use_ground_homography=True,
        ground_y_threshold=600,
        ground_params_path=ls.CALIB_GROUND_FILE
    )
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"特征点位置图: {output_points}")
    print(f"匹配误差图: {output_errors}")
    print("=" * 60)

if __name__ == "__main__":
    main()
