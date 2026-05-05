# Debug 脚本清单

debug 脚本仍然可以作为直接入口使用，但公共逻辑已经逐步迁移到 `src/lib_pano/`。推荐可重复流程使用 `src/script/test_function/run_pipeline.py`。

## A. Pipeline Runner

- `run_pipeline.py`：step registry、config validation、命令构建、`--dry_run`、`--list_steps`。

## B. 预处理：导帧 / 投影 / 裁剪

- `debug020_run_export.py`：导出 reviewed frames。
- `debug021_run_fisheye_to_equirect.py`：鱼眼图转 rotated ERP。
- `debug025_crop_equirect_rotate_from_bboxes.py`：根据 bbox JSON 裁剪 ERP 图。

## C. 预处理：标注 / review / invalid frame

- `debug034_generate_review.py`：生成 review 列表。
- `debug035_annotate_click_points.py`：给 reviewed frames 标注 click points。
- `debug036_review_invalid_images.py`：review 并标记 invalid images。
- `debug022_merge_invalid_jsons.py`：合并 invalid JSON 的工具脚本。

## D. 预处理：检测 / 姿态 / 分割 / mask fusion

- `debug040_detect_grounding_dino.py`：检测人体 bbox。
- `debug041_smooth_bboxes.py`：平滑 bbox 序列。
- `debug026_run_rtmlib_erp.py`：在 ERP crop 上运行 RTMLib pose。
- `debug027_run_rtmw_mmpose.py`：在 ERP crop 上运行 RTMW MMPose。
- `debug038_seg_sam3.py`：运行 SAM3 crop segmentation。
- `debug043_mask_translation_fusion.py`：用双相机 translation baseline 融合 SAM3 masks。

## E. 重建：SMPL-X / fitting / 可视化

- `debug024_run_hmr_convert.py`：转换 HMR / 4D-Humans 输出。
- `debug029_run_dual_erp_fitting.py`：dual ERP SMPL-X fitting。
- `debug030_vis_dual_erp.py`：fitting 可视化和 invalid frame 检查。

## F. Composite 效果验证：semantic flow / human composite

- `debug031_batch_semantic_flow.py`：根据 fitted SMPL-X geometry 生成 semantic flow。
- `debug037_composite_human.py`：跨相机 warp、补全、blend 人体，用于验证 semantic flow 和人体补全效果。

composite 输出主要用于检查 semantic flow 和人体补全行为，不一定是最终合成结果。

## G. Sparse Correspondence / 过滤 / 调试

- `debug045_build_sparse_smpl_correspondence.py`：构建 sparse SMPL correspondence。
- `debug046_vis_sparse_corr.py`：统一 sparse correspondence 可视化入口。
- `debug046_vis_sparse_corr_fisheye.py`：fisheye 可视化兼容 wrapper。
- `debug047_vis_sparse_corr_erp.py`：ERP 可视化兼容 wrapper。
- `debug050_draw_bad_regions_to_vertex_ids.py`：交互式标注 bad regions 并输出 vertex ids。
- `debug051_npz_corr_to_json.py`：将 correspondence NPZ 转成 JSON。

sparse correspondence 是调试、可视化和过滤工具链，不是主流程必需模块。

## H. 最终光流 / 合成相关模块

这些是库模块，不是当前固定 pipeline step：

- `src/lib_pano/flow/raft_flow.py`
- `src/lib_pano/flow/warp_strategy.py`
- `src/lib_pano/flow/roi_process.py`
- `src/lib_pano/flow/mesh_warp_field.py`
- `src/lib_pano/flow/distance_warp_field.py`
- `src/lib_pano/flow/debug_writer.py`

## I. 其它实验 / 工具脚本

- `debug015_make_video.py`：视频生成工具。
- `debug019_batch_flow_stitch.py`：flow stitch 实验 / 批处理工具。
- `debug042_compare_pose_vis.py`：姿态可视化对比。
- `debug_inspect_kp_conf.py`：关键点置信度检查工具。

这些脚本可能是实验性、legacy 或特定用途工具。可重复主流程优先使用 `run_pipeline.py` 和已登记 step。
