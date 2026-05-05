# Pipeline 流程总览

推荐入口是 `src/script/test_function/run_pipeline.py`。它负责 step 注册、config 校验、命令构建、`--dry_run` 和 `--list_steps`。各个 debug 脚本仍然可以直接运行。

常用命令：

```bash
python src/script/test_function/run_pipeline.py --list_steps
python src/script/test_function/run_pipeline.py --config data_process/Output/config.json --steps 020,021,035,040,041,025,026,024,038,043,029,030,031,037 --dry_run
```

## A. 预处理模块

预处理阶段负责准备帧、crop、检测框、姿态、mask 和 fitting 初始化数据。

| Step | 脚本 | 输入 | 输出 | 作用 | 是否可选 |
| --- | --- | --- | --- | --- | --- |
| 020 | `debug020_run_export.py` | `reviewed` | `fisheye_dir` | 导出 reviewed 鱼眼帧。 | 否 |
| 021 | `debug021_run_fisheye_to_equirect.py` | `fisheye_dir`, `camera` | `equirect_rotate_dir` | 鱼眼图转 rotated ERP。 | 否 |
| 035 | `debug035_annotate_click_points.py` | `equirect_rotate_dir`, `reviewed` | `click_points` | 给 reviewed 帧标注 click points，辅助检测。 | 是 |
| 040 | `debug040_detect_grounding_dino.py` | `equirect_rotate_dir`, 可选 `click_points` | `bbox_dir` | 使用 GroundingDINO 检测人体 bbox。 | 否 |
| 041 | `debug041_smooth_bboxes.py` | `bbox_dir`, `equirect_rotate_dir` | `bbox_smooth_dir` | 平滑 bbox 序列，可保存可视化。 | 是 |
| 025 | `debug025_crop_equirect_rotate_from_bboxes.py` | `bbox_dir` 或 `crop_bbox_dir`, `equirect_rotate_dir` | `equirect_crop_rotate_dir` | 从 ERP 图裁剪人体 crop，供 pose/HMR/segmentation 使用。 | 否 |
| 026 | `debug026_run_rtmlib_erp.py` | `equirect_crop_rotate_dir` | `rtm_erp_dir` | 在 ERP crop 上运行 RTMLib pose。 | 使用 RTMW 时可选 |
| 027 | `debug027_run_rtmw_mmpose.py` | `equirect_crop_rotate_dir`, `pose_config`, `pose_checkpoint` | `rtmw_erp_dir` 或 fallback `rtm_erp_dir` | 在 ERP crop 上运行 RTMW MMPose。 | 可作为 pose 替代路径 |
| 024 | `debug024_run_hmr_convert.py` | `4d_humans_dir` | `hmr_cache_dir` | 转换 HMR / 4D-Humans 输出，供 fitting 初始化。 | 否 |
| 038 | `debug038_seg_sam3.py` | `equirect_crop_rotate_dir`, `sam3_model_dir` 或 `sam_model` | `seg/equirect_crop_rotate_sam3` | 使用 SAM3 对 ERP crop 做人体分割。 | 使用 silhouette fitting 时通常需要 |
| 043 | `debug043_mask_translation_fusion.py` | SAM3 masks、crop images、crop bboxes | `seg/equirect_crop_rotate_sam3_fused_translation` | 用双相机 translation baseline 融合 SAM3 masks。 | 是 |

注意：

- step 038 如果 config 里没有 `sam3_model_dir` 或 `sam_model`，会正常报错，这是轻量校验的一部分。
- step 043 是可选步骤。如果设置 `use_mask_fusion_for_fitting=True`，step 029 会使用 fused mask 输出。

## B. 重建模块

重建阶段负责 SMPL-X fitting 和 fitting 结果检查。

| Step | 脚本 | 输入 | 输出 | 作用 | 是否可选 |
| --- | --- | --- | --- | --- | --- |
| 029 | `debug029_run_dual_erp_fitting.py` | pose cache、HMR cache、crop bboxes、segmentation masks | `fitting_dir` | 执行 dual ERP SMPL-X fitting。 | 否 |
| 030 | `debug030_vis_dual_erp.py` | `fitting_dir`, pose cache, ERP images | `vis_dir` | 可视化 fitting 结果，并生成 invalid frame review 线索。 | 是 |

step 029 的 segmentation 路径选择优先级：

1. 如果提供 `fitting_seg_dir`，直接使用它作为 `debug029 --seg_dir`。
2. 如果 `use_mask_fusion_for_fitting=True`，使用 `mask_fusion_output_dir`；未提供时默认 `seg/equirect_crop_rotate_sam3_fused_translation`。
3. 如果提供 `fitting_seg_subdir`，使用 `Path(seg_dir) / fitting_seg_subdir`。
4. 默认使用 `seg/equirect_crop_rotate_sam3`。
5. 如果旧数据确实需要旧路径，可以显式配置 `legacy_fitting_seg_subdir`，例如 `equirect_rotate_crop`。

默认路径已经不再是旧的 `equirect_rotate_crop`。

## C. Composite 效果验证模块

这一阶段用于验证人体区域几何和补全效果，不一定代表最终合成结果。

| Step | 脚本 | 输入 | 输出 | 作用 | 是否可选 |
| --- | --- | --- | --- | --- | --- |
| 031 | `debug031_batch_semantic_flow.py` | dual fitting、ERP images | `flow_dir` | 基于 SMPL-X vertices 生成 semantic flow。 | 是 |
| 037 | `debug037_composite_human.py` | segmentation masks、semantic flow、ERP images、bboxes | `composite_dir` | 跨相机 warp / complete / blend 人体，用于效果验证。 | 是 |

semantic flow 更适合作为人体区域的几何先验和验证信号，不应被理解为唯一最终输出路径。

## D. 最终光流 / 合成模块

最终 synthesis 方向目前主要体现在 flow 和 warp 相关库模块中，而不是一个固定 pipeline step：

- `src/lib_pano/flow/raft_flow.py`
- `src/lib_pano/flow/warp_strategy.py`
- `src/lib_pano/flow/roi_process.py`
- `src/lib_pano/flow/mesh_warp_field.py`
- `src/lib_pano/flow/distance_warp_field.py`
- `src/lib_pano/flow/debug_writer.py`

除非 `run_pipeline.py` 后续加入明确 step，否则不要假设已有固定的最终 dense optical flow / synthesis step id。
