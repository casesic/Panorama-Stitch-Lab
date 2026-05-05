# 数据流与目录约定

本文记录当前 pipeline 中主要数据产物的关系。具体 dataset 名会随 config 变化，下面用 `data_process/Output/example_dataset/...` 作为占位路径。

## 主要数据产物

| 数据 | 典型 config 字段 / 目录 | 产生步骤 | 消费步骤 |
| --- | --- | --- | --- |
| reviewed frames | `reviewed` | review 工具 / 手工整理 | 大多数预处理步骤 |
| 鱼眼帧 | `fisheye_dir` | step 020 | step 021 |
| rotated ERP 帧 | `equirect_rotate_dir` | step 021 | 检测、标注、fitting 可视化、semantic flow |
| click points | `click_points` | step 035 | step 040 可选使用 |
| bbox JSON | `bbox_dir` | step 040 | step 041、step 025 |
| 平滑 bbox JSON | `bbox_smooth_dir` | step 041 | 可通过 `crop_bbox_dir` 传给 step 025 |
| ERP crops | `equirect_crop_rotate_dir` | step 025 | pose、SAM3、HMR / 4D-Humans、mask fusion |
| RTMLib pose cache | `rtm_erp_dir` | step 026 | step 029 |
| RTMW pose cache | `rtmw_erp_dir` | step 027 | 可作为 pose cache 路径 |
| SAM3 masks | `seg/equirect_crop_rotate_sam3` | step 038 | step 029、step 043、step 037 |
| fused masks | `seg/equirect_crop_rotate_sam3_fused_translation` | step 043 | 可作为 step 029 输入 |
| HMR cache | `hmr_cache_dir` | step 024 | step 029 |
| dual fitting | `fitting_dir`, `fitting_dual_dir` | step 029 | step 030、step 031 |
| fitting visualization | `vis_dir` | step 030 | review / invalid flow |
| semantic flow | `flow_dir` | step 031 | step 037 |
| human composite | `composite_dir` | step 037 | 质量验证 |
| sparse correspondence | 项目内 correspondence 相关目录 | debug045/046/047/050/051 | 调试和过滤 |

## 关键默认路径

SAM3 segmentation：

```text
seg/equirect_crop_rotate_sam3
```

Mask translation fusion：

```text
seg/equirect_crop_rotate_sam3_fused_translation
```

Fitting 默认 mask 输入：

```text
seg/equirect_crop_rotate_sam3
```

使用 fused mask 做 fitting：

```json
{
  "use_mask_fusion_for_fitting": true,
  "mask_fusion_output_dir": "data_process/Output/example_dataset/seg/equirect_crop_rotate_sam3_fused_translation"
}
```

旧 fitting mask 路径：

```json
{
  "legacy_fitting_seg_subdir": "equirect_rotate_crop"
}
```

只有旧数据确实使用该路径时才建议配置 `legacy_fitting_seg_subdir`。新配置优先使用：

```json
{
  "fitting_seg_subdir": "equirect_crop_rotate_sam3"
}
```

或者直接使用：

```json
{
  "fitting_seg_dir": "data_process/Output/example_dataset/seg/equirect_crop_rotate_sam3"
}
```

## Composite 效果验证

`debug031_batch_semantic_flow.py` 将 semantic flow 写入 `flow_dir`。`debug037_composite_human.py` 读取 semantic flow、segmentation masks、bboxes 和 ERP images，输出到 `composite_dir`。

这条分支用于验证跨相机人体 warp、补全和 blend 效果，不一定是最终 synthesis 分支。

## 最终光流 / 合成

最终合成更可能使用 dense optical flow、RAFT 和 warp strategy 相关模块。当前相关模块包括：

- `src/lib_pano/flow/raft_flow.py`
- `src/lib_pano/flow/warp_strategy.py`
- `src/lib_pano/flow/roi_process.py`
- `src/lib_pano/flow/mesh_warp_field.py`
- `src/lib_pano/flow/distance_warp_field.py`

除非 `run_pipeline.py` 中加入明确 step，否则本文不编造固定的最终 synthesis step。

## Sparse Correspondence

sparse correspondence 输出属于调试、可视化、过滤工具链。它可以帮助检查 SMPL correspondence、排除 bad vertex regions，但不是主流程必需结果。
