# DenseKP overlap ROI 与多人跨相机匹配

## Production 数据流

```text
cam2/cam4 fisheye
  → full ERP（保持现有投影）
  → calibration valid masks
  → strict overlap = valid_cam2 ∩ valid_cam4
  → periodic dilation + overlap component ROI
  → ROI detector / DenseKP
  → crop-local 点恢复到 full ERP
  → DenseKP 球面代价与一对一 assignment
  → densekp_pairs_v2 persons[]
```

`strict overlap` 只决定最终有效点；检测 ROI 在它周围扩张 160 px，为检测器保留人体上下文。ERP 的 x 轴按周期域处理：跨 `x=0/W` 的区域用连续的 unwrapped crop 表示，恢复时执行 `x_global=(x_local+x_start)%W`。极区把两条 overlap band 连通的问题只在 ROI component 提取时处理，不改变最终 validity。

## 多人匹配

每个 cam2/cam4 候选对只使用相同 DenseKP semantic ID。有效行必须同时满足有限值、ERP 范围以及两相机各自 strict valid mask。点先转换成单位球面射线，pair cost 为有效行角距离的中位数：

```text
cost(i,j) = median_k acos(ray_cam2(i,k) · ray_cam4(j,k))
```

少于 8 个有效点或中位角误差超过 12° 的 pair 被 gating。assignment 先最大化匹配人数，再最小化总 cost，并保持确定性；每个 detection 最多使用一次，无法匹配者进入 unmatched 列表。`raw_scores/keypoint_value` 不参与 cost、validity、loss 或 weighting。

## JSON v2

顶层不再提供含糊的单一 `pairs`：

```json
{
  "schema_version": "densekp_pairs_v2",
  "coord_space": "erp_rotate",
  "matched_person_count": 2,
  "persons": [
    {
      "person_id": 0,
      "cam2_person_index": 1,
      "cam4_person_index": 0,
      "point_ids": [0],
      "pairs": [[100.0, 200.0, 110.0, 201.0]],
      "valid": [1]
    }
  ],
  "unmatched_cam2_person_indices": [],
  "unmatched_cam4_person_indices": [1]
}
```

复数 loader `load_densekp_delivery_observations()` 返回全部人物。旧 singular loader 继续支持 v1 和恰好一人的 v2；v2 含多人时明确报错，绝不静默选择 `persons[0]`。当前 C++ staged-LM runner 也保持单人边界：遇到多人 v2 会要求先完成身份选择/跟踪。

## 性能与边界

profile 新增 full ERP pixels、ROI pixels/ratio、ROI 数量和 ROI 构建时间。当前实现仍先生成 full ERP，再裁 ROI；CUDA ROI-only remap 不在本阶段。

本阶段未修改 C++ LM 数学、CUDA Jacobian、loss、solver 或 tolerance，也未实现逐人 initial state、fixture 和独立 LM orchestration。
