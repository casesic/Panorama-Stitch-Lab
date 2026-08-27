# Phase 1.5：DenseKP Detection ROI 紧凑化归因

## 结论

当前 production 继续使用 `component_bbox`，而不是把 `compact_strips` 升为默认。原因不是 compact geometry 无效，而是当前 YOLOX 模型输入固定为 `[1,3,640,640]`：strip 数增加会线性增加 detector inference 次数，实测损失远大于 crop 像素减少带来的收益。

## 91.11% 的来源

固定 calibration 下：

```text
full ERP pixels             16,588,800
strict overlap pixels        4,685,658  (28.25%)
context detection pixels     7,723,216  (46.56%)
2 component bbox pixels     15,114,240  (91.11%)
bbox/detection fill ratio        51.10%
```

overlap 是两条全纬度、随纬度弯曲的带状区域。每条 context mask 实际约占对应 bbox 的一半，但 axis-aligned bbox 必须覆盖弯曲带两侧的空背景。两条 band 是不同空间区域，不是 `x=0/W` 两侧的同一个 seam component；把它们 circular-shift 成一个框，最小周期区间反而接近整幅宽度。

production 两个 ROI 均为 `2624×2880`，x 段分别为 `[118,2742]` 和 `[3019,5643]`，fill ratio 分别为 50.44% 和 50.20%。每个 ROI 对每台相机调用一次 detector，所以双相机共 4 次 detector inference。

## 候选方案

- Circular single ROI：两条 band 并非 seam 的同一块，合并后接近 full ERP，否决。
- Packed atlas：可把调用降为双相机共 2 次，但拼接边界和整体缩放破坏人物尺度；2214 正确匹配消失，否决。
- Compact horizontal strips：4 个纬度 band × 2 个 component，共 8 ROI；源像素降至 76.44%，fill ratio 升至 60.91%，并通过 periodic restore 与 dedup 保持 2214 匹配。但双相机 detector 调用增至 16 次，性能显著下降，因此只保留为 attribution/diagnostic candidate。

compact strips 在 full ERP 恢复后使用 periodic bbox NMS。除了普通 periodic IoU，还使用扩张框的 intersection-over-smaller-box，去除 tile 边界产生的同一人物 full/partial 重复框。DenseKP runner 当前将 detector 与 DenseKP 绑定，因此去重发生在 DenseKP 后；这仍会产生重复 DenseKP 推理，是该候选的已知限制。

## 2214 A/B（persistent runtime，warmup 后 3 次）

| Metric | component bbox | compact strips |
|---|---:|---:|
| Full ERP pixels | 16,588,800 | 16,588,800 |
| ROI count | 2 | 8 |
| ROI source pixels | 15,114,240 | 12,680,592 |
| ROI/full ERP | 91.11% | 76.44% |
| Detection-mask fill ratio | 51.10% | 60.91% |
| 双相机 detector calls | 4 | 16 |
| Detector p50 | 54.45 ms | 210.76 ms |
| Detector p90 | 58.95 ms | 211.18 ms |
| DenseKP p50 | 65.35 ms | 180.80 ms |
| Total delivery p50 | 736.57 ms | 920.78 ms |
| Total delivery p90 | 739.97 ms | 922.42 ms |
| 2214 matched persons | 1 | 1 |
| matched valid DenseKP | 113/138 | 115–116/138 |
| median angular error | 4.92° | 4.51° |

`detector_inference_ms` 是 RTMLib `detect()` 的可靠总计，包含其内部 preprocess、固定 640×640 inference 与 decode；当前 wrapper 没有可靠的独立三段 timing，因此没有伪造细分数字。

## Production 决策

`component_bbox` 仍为唯一默认路径。ROI geometry 通过 `DenseKPRuntime` 的 calibration static-geometry cache 常驻；steady state 的 ROI geometry lookup 约 0.02 ms，不再逐帧 connected component/dilation。

新增 profile 字段包括 ROI specification、source/mask pixels、fill ratio、detector call count，以及 dedup 前后人物数。`overlap_roi_cover.jpg` 展示 strict mask、detection mask 和最终 rectangles；每个实际 detector crop 单独保存。

下一步应走 detector-input 路线，而不是继续增加 tiles：要么让 detector 支持真正的动态 batch/多 ROI batch，要么设计不会改变人物尺度的单次 ROI atlas。CUDA ROI-only ERP remap 只有在 compact geometry 成为 production 后才有直接收益；本阶段不应继续实现。
