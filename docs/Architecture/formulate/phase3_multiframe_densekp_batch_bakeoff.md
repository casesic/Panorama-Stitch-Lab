# Phase 3：Detector + DenseKP 跨帧 Batch Bake-off

## 结论

本轮没有达到 10 FPS。当前 detector ONNX 固定为 B=1，无法进行真实 detector
batch；DenseKP ONNX 虽支持动态 batch，但 B=2/4/8/16/32 在该真实多人负载上均未
优于 B=1，因此不应切换 production 默认策略。

本轮最有价值的发现不是 ORT batch，而是定位了旧 profile 中约 490 ms 的主要来源：
不跨 seam 的 ERP ROI 仍使用 NumPy 高级索引逐列 gather。改为数学等价的连续切片后，
ROI extraction 从约 440 ms/frame 降至约 16 ms/frame，像素 bitwise equal。

正式 benchmark 位于：

```text
data_process/Vis/densekp_phase3_batch_bakeoff_v2/benchmark.json
data_process/Vis/densekp_phase3_batch_bakeoff_v2/correctness_t4.json
```

测试使用 2213–2221 九个真实帧；正确性只运行一次真实顺序，性能测试将预加载 ERP
循环复用。每项 warmup=3、repeat=20，不包含 image load、ERP remap、模型初始化、
HMR2 或 C++ LM。

## Baseline 细分

正式 T=8 baseline 的均值如下。总时间中未单列的约 11 ms 包括 ROI copy、bbox 整理、
matching、坐标转换和 Python 调度。

| Component | ms/frame |
| --- | ---: |
| Detector preprocess | 4.72 |
| Detector ORT | 73.35 |
| Detector postprocess | 0.32 |
| DenseKP crop/affine | 3.39 |
| DenseKP normalization | 0.95 |
| DenseKP tensor pack | 1.86 |
| DenseKP ORT | 169.60 |
| DenseKP postprocess | 0.64 |
| Other delivery | 8.39 |
| Total delivery | 273.23 |

ROI fast slice 后，当前主要瓶颈已经变成 DenseKP ORT，其次是四次 B1 detector ORT。

## ONNX batch capability

| Model | Input | Batch | 结论 |
| --- | --- | --- | --- |
| YOLOX HumanArt end-to-end | `[1,3,640,640]` | static B1 | batch unavailable；需重新导出含 NMS 的图 |
| DenseKP | `[batch,3,256,256]` | dynamic | B1/2/4/8/16/32 均可执行 |

不能只改 YOLOX input shape：其 embedded NMS 输出也带固定 batch 语义。本轮没有在
production 中 hack ONNX graph。

## DenseKP batch-size bake-off

固定 T=8，单位为 amortized ms/frame：

| Batch | delivery mean | delivery p50 | DenseKP ORT mean | FPS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 174.3 | 173.3 | 96.2 | 5.74 |
| 2 | 208.5 | 209.7 | 125.7 | 4.80 |
| 4 | 215.7 | 214.4 | 132.7 | 4.64 |
| 8 | 236.3 | 235.0 | 153.9 | 4.23 |
| 16 | 225.5 | 225.6 | 142.7 | 4.43 |
| 32 | 219.2 | 217.6 | 136.0 | 4.56 |

因此 `RECOMMENDED_DENSEKP_BATCH=1`。阶段化调度（先完成 detector，再运行 DenseKP）
有时减少了两个 ORT session 交替产生的开销，但这不是模型 batch 收益。

## T bake-off

第二轮后半段出现明显长程计时漂移，因此以下采用保守的同阶段正式结果，不使用前半段
174.3 ms/frame 的乐观值作为 10 FPS 结论。`densekp_batch` 名称表示跨帧统一队列，
实测最优 batch size 仍为 B1。

| T | Mode | detector calls/frame | DenseKP calls/frame | delivery mean ms/frame | p50 | p90 | FPS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | 4.0 | 6.4 | 251.3 | 242.6 | 287.4 | 3.98 |
| 1 | phase-separated B1 | 4.0 | 6.4 | 284.9 | 236.9 | 392.3 | 3.51 |
| 2 | baseline | 4.0 | 6.4 | 255.7 | 247.4 | 271.1 | 3.91 |
| 2 | phase-separated B1 | 4.0 | 6.4 | 266.0 | 243.3 | 327.4 | 3.76 |
| 4 | baseline | 4.0 | 6.41 | 261.0 | 248.7 | 301.3 | 3.83 |
| 4 | phase-separated B1 | 4.0 | 6.41 | 275.4 | 275.0 | 298.5 | 3.63 |
| 8 | baseline | 4.0 | 6.43 | 262.9 | 263.0 | 283.8 | 3.80 |
| 8 | phase-separated B1 | 4.0 | 6.43 | 279.0 | 277.8 | 291.8 | 3.58 |

在同一长测内，同一路径前半段约 174 ms/frame、后半段约 279 ms/frame，说明当前环境
存在不可忽略的长程性能漂移。它不影响 gate 判断：两者都大于 100 ms/frame。
ORT 没有暴露可靠的 peak allocator counter，因此本轮没有伪造 peak VRAM 数字，JSON
中明确记录为 unavailable；T=8 实际运行未 OOM。

## Correctness

- 2213–2221：baseline 和 B16 candidate 均为 12 个 matched observations。
- 每帧 cam2/cam4 detection count、matched count 和 valid count 全部一致。
- 2214：cam2=1、cam4=4、matched=1、valid=113；median angle 为
  4.9186°（B1 baseline）与 4.9136°（B16）。
- batch 与 B1 的 DenseKP 最大坐标差为 0.6484 px；这是动态 batch 的数值差异，未改变
  finite/valid、匹配关系或 2213–2221 的 track 结构。
- 原 Phase 2.2 的主 Track 0 仍覆盖 2213–2221；输入匹配拓扑未变化。
- T=1 fallback、mixed camera、unequal person count 和 empty ROI 的 metadata restoration
  有独立单元测试。

因为 B>1 不仅没有性能收益，还引入了亚像素数值差异，所以本轮不把它提升为 production
默认。当前 production fallback 保持原样。

## 10 FPS 与流水线估算

保守 steady-state delivery 为约 251–279 ms/frame，即约 3.6–4.0 FPS。即使与平均
58.7 ms/frame 的 CPU LM 完全流水并行：

```text
pipeline period ≈ max(delivery, CPU LM) ≈ 251–279 ms/frame
estimated throughput ≈ 3.6–4.0 FPS
```

因此当前不应进入复杂 async pipeline integration。下一步应先解决：

1. 重新导出并验证真正支持 batch 的 detector，减少固定的 4 calls/frame；
2. 分析 DenseKP dynamic batch 在 CUDA EP 上随 B 增大反而变慢的原因，包括 H2D、
   provider partition 和 kernel shape；
3. 在隔离 GPU、锁定功耗/时钟的环境复测长程漂移；
4. 只有 delivery 稳定低于 100–150 ms/frame 后，再做 GPU delivery 与 CPU LM 流水线。

## 测试

```text
PYTHONPATH=src:Circular_Video/01_recv/20260708 pytest -q \
  Circular_Video/01_recv/20260708/densekp_delivery/tests \
  src/lib_pano/fitting_io/tests src/lib_pano/human_fitting/tests
```

结果：`32 passed`。

## Decision flags

```text
DETECTOR_BATCH_READY=FAIL
DENSEKP_BATCH_READY=FAIL

T2_BATCH_READY=FAIL
T4_BATCH_READY=FAIL
T8_BATCH_READY=FAIL

DELIVERY_BATCH_GATE1=FAIL
DELIVERY_BATCH_GATE2=FAIL
DELIVERY_10FPS_GATE=FAIL

SEQUENCE_BATCH_CORRECTNESS=PASS
FRAME2214_BATCH_REGRESSION=PASS

BEST_DELIVERY_MS_PER_FRAME=251.3_CONSERVATIVE_BASELINE
BEST_DELIVERY_FPS=3.98_CONSERVATIVE_BASELINE
OBSERVED_OPTIMISTIC_B1_SCHEDULE_MS_PER_FRAME=174.3

RECOMMENDED_WINDOW_T=1
RECOMMENDED_DETECTOR_BATCH=1
RECOMMENDED_DENSEKP_BATCH=1

ESTIMATED_PIPELINED_10FPS_FEASIBLE=NO

CPP_LM_MATH_CHANGED=NO
ERP_REMAP_INCLUDED_IN_TARGET=NO
```

## Non-changes

ERP remap、DenseKP validity、跨相机匹配数学、tracking policy、HMR2、C++ runtime
preparation、C++ LM objective/Jacobian/normal equations/Schur/solver/loss/tolerance 均未修改。

## 修改文件

- `densekp_delivery/runners/ort_runner.py`：增加跨 image 的 DenseKP crop queue、动态
  batch 执行及细粒度 detector/DenseKP profiling；B1 fallback 保留。
- `densekp_delivery/runtime.py`：增加最小 `run_batch()` 转发接口。
- `densekp_delivery/pipeline.py`：增加只供实验脚本使用的 preloaded-ERP window 调度与
  metadata restoration。
- `densekp_delivery/overlap_roi.py`：不跨 seam ROI 使用等价连续切片，消除高级索引
  gather；跨 seam 仍走原逻辑。
- `CameraHMR/detectors/rtmlib_yolox_detector.py`：仅增加 preprocess/ORT/postprocess
  分项计时。
- `densekp_delivery/tests/test_batch_restoration.py`：覆盖 frame/camera/ROI 顺序、空结果、
  人数不等和 ROI bitwise parity。
- `debug087_phase3_multiframe_densekp_batch_bakeoff.py`：独立 T/batch-size benchmark、
  correctness artifact 和 gate 输出。
- 本文档：记录能力审计、真实数据、结论与 non-changes。
