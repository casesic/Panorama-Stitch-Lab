# Phase 4：Track-Guided Sparse Delivery

## 结论

Track-guided sparse delivery 的性能路线成立，但当前身份 gate 尚不能封板为 production。

在不包含 image I/O、ERP remap、模型初始化和 HMR2 的条件下，K=3 的实际重复调用结果为：

```text
mixed delivery = 84.94 ms/frame = 11.77 FPS
detector calls = 4.00 → 1.33 / frame
DenseKP calls  = 6.44 → 3.89 / frame
```

它通过 delivery 10 FPS 门槛；但串行叠加 CPU LM 后仍约 151.35 ms/frame，只有约
6.61 FPS。主 Track 0 的独立真实闭环（实际调用 detector-free DenseKP 和 CPU C++ LM）
为 8/8 LM success、8/8 warm-state bitwise exact，串行均值 130.06 ms/frame、p50
98.58 ms/frame，即均值约 7.69 FPS。

不能立即进入 production 的原因是 2214 身份歧义：Full baseline 中 Track 1 已不在
matched 集合，guided 路径仍以 138 valid points、低 cross-camera/temporal error 接受
了该 crop。它可能是 full detector 漏检，也可能是错误身份；现有几何 gate 无法区分，
因此不能 silent accept。

## 两条调用路径

```text
FULL_DETECTION
  preloaded ERP cam2/cam4
  → 2 overlap ROI × 2 cameras
  → YOLO B1
  → all candidate DenseKP B1
  → existing cross-camera angular matching
  → temporal association
  → HMR2 only for new/reinit track
  → C++ preparation + LM
```

```text
TRACK_GUIDED
  active TrackState
  → previous matched DenseKP geometry
  → ERP_ROTATE to normal ERP
  → seam-aware circular bbox + margin
  → cam2/cam4 detector-free DenseKP B1
  → existing finite/bounds/strict-mask/angular gate
  → previous-observation temporal geometry gate
  → previous C++ state warm start
  → C++ preparation + LM

  any explicit failure
  → same-frame FULL_DETECTION recovery
```

## Guided bbox 语义

审计发现当前 C++ `optimized_densekp` 投影范围会显著收缩，不能作为可靠人体 envelope。
因此第一版明确使用 TrackState 中上一帧已匹配的 DenseKP observation geometry，而不是
上一帧 detector bbox，也不使用收缩后的 optimized points。未来 C++ 若输出可靠的 final
support/body extent，应再切回 fitted-body projection。

步骤如下：

1. 上一帧 point pairs 位于 ERP_ROTATE；用既有旋转矩阵的转置转换回 normal ERP。
2. 对有效、finite 点的 x 计算 minimum circular interval，避免 seam 人体变成整幅宽度。
3. x/y 按比例和 minimum pixel margin 扩张，clip y、wrap x。
4. crop 使用 Phase 3 保留的 contiguous fast slice；跨 seam 时仍用原周期 gather。
5. crop 内 bbox 直接进入相同 DenseKP affine/preprocess/ORT/decode，不调用 detector。

## Margin bake-off

真实转换为 2213→2214 … 2220→2221，共 10 个 active-track attempts，其中 9 个在下一帧
Full baseline 中存在。四个 margin 对正确人物均为 9/9 recall：

| Margin | present recall | all guided accepted | mean valid | mean crop/full ERP |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 100% (9/9) | 10/10 | 118.6 | 1.25% |
| 20% | 100% (9/9) | 10/10 | 115.3 | 1.67% |
| 30% | 100% (9/9) | 10/10 | 115.7 | 2.17% |
| 40% | 100% (9/9) | 10/10 | 110.6 | 2.74% |

选择 10%，因为在相同 recall 下 crop 最小。样本只有 9 个正确人物 transition，尚不足以
把 ≥95% recall gate 推广到 production 数据分布。

## Detection interval bake-off

每个 K 实际执行 1 cycle warmup + 7 steady cycles，共 63 个 frame-equivalent samples。
ERP 预加载；没有重复使用预先记录的耗时。`serial E2E` 为 delivery 实测加 Phase 2.2
同帧 CPU LM workload，属于估算；另有主 Track 0 的真实 C++ 闭环。

| K | Full % | detector calls/frame | DenseKP calls/frame | delivery mean | p50 | p90 | delivery FPS | serial E2E estimate | FPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100.0% | 4.00 | 6.44 | 170.63 ms | 163.29 | 193.70 | 5.86 | 237.03 ms | 4.22 |
| 3 | 33.3% | 1.33 | 3.89 | 84.94 ms | 64.35 | 176.15 | 11.77 | 151.35 ms | 6.61 |
| 5 | 22.2% | 0.89 | 3.44 | 70.47 ms | 32.70 | 176.31 | 14.19 | 136.87 ms | 7.31 |
| 10 | 11.1% | 0.44 | 3.00 | 56.33 ms | 32.82 | 157.62 | 17.75 | 122.73 ms | 8.15 |

推荐 K=3，而不是最快的 K=10：K=3 已通过 delivery 10 FPS，detector 调用下降
66.7%，新人物理论最坏发现延迟仅 2 帧。K=5/10 更快，但发现延迟分别增至 4/9 帧。

K=3 drift 检查：

```text
first 10  = 94.35 ms/frame
middle 10 = 82.94 ms/frame
last 10   = 82.82 ms/frame
```

没有复现 Phase 3 后半段持续恶化。

## FULL / GUIDED / RECOVERY timing

- FULL mean：约 168–171 ms/frame。
- GUIDED mean：约 42.9 ms/frame；单 track 常见约 32 ms，双 track 约 64 ms。
- 实际 sequence 没有由现有 gate 触发的 recovery，因此没有虚构真实 recovery p50/p90。
- recovery 实现会把 failed guided cost 与 same-frame full cost 全部计入，并有单元测试。

## 主 Track 0：真实 DenseKP + C++ LM 闭环

K=3，2214–2221：

```text
LM success       = 8/8
warm state exact = 8/8
CPP preparation  ≈ 0.48–1.21 ms
CPP LM           ≈ 41.47–53.47 ms
serial mean      = 130.06 ms/frame
serial p50       = 98.58 ms/frame
serial p90       = 228.72 ms/frame
serial FPS       = 7.69
```

这证明 guided observation 可实际进入原 C++ runtime preparation/LM，且没有改变 warm
state、solver 或数学。但它只验证主 Track 0，不等价于完整 multi-person production
orchestration。

## Correctness 与未解决项

通过：

- ERP seam-aware bbox 单元测试；
- margin expansion；
- active normal frame detector calls=0；
- periodic FULL；
- explicit guided failure → same-frame FULL recovery；
- synthetic periodic new-person discovery；
- raw score/confidence 不进入 acceptance；
- guided path 不调用 HMR2；
- 主 Track 0 2213–2221 持续且 8 次 warm state exact；
- C++ LM 8/8 success。

未通过 production gate：

- 2214 Track 1 在 Full baseline 缺失，但 guided geometry 被接受；
- 当前没有 ReID，现有 cross-camera + temporal geometry 无法证明它仍是同一物理人物；
- K=3 在 2220 新出现的 Track 2 要到 2222 periodic frame 才会发现，而数据在 2221
  结束；这符合两帧延迟上界，但本短序列无法实证 discovery；
- 未生成完整多人 guided visual debug panel；歧义已保存为 JSON，不作静默 PASS。

歧义记录：

```text
data_process/Vis/densekp_phase4_track_guided_2213_2221_v2/
  guided_failure_cases.json
```

## 10 FPS 判断

```text
normal guided delivery       ≈ 42.9 ms/frame  PASS
K=3 mixed delivery           = 84.94 ms/frame PASS (11.77 FPS)
K=3 serial E2E estimate      = 151.35 ms/frame FAIL
main Track actual serial E2E = 130.06 ms/frame FAIL (7.69 FPS)
```

如果 GPU delivery 与 CPU LM 真正异步并行，K=3 理论 period 约为：

```text
max(84.94 ms delivery, ~66.4 ms CPU LM workload)
≈ 84.94 ms/frame
≈ 11.77 FPS
```

因此 `ASYNC_10FPS_CANDIDATE=YES`，但在身份歧义解决前不应开始 production async
integration。

## Artifacts

- 正式 63-sample benchmark：
  `data_process/Vis/densekp_phase4_track_guided_2213_2221_v2/phase4_bakeoff.json`
- 主 Track 0 C++ 闭环：
  `data_process/Vis/densekp_phase4_main_track_cpp_integration_v3/phase4_bakeoff.json`
- 身份歧义：
  `data_process/Vis/densekp_phase4_track_guided_2213_2221_v2/guided_failure_cases.json`

## Tests

```text
PYTHONPATH=src:Circular_Video/01_recv/20260708 pytest -q \
  src/lib_pano/fitting_io/tests \
  src/lib_pano/human_fitting/tests \
  Circular_Video/01_recv/20260708/densekp_delivery/tests
```

结果：`38 passed`。

## Decision flags

```text
TRACK_GUIDED_BBOX_READY=PASS
ERP_SEAM_GUIDED_BBOX=PASS

GUIDED_DENSEKP_READY=PASS
GUIDED_GEOMETRY_GATE=FAIL

SAME_FRAME_FULL_RECOVERY=PASS
PERIODIC_FULL_DETECTION=PASS
NEW_PERSON_DISCOVERY_PRESERVED=PASS_SYNTHETIC_ONLY

TRACK_ID_PERSISTENCE_GUIDED=PASS_MAIN_TRACK_ONLY
MATCHED_ONLY_HMR2_PRESERVED=PASS
GUIDED_HMR2_CALLS_ZERO=PASS
PREVIOUS_LM_WARM_START_PRESERVED=PASS

BEST_DETECTION_INTERVAL=3

BASELINE_DETECTOR_CALLS_PER_FRAME=4.00
GUIDED_DETECTOR_CALLS_PER_FRAME=1.33

BASELINE_DENSEKP_CALLS_PER_FRAME=6.44
GUIDED_DENSEKP_CALLS_PER_FRAME=3.89

GUIDED_DELIVERY_MS_PER_FRAME=42.9
MIXED_DELIVERY_MS_PER_FRAME=84.94
SERIAL_E2E_MS_PER_FRAME=130.06_MAIN_TRACK_ACTUAL

TRACK_GUIDED_DELIVERY_10FPS_GATE=PASS
SERIAL_END_TO_END_10FPS_GATE=FAIL

ASYNC_10FPS_CANDIDATE=YES

DENSEKP_BATCH_ENABLED=NO
DETECTOR_BATCH_ENABLED=NO

CPP_LM_MATH_CHANGED=NO
ERP_REMAP_INCLUDED_IN_TARGET=NO
```

## 修改文件

- `densekp_delivery/runners/ort_runner.py`：增加 detector-free `run_bboxes()`；复用原
  DenseKP preprocessing/ORT/decode，保持 B1。
- `densekp_delivery/runtime.py`：增加最小 bbox inference 转发。
- `track_guided_delivery.py`：seam-aware bbox、guided crops、现有 cross-camera 与 temporal
  gate、失败原因枚举。
- `track_guided_policy.py`：FULL/GUIDED/RECOVERY 调度和 same-frame fallback。
- `matched_person_tracks.py`：只增加轻量 guided orchestration metadata，不改 frame75/beta。
- `test_track_guided_delivery.py`：seam、margin、skip detector、periodic、recovery、new
  person 与 confidence invariant 测试。
- `debug088_phase4_track_guided_sparse_delivery.py`：真实 margin/K/timing 和可选主 Track
  C++ 闭环。
- 本文档与 benchmark artifacts。

## Non-changes

ERP remap、Phase 3 ROI fast slicing、DenseKP 模型/B1/validity、cross-camera matching
数学、tracking identity 语义、HMR2 cold-start-only、C++ runtime preparation、LM
objective/Jacobian/loss/solver/stage/tolerance/CUDA evaluator 均未修改。

