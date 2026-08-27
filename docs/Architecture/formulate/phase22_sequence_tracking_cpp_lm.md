# Phase 2.2：真实连续 DenseKP tracking + C++ staged-LM

## Sequence selection

数据集：`20260320_riding_dataset3`。

2214 附近实际只有以下帧同时存在 cam2/cam4 鱼眼图：

```text
2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221
```

因此真实测试为连续 9 帧；没有虚构第十帧。结果位于：

```text
data_process/Vis/densekp_phase22_sequence_2213_2221_v2/
```

## DenseKP delivery

九帧均成功生成 `densekp_pairs_v2`、ERP_ROTATE observations：

| Frame | matched persons |
| ---: | ---: |
| 2213 | 2 |
| 2214–2219 | 每帧 1 |
| 2220 | 2 |
| 2221 | 2 |

总检测：cam2 `14`、cam4 `44`；真正通过 spherical gating 的 matched observations 为 `12`。unmatched detections 没有进入 HMR2 或 LM。

DenseKP runtime、detector/session、valid mask、LUT 均在 sequence 生命周期内复用。使用 production `component_bbox`，没有启用 compact strips，也没有使用 keypoint values/confidence。

## Temporal tracks

- Track 0：2213–2221 连续存在；2213 HMR2 cold start，之后 8 帧全部 previous-frame warm start。
- Track 1：只在 2213 出现，随后进入 LOST。
- Track 2：次要人物在 2220–2221 重新出现；由于与 Track 1 间隔超过 reconnect window，按 production timeout policy 创建新 track，2221 使用 warm start。
- 没有观察到错误 ID swap；Track 0 的 bbox、DenseKP geometry 和拟合点始终跟随同一主人物。

统计：3 个 unique tracks，1 个 lost event，0 个 reconnect，0 个真实 LM reinit。

## Per-frame fitting

| Frame | person | track | init | HMR2 | prepare ms | LM ms | initial obj | final obj | trials | success |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2213 | 0 | 0 | hmr2/cam4 | YES | 0.457 | 42.738 | 3358.756 | 256.952 | 39 | YES |
| 2213 | 1 | 1 | hmr2/cam2 | YES | 1.265 | 68.855 | 3160.852 | 124.807 | 37 | YES |
| 2214 | 0 | 0 | previous/2213 | NO | 0.447 | 43.284 | 260.501 | 256.348 | 38 | YES |
| 2215 | 0 | 0 | previous/2214 | NO | 0.494 | 42.983 | 256.967 | 256.397 | 42 | YES |
| 2216 | 0 | 0 | previous/2215 | NO | 0.469 | 44.182 | 252.463 | 252.126 | 43 | YES |
| 2217 | 0 | 0 | previous/2216 | NO | 1.125 | 67.181 | 253.959 | 253.810 | 40 | YES |
| 2218 | 0 | 0 | previous/2217 | NO | 0.485 | 44.050 | 259.260 | 259.074 | 42 | YES |
| 2219 | 0 | 0 | previous/2218 | NO | 0.493 | 39.803 | 258.644 | 257.840 | 40 | YES |
| 2220 | 0 | 0 | previous/2219 | NO | 0.491 | 42.024 | 269.720 | 269.623 | 42 | YES |
| 2220 | 1 | 2 | hmr2/cam2 | YES | 0.846 | 49.346 | 3146.878 | 117.874 | 38 | YES |
| 2221 | 0 | 0 | previous/2220 | NO | 0.483 | 44.550 | 274.109 | 273.924 | 45 | YES |
| 2221 | 1 | 2 | previous/2220 | NO | 1.274 | 68.619 | 91.818 | 82.936 | 34 | YES |

这里的 `trials` 是 Stage1、Pose、PoseShape 三阶段 trial 总数，不是 accepted-step 数。

## Aggregate benchmark

| Metric | Value |
| --- | ---: |
| Frames | 9 |
| Matched observations | 12 |
| Unique tracks | 3 |
| HMR2 calls/persons | 3 / 3 |
| Previous-frame warm starts | 9 |
| HMR2 call ratio | 3/12 = 25% |
| LM attempts/successes | 12 / 12 |
| LM success rate | 100% |
| Runtime preparation p50 / p90 | 0.492 / 1.251 ms |
| CPU LM p50 / p90 | 44.116 / 68.476 ms |
| Fitting p50 / p90 per frame | 57.543 / 487.040 ms |

Fitting p90 包含新 track 的 HMR2 cold start；正常单 track warm frame 大约为 `56–84 ms/frame`（包括 per-person subprocess adapter）。

## DenseKP and HMR2 timing

- DenseKP startup：`2150.0 ms`，单独记录，不计稳态。
- DenseKP full delivery p50：`810.6 ms/frame`。
- 去除 ERP remap 后 p50：`784.5 ms/frame`。
- 当前真实多人 frame 每帧有四次 detector ROI call，DenseKP 也按实际检测人数执行；因此这里不是“单个 DenseKP model forward 约 20 FPS”的 micro benchmark。
- HMR2 checkpoint startup：`8689.6 ms`，单独记录。
- 3 次真实 HMR2 callback 中，额外 normal-ERP preparation 合计 `140.0 ms`。
- HMR2 infer/preprocess 合计 `1228.9 ms`，call p50 `154.5 ms`；均摊到 9 帧为 `136.5 ms/frame`。

HMR2 本轮实际延迟高于此前 B=1 micro benchmark，主要因为真实 sequence 包含多人/不同相机请求以及完整 HMR crop preprocess。核心 policy 仍成立：HMR2 只在 3 个 new tracks 上运行，而不是 12 个 matched observations 每次运行。

## Warm-state gate

所有 9 次 warm start 都自动比较：

```text
frame t C++ optimized frame75/beta10
==
frame t+1 person runtime fixture frame75/beta10
```

结果全部 bitwise exact。每次 C++ invocation 都带 `--runtime-prepare-stage1`，所以 Stage1 geometry 按当前 warm pose/beta 重新生成，没有复用上一帧 prepared geometry。

Cold-start initial objectives 约为 `3147–3359`；主 track warm-start initial objectives约为 `252–274`，说明上一帧 optimized state 确实成为了更接近当前观测的初值。

## Reconnect / reinit regression

真实片段观察到 LOST，但没有在允许 gap 内的 reconnect，也没有自然 LM failure。

自动测试另外覆盖：

- missing frame 后 `ACTIVE → LOST`；允许 gap 内恢复为同一 track；超过 timeout 创建新 track。
- warm-start LM failure 后，同一 frame 立即执行 best-view HMR2 reinit、fresh runtime preparation 和第二次 LM；JSON 显式记录 `hmr2_reinit`，不会 silent retry。
- 多 track beta/frame state 不共享内存。

## Visualization

每帧 raw/optimized correspondence、track ID、person ID、init source 和 LM status：

```text
data_process/Vis/densekp_phase22_sequence_2213_2221_v2/visualization/
```

首帧、中间帧、末帧分别为 `frame_002213.jpg`、`frame_002217.jpg`、`frame_002221.jpg`。人工检查未发现主 track 跳到其他人物。

## Tests and non-changes

```text
PYTHONPATH=src pytest -q src/lib_pano/fitting_io/tests src/lib_pano/human_fitting/tests
16 passed
```

没有修改 C++ LM objective、Jacobian、normal equations、Schur、loss、solver、tolerance、stage schedule 或 CUDA evaluator math；没有修改 DenseKP spherical matching 或 HMR2 模型。

## Decision flags

```text
CONTINUOUS_DENSEKP_V2_DELIVERY=PASS
SINGLE_TRACK_SEQUENCE=PASS
MULTI_TRACK_SEQUENCE=PASS
TRACK_ID_PERSISTENCE=PASS
MATCHED_ONLY_HMR2_SEQUENCE=PASS
HMR2_COLD_START_ONLY=PASS
PREVIOUS_FRAME_WARM_START_SEQUENCE=PASS
WARM_STATE_EXACT_PROPAGATION=PASS
LOST_RECONNECT_SEQUENCE=PASS        # synthetic reconnect gate; real LOST observed
REINIT_SEQUENCE=PASS                # synthetic forced-failure gate
PER_FRAME_CPP_PREPARATION=PASS
SEQUENCE_CPP_LM=PASS
CPP_LM_MATH_CHANGED=NO
```
