# Per-person C++ runtime input preparation

## 结论

Phase 2.1 已将 Stage1 的初始化相关几何从“Python 预先固化在 fixture 中”改为可选的 C++ 运行时重建。新 person fixture 带有
`runtime_prepare_stage1_required=true`；若运行时没有传入 `--runtime-prepare-stage1`，C++ 会直接报错，避免把新 pose/beta 与旧 local geometry 混用。

现有 LM、Jacobian、IRLS、normal equations、Schur、solver、参数布局和 tolerance 均未修改。

## Fixture dependency audit

| Field | Static | Pose-dependent | Beta-dependent | Observation | Runtime rebuild |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_selected_base_v_shaped` | 是（beta=0 基底） | 否 | 通过 basis | 否 | 输入共享 |
| `model_selected_beta_shapedirs` | 是 | 否 | 是 | 否 | 输入共享 |
| `model_selected_posedirs` | 是 | 是 | 否 | 否 | 输入共享 |
| `model_selected_lbs_weights` | 是 | 是（FK/LBS） | 间接 | 否 | 输入共享 |
| `model_base_joints` / `model_d_joints_d_beta` | 是 | 否 | 是 | 否 | 输入共享 |
| `model_parents` | 是 | 否 | 否 | 否 | 输入共享 |
| DenseKP support IDs / sparse regressor | 是 | 否 | 否 | 否 | 输入共享 |
| body pose anchor / beta anchor | 否 | 是 | 是 | 否 | person 输入 |
| hand pose / scale | 否 | 是 | 否 | 否 | person 输入（优化中冻结） |
| cam2/cam4 orient + translation | 否 | 否（不进入 local geometry） | 否 | 否 | person 输入 |
| Stage1 selected/local vertices | 否 | 是 | 是 | 否 | **C++ 重建** |
| Stage1 rotation center | 否 | 间接（root joint） | 是 | 否 | **C++ 重建** |
| targets / DenseKP IDs / valid rows | 否 | 否 | 否 | 是 | person×frame 输入 |

生产者/消费者关系：Python runtime exporter 只写 normalized initial state 与 observation；C++ app 读取共享模型常量，`prepare_stage1_geometry()` 生成 Stage1 几何；原 Stage1/Pose/PoseShape LM 原样消费 prepared problem。

## 几何语义

旧 Python 链：

```text
body pose + beta + frozen hand pose + scale
→ Python SMPL-X forward (global orient=0, translation=0)
→ support/selected vertices + body root joint 0
→ local vertices = selected vertices - root center
→ DenseKP regression（DenseKP observation path）
→ fixture
```

新 C++ 链：

```text
physical beta → dynamic joints / dynamic v_shaped
body pose + hands → existing FK + selected LBS forward
scale → selected vertices and root center
local = selected - root center
DenseKP sparse operator → active Stage1 rows
→ existing Stage1 LM
```

Stage1 local geometry依赖 body pose、physical beta、手姿态和 scale；不依赖相机 global orient 或 translation。相机 orient/translation 只在 Stage1 residual 中把 local geometry 变换到投影空间。SMPL-X body pose 顺序沿用现有 `[21,3]` parent-relative、right-handed axis-angle 布局。

## API 和 ownership

- `PersonFrameInitialState`：每个 person/frame 的 75 维 frame state、10 维 physical beta、双相机冻结手姿和 scale。
- `materialize_person_runtime_fixture()`：静态模型文件用链接共享，只覆盖 person state 与 observation，并写入强制 preparation 标记。
- `prepare_stage1_geometry()`：C++ runtime preparation；输入现有 static model rows 与 normalized state，输出双相机 support vertices、local vertices 和 rotation center。
- `CppPerPersonFitter`：Phase 2 tracker 的 production callback；cold start 接 `PersonFrameInitialState`，warm start 接上一帧 `OptimizedPersonState`，每次强制重新 preparation 并把 C++ 结果写回 track。
- legacy fixture path 保留；只有显式 `--runtime-prepare-stage1` 才重建旧 fixture，便于 oracle 回归。

## Numerical oracle

### frame 513 old fixture self-regeneration

| Quantity | rel_frob | max_abs | PASS |
| --- | ---: | ---: | ---: |
| selected vertices（Python/C++） | 1.23e-7 | 3.58e-7 | YES |
| rotation center（stored/C++） | 8.59e-8 | 2.98e-8 | YES |
| Stage1 local vertices（stored/C++） | 1.58e-7 | 3.58e-7 | YES |

### frame 2214 HMR2 initialization

| Quantity | rel_frob | max_abs | PASS |
| --- | ---: | ---: | ---: |
| selected vertices | 1.44e-7 | 4.77e-7 | YES |
| support local vertices | 1.53e-7 | 4.77e-7 | YES |
| rotation center | 1.20e-8 | 4.19e-9 | YES |

门禁为 `rel_frob <= 2e-6 OR max_abs <= 2e-6`，当前结果有明显余量。person runtime fixture 内旧 Stage1 数组只是兼容占位，不是 oracle，也不会被 LM 消费。

## Legacy LM regression

frame 513 使用相同初值和 observation：

| Metric | Legacy | Runtime prepared |
| --- | ---: | ---: |
| Stage1 initial objective | 1195.839844 | 1195.839844 |
| Stage1 final objective | 47.415382 | 47.410751 |
| Stage1 trials | 24 | 24 |
| Pose final objective | 11.052181 | 11.052062 |
| Pose/Shape final objective | 10.838111 | 10.837968 |
| termination | MAX_ACCEPTED_STEPS | MAX_ACCEPTED_STEPS |

跨 Python/C++ FP32 FK/LBS 的约 `1e-7` 几何差异会令非线性 trajectory 数值略有变化，但 trial 数与 termination 一致；没有修改 tolerance。

## frame 2214 integration

```text
cam2 detections=1, cam4 detections=4, matched=1
HMR2=1 (best view cam4), prepared=1, fitted=1
valid DenseKP=113, optimized DenseKP=138, all finite
Stage1 objective: 3370.827637 → 257.720276
Pose objective: 257.720276 → 256.972198
Pose/Shape objective: 256.972168 → 256.941162
status: MAX_ACCEPTED_STEPS
```

CPU correctness run：person input write `5.28 ms`，C++ load+prepare `6.58 ms`，其中 runtime preparation `1.29 ms`，LM `45.84 ms`，subprocess total `58.48 ms`。这些是单次 correctness measurement，不作为稳定性能 benchmark。

重复 10 次的 CPU B=1 steady-state measurement：runtime preparation p50 `0.47 ms`，LM p50 `42.22 ms`，其余 fixture/static load p50 `2.04 ms`。独立 production adapter 的 cold/warm 两次真实调用也均成功，warm input 与上一轮 optimized frame/beta bitwise exact。

同一 fixture 也已用 CUDA evaluator 跑通：Stage1 参数 exact，Pose/PoseShape frame 参数最大绝对差分别为 `3.81e-6` / `7.63e-6`，shared beta 最大绝对差 `5.96e-8`；三个 stage 的 trial 数与 accept/reject decisions 全部一致。该次 CUDA run 含冷启动，不能作为性能数据。

debug artifacts 位于 `data_process/tmp/phase21_frame2214/`，包括 Python oracle、C++ geometry dump、comparison JSON、initial state、LM outputs 和 correspondence visualization。

## Warm start / reinit

实际 frame 2214 的优化输出被重新序列化成下一次 preparation 输入：75 维 frame parameters 与 10 维 physical beta 均 bitwise exact。重新 preparation 后 geometry 确实变化；切换回 HMR2 reinit state 后，prepared geometry 与第一次 HMR2 preparation bitwise exact。这证明 cache 不会跨 initialization state 串用。

当前验证的是两次真实 C++ preparation/LM 的 state propagation gate；真实 5–20 帧匹配序列还需在相邻帧 `densekp_pairs_v2` 全部生成后运行，不应把同一 observation 的 refresh test 冒充完整 tracking benchmark。

## Commands

```bash
/usr/bin/cmake -S cpp/staged_lm -B data_process/tmp/staged_lm_phase21_build \
  -DSTAGED_LM_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build data_process/tmp/staged_lm_phase21_build -j2
/usr/bin/ctest --test-dir data_process/tmp/staged_lm_phase21_build --output-on-failure
```

Python tests：`15 passed`。C++ self tests：`11/11 passed`。

## Decision flags

```text
PER_PERSON_CPP_PREPARATION=PASS
OLD_FIXTURE_REGENERATION=PASS
PYTHON_CPP_STAGE1_GEOMETRY_MATCH=PASS
FRAME2214_CPP_LM=PASS
PREVIOUS_FRAME_WARM_START_CPP=PASS
REINIT_CPP_PREPARATION=PASS
PER_PERSON_CPP_LM=PASS
CPP_LM_MATH_CHANGED=NO
```
