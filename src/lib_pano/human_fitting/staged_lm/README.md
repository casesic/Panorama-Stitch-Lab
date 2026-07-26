# 分阶段 LM 人体拟合

## 包的职责

`lib_pano.human_fitting.staged_lm` 是双相机 ERP SMPL-X 分阶段
Levenberg–Marquardt（LM）拟合器的正式生产实现。

Production 只保留一条受支持的执行路径。应用、正确性验证和性能测试分别位于
`src/script/human_fitting/staged_lm`、`src/script/validation/staged_lm` 和
`src/script/benchmark/staged_lm`，不会被 production 反向导入。

## 正式执行路径

- **Stage 1：** 使用解析刚体旋转/平移 Jacobian、直接法向方程，以及解析快速前向
  计算 objective 和 metrics。
- **Stage 2 pose：** beta 固定，使用解析 pose Jacobian 和 selected-vertex 快速
  objective/metrics。
- **Stage 2 joint：** serial residual-only 路径构建 IRLS weights；FULL_BATCH FP32
  rotation/shape/FK/LBS/projection 同时生成 pose/shared-beta Jacobian；FP64 batched
  GEMM 构建 Arrowhead A/B/C/g 并按 production group 顺序归约；objective/metrics
  仍为 serial fast，求解仍为原 Schur complement。

执行设备默认值为 `auto`。优化参数和模型前向使用 float32；法向方程累计与线性求解
使用 float64，并始终与优化参数位于同一个 CPU 或 CUDA 设备上。

## 主要模块

- `config.py`：正式 solver/objective dataclass、数值常量、配置校验和固定的生产路径
  provenance 标签；不依赖 argparse。
- `types.py`：context、Stage 定义、静态输入摘要、准备阶段计时、结果记录，以及
  dense/structured normal equations 数据结构。
- `device.py`：显式 CPU/CUDA 解析、tensor 迁移和 device contract 检查。
- `parameter_layout.py`：各 Stage 参数向量的稳定布局、打包和切片。
- `residuals.py`：重投影残差展开和原始像素 metrics。
- `irls.py`：由 observation 元数据构建静态聚合规格、Charbonnier IRLS 和冻结权重的
  surrogate 构建。
- `linearization.py`：Stage 与 solver 之间唯一正式、不可变且可类型检查的
  `Linearization` 合同。
- `geometry/so3.py`：Stage1/Pose/Joint 共用的 Rodrigues、SO(3) right Jacobian
  和旋转点 Jacobian。
- `geometry/erp.py`：统一 cam2 三维坐标约定下的 ERP 投影和解析投影 Jacobian。
- `lm/equations.py`：Jacobian、H/g，以及 dense/structured 求解分派。
- `lm/trial.py`：predicted/actual reduction、rho 和 trial record。
- `lm/convergence.py`：damping 更新、收敛判断和 legacy no-accepted 状态。
- `lm/runner.py`：使用显式局部变量实现通用 LM 主状态机。
- `analytic_stage1.py`：Stage 1 快速前向、解析 Jacobian 和 direct H/g。
- `pose/cache.py`：Pose 模型、观测 selected rows、serial/batch cache 与布局。
- `pose/forward.py`：beta 固定时 rotations → pose blend → FK → selected LBS →
  ERP residual 的无导数阅读链。
- `pose/derivatives.py`：Pose serial/reference 解析链及 dense H/g。
- `pose/objective.py`：Pose true objective、raw metrics 和对应 timing/counters。
- `pose/batched.py`：Pose FULL_BATCH 几何、解析 Jacobian，以及 FP64 batched H/g
  和 production-order frame reduction。
- `stage2_shared_cache.py`：Pose/Joint 共用的不可变 observation、posedirs rows、
  LBS weights rows 和模型布局。
- `joint/cache.py`：Joint 模型、selected rows、serial cache 与 timing counters。
- `joint/shape.py`：beta latent → physical beta → dynamic joints。
- `joint/forward.py`：dynamic shape → rotations/FK/LBS → ERP residual value链。
- `joint/beta_derivatives.py`：独立 beta-only 解析 reference链。
- `joint/derivatives.py`：combined frame/beta 解析 Jacobian链。
- `joint/objective.py`：Joint objective、anchors、terms 与 metrics。
- `joint/arrowhead.py`：serial Arrowhead A/B/C/g 与 Linearization绑定。
- `joint/batched.py`：production FULL_BATCH geometry/Jacobian 与 FP64 Arrowhead H/g。
- `arrowhead_schur.py`：shared-beta arrowhead 系统和带 LM damping 的 Schur 求解。
- `stage_runtime.py`：通用单 Stage 执行与 LM 调用。
- `stage_definitions/rigid.py`：Stage 1 刚体 definition 构建。
- `stage_definitions/pose.py`：Pose warmup definition 构建。
- `stage_definitions/joint.py`：Joint pose/beta definition 构建。
- `stage_handoff.py`：参数状态传递与 state-only handoff 检查。
- `stage_timing.py`：preparation/cache timing 写入。
- `pipeline.py`：Stage 2 Pose/Joint 编排。
- `data_loading.py`：输入数据读取和 StageContext 构建。
- `output.py`：正式结果数组、objective breakdown，以及 JSON/NPZ 序列化。

## Code map 与 reading guide

推荐按以下顺序阅读：

1. `pipeline.py`：Stage 2 编排；
2. `stage_runtime.py`：通用单 Stage 执行；
3. `stage_definitions/rigid.py`；
4. `stage_definitions/pose.py`；
5. `pose/cache.py` → `pose/forward.py` → `pose/derivatives.py`；
6. `pose/objective.py` → `pose/batched.py`；
5. `stage_definitions/joint.py`；
6. `stage_handoff.py`；
7. `stage_timing.py`；
8. `linearization.py`；
9. `lm/equations.py`；
10. `lm/trial.py`；
11. `lm/convergence.py`；
12. `lm/runner.py`；
13. `stage_runtime.py`；
9. `linearization.py`；
10. `geometry/so3.py`、`geometry/erp.py`；
11. Stage kernels；
12. `arrowhead_schur.py`。

## Stage construction map

代码按职责从 `stage_runtime`、`stage_definitions.*`、`stage_handoff`
和 `pipeline` 导入。仓库不再支持重构前的 `stage2_pose_*`、
`stage2_joint_*`、`lm_solver` 或 `stages` import 路径。

数学公式与具体函数的一一对应见
`docs/Architecture/staged_lm/math_to_code.md`，总体数据流见
`docs/Architecture/staged_lm/overview.md`。

## Mathematical kernels

公共 SO(3) 和 ERP 数学只在 `geometry/` 中实现。`analytic_stage1.py` 仅保留旧名称
的兼容 re-export，production Stage2 直接使用 `skew`、
`project_unified_erp` 和 `project_unified_erp_with_jacobian` 等 canonical API。

## Linearization contract

`Linearization` 是 solver 与 Stage 之间唯一正式合同，包含：

- `surrogate`；
- 可选 `jacobian`；
- 可选 `normal_equations`；
- `weight_stats` 和 `aggregation_spec`；
- 不改变输出值的 backend provenance。

求解器优先使用 direct normal equations；没有 direct H/g 时使用 analytic
Jacobian；两者都没有时通过 `torch.func.jacfwd` 对 surrogate 求 Jacobian。
`Linearization` 不实现 `dict`、`get` 或字符串索引接口。

Within one LM solve, the current true-objective Python scalar is carried
across accepted states. Candidate objectives are still evaluated exactly
once, while accepted rechecks, the next outer iteration and the final result
reuse that exact value. This is local state transition bookkeeping, not
general parameter memoization.

正式 application 和 argparse 层位于：

```text
src/script/human_fitting/staged_lm/
├── app.py
├── cli.py
├── run.py
└── visualize.py
```

CLI 解析位于 library 之外，可避免实验性开关重新进入 production 配置。

## Validation reference

以下 reference 实现位于：

```text
src/script/validation/staged_lm/references/
```

它们可以导入 production kernel，但 production 不会导入它们：

- autograd Jacobian 和 full-forward oracle；
- joint dense global normal equations；
- pose/beta 分开递推的 FK reference；
- L-BFGS oracle 和 Gate C；
- debug117、debug118、debug120 使用的 reference Stage builders。

这些实现继续用于导数验证、H/g 对比、solver 回归和 LM/L-BFGS 对齐检查，但不能通过
正式拟合命令选择。

## 已从 production 移除的开关

正式 CLI 不再暴露以下实现选择：

- Jacobian backend；
- Stage evaluation backend；
- pose/joint linearization backend；
- joint dense/Schur solver 选择；
- separate/combined FK 选择；
- L-BFGS oracle 开关；
- fast/full 和 IRLS first-order validator 开关。

真实算法参数仍然可以配置，包括：

- `--device`；
- 输入和输出路径；
- `--reproj_objective`（`charbonnier_irls` 或 `raw_l2`）；
- Stage 停止位置；
- profiling 和日志选项。

## 正确性状态

当前正式路径已经通过：

- analytic/full-fast 对齐验证；
- Stage 1、pose、joint Jacobian 测试；
- direct H/g 验证；
- dense/Schur solver 回归；
- separate/combined FK 回归；
- LM/L-BFGS 对齐；
- CPU/CUDA 最终结果回归。

Production 的 pose→joint handoff 只检查参数状态、bounded beta 映射和 shape anchor；
需要 full SMPL-X 重投影和 objective 对齐时，使用 validation 的 `handoff` 子命令：

```bash
python src/script/validation/staged_lm/run.py handoff \
  --frames 513 \
  --hmr_dir HMR_DIR \
  --sam3d_dense_dir SAM3D_DIR \
  --bbox_dir BBOX_DIR
```

## 当前性能状态

在 RTX 3090 主机和当前 solver iteration budget 下，历史 steady-state 结果约为：

- CPU，T=1：p50 约 1278 ms；
- CUDA，T=1：p50 约 1587 ms；
- CUDA，T=8：每个 segment 的 p50 约 15844 ms。

Stage 2 Pose 的正式 normal-equation 路径现已使用多 group FULL_BATCH；此前还消除了准备、日志、JSON 和 objective breakdown 中
的重复 full SMPL-X forward。在当前 CPU T=1 数据上，fitting steady p50 从约
921 ms 降至约 696 ms；objective、参数、metrics 和 NPZ 数组保持逐元素一致。

Stage 2 当前进一步采用三层 cache：

- Shared immutable geometry：context observation、hands/scale、parents，以及每个
  group 只选择一次的 posedirs/LBS weights；
- Pose derived：固定 beta-anchor 的 `v_shaped`、joints 和 selected vertices；
- Joint derived：动态 beta 所需的 base shape、beta shapedirs、joint derivatives 和
  selected beta rows。

Pose/Joint 对公共 tensor 持有相同 storage 引用，不在 `index_select()` 后再次 clone。
CPU T=1 实测 Stage2 cache p50 从约 127.51 ms 降至约 63.88 ms，fitting p50 从约
695.78 ms 降至约 640.06 ms。Shared cache 仍需执行一次约 62.25 ms 的 posedirs row
selection，因此 cache 并未被完全省略。

Pose 的数值策略固定为：serial residual-only IRLS weights、FULL_BATCH
rotation/FK/LBS/projection/Jacobian、FP64 batched H/g、production group-order frame
reduction、serial fast objective/metrics，以及原 dense LM solve。FP32 batch 与 serial
的首个差异来自 rotation value；FP64 单算子诊断将误差降至约 `1e-16`，完整 LM
trajectory 和最终结果通过 production gate。旧 microkernel gate 继续作为诊断指标。

Pipeline timing 当前使用 schema v5，明确区分：

- `*_prepare_total_ms`；
- `*_handoff_ms`；
- `*_cache_setup_ms`；
- `*_definition_bind_ms`；
- `*_run_ms` 和 `*_run_overhead_ms`；
- `stage2_shared_*`、Pose/Joint derived cache 和 `stage2_total_cache_setup_ms`；
- `stage2_pose_fast_cache_setup_ms`、`stage2_pose_batch_cache_setup_ms` 和 batch
  padding/persistent-memory metadata；
- Joint fast/batch cache setup、batch padding/persistent metadata，以及 batched NE、
  shape-state、full-analytic 调用计数；
- `objective_breakdown_ms`。

必要的 Stage cache 预计算仍计入 fitting total，不会被隐藏到 solver 内。

## 五层代码布局

- Production：`src/lib_pano/human_fitting/staged_lm/`
- 正式应用与可视化：`src/script/human_fitting/staged_lm/`
- 正确性与回归验证：`src/script/validation/staged_lm/`
- 性能与候选 kernel：`src/script/benchmark/staged_lm/`
- 单元测试：`tests/human_fitting/staged_lm/`

Validation 的统一入口按 Stage 提供 `stage1`、`pose-forward`、
`pose-jacobian`、`pose-solver`、`joint-beta`、`joint-jacobian`、
`joint-solver`、`joint-schur`、`joint-fk`、`handoff` 和 `all` 子命令。候选 kernel 只允许
benchmark 和 tests 使用，不进入 production。

## 使用方式

### 正式运行

```bash
python src/script/human_fitting/staged_lm/run.py \
  --frames 513 \
  --hmr_dir HMR_DIR \
  --sam3d_dense_dir SAM3D_DIR \
  --bbox_dir BBOX_DIR \
  --output_dir OUTPUT_DIR \
  --device cuda:0 \
  --reproj_objective charbonnier_irls \
  --profile_timing
```

### 可视化

```bash
python src/script/human_fitting/staged_lm/visualize.py --help
```

### 正确性验证

```bash
python src/script/validation/staged_lm/run.py joint-schur \
  --frames 513 \
  --hmr_dir HMR_DIR \
  --sam3d_dense_dir SAM3D_DIR \
  --bbox_dir BBOX_DIR
```

### 端到端 benchmark

```bash
python src/script/benchmark/staged_lm/profile_end_to_end.py \
  --frames 513 \
  --hmr_dir HMR_DIR \
  --sam3d_dense_dir SAM3D_DIR \
  --bbox_dir BBOX_DIR \
  --output_dir OUTPUT_DIR \
  --device cuda:0 \
  --benchmark_warmup_runs 1 \
  --benchmark_repeats 5 \
  --audit_full_forward_calls
```

### Tests

```bash
pytest -q tests/human_fitting/staged_lm
```

## 历史路径映射

| 旧职责/编号 | 新位置 |
| --- | --- |
| debug109 正式拟合 | `script/human_fitting/staged_lm/run.py`、`app.py` |
| debug110 可视化 | `script/human_fitting/staged_lm/visualize.py` |
| debug111 Stage 1 验证 | `script/validation/staged_lm/stage1.py` |
| debug112–114 pose 验证 | `script/validation/staged_lm/pose.py` |
| debug115–116 joint Jacobian 验证 | `script/validation/staged_lm/joint.py` |
| debug117、118、120 solver 回归 | `script/validation/staged_lm/solver.py` |
| debug119 kernel profile | `script/benchmark/staged_lm/profile_joint_kernels.py` |
| debug121 端到端 profile | `script/benchmark/staged_lm/profile_end_to_end.py` |
| staged_lm_validation | `script/validation/staged_lm/references/` |
| staged_lm_bench | `script/benchmark/staged_lm/joint_kernel_candidates.py` |
| test_debug109/test_debug110 | `tests/human_fitting/staged_lm/` |

## 后续整理项

后续可以继续进行以下架构工作：

- 将 input/output 职责进一步迁移到 `fitting_io`；
- 整理和公开当前 private API；
- 完成 Joint FULL_BATCH shadow regression，并据结果决定是否接入 production。

上述事项不属于当前单路径代码整理范围。
