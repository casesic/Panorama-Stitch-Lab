# Pano-Stitch：Staged LM 人体拟合使用说明

本文档介绍当前 staged Levenberg–Marquardt（LM）SMPL-X 拟合流程的运行方法、
输入输出和代码结构。命令均假定在仓库根目录执行。

## 1. 功能概览

当前 production 流程面向双相机 ERP 观测，分为三个阶段：

1. **Stage 1 rigid alignment**：逐帧优化 cam2/cam4 的 global orientation 和
   translation。
2. **Stage 2 pose warmup**：逐帧继续优化相机姿态与 body pose，beta 固定。
3. **Stage 2 joint refinement**：联合优化逐帧 pose 和跨帧共享的 bounded beta
   latent。

正式路径固定使用：

- Stage 1 analytic Jacobian 和快速解析 evaluation；
- Stage 2 pose analytic Jacobian 和 selected-vertex fast evaluation；
- Stage 2 joint dynamic pose+beta analytic Jacobian；
- combined pose/beta FK derivative；
- direct arrowhead normal equations；
- shared-beta Schur complement；
- CPU/CUDA 显式 device contract。

Autograd、full-forward、dense joint Hessian、separate FK 和 L-BFGS 等 reference
实现只位于 validation 层，不会进入正式运行路径。

## 2. 运行准备

建议使用项目已有的 CameraHMR 环境：

```bash
conda activate camerahmr
cd /path/to/pano-stitch
```

运行需要以下输入：

- HMR 初始化目录；
- SAM3D/SMPL-X dense observation 目录；
- bbox/ERP crop metadata 目录；
- 本机可用的 SMPL-X 模型文件和项目配置。

脚本会自行把仓库的 `src/` 加入 `sys.path`，无需 editable install，也不要求使用
`python -m`。

## 3. 正式拟合

正式入口为：

```text
src/script/human_fitting/staged_lm/run.py
```

### 3.1 CPU 单帧示例

```bash
python src/script/human_fitting/staged_lm/run.py \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm \
  --device cpu \
  --reproj_objective charbonnier_irls \
  --profile_timing
```

### 3.2 CUDA 单帧示例

```bash
python src/script/human_fitting/staged_lm/run.py \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm_cuda \
  --device cuda:0 \
  --reproj_objective charbonnier_irls \
  --profile_timing
```

显式请求 CUDA 但 CUDA 不可用时会抛出：

```text
CUDA_REQUESTED_BUT_UNAVAILABLE
```

### 3.3 多帧 shared-beta 示例

```bash
python src/script/human_fitting/staged_lm/run.py \
  --frames 513,514,515,516 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm_t4 \
  --device auto \
  --reproj_objective charbonnier_irls
```

`--device auto` 会优先选择 `cuda:0`，没有 CUDA 时使用 CPU。

### 3.4 正式 CLI 参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `--frames` | 帧号，多个帧以逗号分隔 | 必填 |
| `--hmr_dir` | HMR 初始化目录 | 必填 |
| `--sam3d_dense_dir` | dense SMPL-X observation 目录 | 必填 |
| `--bbox_dir` | bbox/ERP metadata 目录 | 必填 |
| `--image_width` | ERP 图像宽度 | `5760` |
| `--image_height` | ERP 图像高度 | `2880` |
| `--output_dir` | 输出根目录 | 必填 |
| `--device` | `auto`、`cpu`、`cuda` 或 `cuda:N` | `auto` |
| `--validate_device_contract` | 构建 context 后执行额外 device 检查 | 关闭 |
| `--reproj_objective` | `charbonnier_irls` 或 `raw_l2` | `charbonnier_irls` |
| `--stop_after_stage` | `stage1`、`stage2_pose` 或 `stage2` | `stage2` |
| `--profile_timing` | 输出细粒度 timing | 关闭 |

查看最新参数：

```bash
python src/script/human_fitting/staged_lm/run.py --help
```

## 4. 输出文件

输出目录会根据帧范围增加 segment 子目录。例如：

```text
data_process/Vis/staged_lm/
└── 513_513/
    ├── staged_stage2b_beta.json
    └── staged_stage2b_beta.npz
```

其中：

- JSON 保存 execution environment、config、各 Stage LM 结果、metrics、solver
  counters 和 pipeline timing；
- NPZ 保存 Stage 参数向量、anchors、physical beta 及用于可视化/复现的数组；
- `--stop_after_stage` 提前停止时仍会写出当前已经完成的 Stage 结果。

## 5. 可视化

入口：

```text
src/script/human_fitting/staged_lm/visualize.py
```

示例：

```bash
python src/script/human_fitting/staged_lm/visualize.py \
  --result_dir data_process/Vis/staged_lm/513_513 \
  --image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --stage pose_shape_refinement \
  --solver lm \
  --output_dir data_process/Vis/staged_lm_render \
  --mesh \
  --save_video
```

可选 Stage 为 `stage1`、`pose_warmup` 和 `pose_shape_refinement`。脚本只读取已有结果，
不会重新执行 LM。

## 6. Correctness 与 solver validation

统一入口：

```text
src/script/validation/staged_lm/run.py
```

### 6.1 查看子命令

```bash
python src/script/validation/staged_lm/run.py --help
```

支持：

```text
stage1
pose-forward
pose-jacobian
pose-solver
joint-beta
joint-jacobian
joint-solver
joint-schur
joint-fk
all
```

### 6.2 Joint Schur 回归示例

```bash
python src/script/validation/staged_lm/run.py joint-schur \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate
```

### 6.3 Combined/separate FK 与 L-BFGS Gate C

```bash
python src/script/validation/staged_lm/run.py joint-fk \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --run_lbfgs_oracle
```

## 7. 性能测试

### 7.1 Joint analytic H/g kernel profile

```bash
python src/script/benchmark/staged_lm/profile_pose_shape_kernels.py \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --device cpu \
  --warmup 10 \
  --repeats 30 \
  --run_candidates
```

### 7.2 端到端 cold-start/steady-state profile

```bash
python src/script/benchmark/staged_lm/profile_end_to_end.py \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm_benchmark \
  --device cuda:0 \
  --benchmark_warmup_runs 2 \
  --benchmark_repeats 10 \
  --benchmark_profile_breakdown \
  --benchmark_output_json data_process/Vis/staged_lm_benchmark/summary.json
```

主 steady-state 统计不会把文件读取、模型加载和输出保存计入 fitting FPS。

## 8. Tests

安装 pytest 时可以运行：

```bash
pytest -q tests/human_fitting/staged_lm
```

各测试也支持直接执行，例如：

```bash
python tests/human_fitting/staged_lm/test_architecture.py
python tests/human_fitting/staged_lm/test_stage1.py
python tests/human_fitting/staged_lm/test_joint_jacobian.py
python tests/human_fitting/staged_lm/test_device.py
```

## 9. Python API 示例

需要在 Python 中调用而不通过 subprocess 时，可以复用正式 parser 和 application：

```python
from script.human_fitting.staged_lm.app import run_production_pipeline
from script.human_fitting.staged_lm.cli import build_argument_parser

parser = build_argument_parser()
args = parser.parse_args([
    "--frames", "513",
    "--hmr_dir", "data_process/Output/20260311_riding/hmr_cache",
    "--sam3d_dense_dir",
    "data_process/Output/20260311_riding/sam3dbody/smplx_dense",
    "--bbox_dir",
    "data_process/Output/20260311_riding/image/equirect_crop_rotate",
    "--output_dir", "data_process/Vis/staged_lm_api",
    "--device", "cpu",
])

result = run_production_pipeline(args, save_outputs=True)
print(result.pose_shape_outcome.lm_result.objective)
print(result.output_directory)
```

如果当前 Python 进程没有配置仓库路径，应先把 `<repository>/src` 加入
`PYTHONPATH`。直接运行上述正式脚本时不需要这一步。

## 10. 代码目录与文件职责

### 10.1 Production library

目录：`src/lib_pano/human_fitting/staged_lm/`

| 文件 | 职责 |
| --- | --- |
| `README.md` | staged LM 的数学流程、production 默认值和架构说明 |
| `config.py` | objective/solver dataclass、数值常量和 production provenance |
| `types.py` | `StageContext`、Stage definition/outcome、LM result、normal equations 类型 |
| `device.py` | CPU/CUDA 解析、tensor 迁移和 device contract 验证 |
| `data_loading.py` | observation/HMR 读取、初始化及 context 数据准备 |
| `parameter_layout.py` | Stage 1、pose、joint 参数布局、打包、解包和切片 |
| `residuals.py` | reprojection group、残差展开及原始像素 metrics |
| `irls.py` | Charbonnier IRLS、聚合规格和 frozen row scales |
| `analytic_stage1.py` | Stage 1 selected forward、解析 Jacobian 和 direct H/g |
| `pose/` | Pose cache、forward、解析导数、objective/metrics 与 FULL_BATCH H/g |
| `joint/` | Joint shape、forward、beta/combined 导数、Arrowhead 与 FULL_BATCH H/g |
| `arrowhead_schur.py` | shared-beta arrowhead 系统、Schur solve 和 dense fallback |
| `lm/` | 通用 LM equations、trial、convergence 和 runner |
| `stage_definitions/` | 三阶段 definition 构建 |
| `stage_runtime.py` / `pipeline.py` | 单阶段执行与 Stage2 orchestration |
| `output.py` | JSON/NPZ 数组构建、序列化和 timing metadata |

Production library 不依赖 `src/script/`、validation、benchmark 或 tests。

### 10.2 Application

目录：`src/script/human_fitting/staged_lm/`

| 文件 | 职责 |
| --- | --- |
| `app.py` | `build_context()`、三阶段 pipeline、结果组装与保存 orchestration |
| `cli.py` | 正式 CLI 参数及默认值 |
| `run.py` | 不超过一层转发逻辑的正式命令入口 |
| `visualize.py` | 读取 JSON/NPZ 并生成 Stage/solver 可视化 |

### 10.3 Validation

目录：`src/script/validation/staged_lm/`

| 文件 | 职责 |
| --- | --- |
| `common.py` | validation 公共参数、误差和 profiling helper |
| `stage1.py` | Stage 1 residual/J/H-g/solver validation |
| `pose.py` | pose selected forward、Jacobian、JVP、finite difference 和 solver 回归 |
| `joint.py` | beta mapping、dynamic joint Jacobian、block/JVP/gradient 检查 |
| `solver.py` | A/B/C、dense/Schur、separate/combined FK 和 Gate C 回归 |
| `run.py` | validation subcommand 统一入口 |
| `references/` | autograd/full-forward/dense/separate-FK/L-BFGS reference 实现 |

依赖方向固定为：

```text
validation checks -> validation references -> production library
```

### 10.4 Benchmark

目录：`src/script/benchmark/staged_lm/`

| 文件 | 职责 |
| --- | --- |
| `pose_shape_kernel_candidates.py` | 隔离的 kernel candidate 和 tensor/memory 审计 helper |
| `profile_pose_shape_kernels.py` | joint group analytic H/g 细粒度 profile |
| `profile_end_to_end.py` | context、cold-start、steady-state、Stage timing 和 CPU/CUDA profile |

候选 kernel 不会被 production 导入。

### 10.5 Tests

目录：`tests/human_fitting/staged_lm/`

- `test_architecture.py`：包结构和依赖方向；
- `test_pipeline.py`：pipeline 纯逻辑与配置；
- `test_visualization.py`：输出读取和可视化纯逻辑；
- `test_stage1.py`：Stage 1 analytic math；
- `test_pose_fast.py`、`test_pose_jacobian.py`：pose forward/Jacobian；
- `test_joint_beta_jacobian.py`、`test_joint_jacobian.py`：joint beta/combined J；
- `test_arrowhead_schur.py`：structured system 和 Schur；
- `test_joint_kernels.py`：benchmark candidate correctness；
- `test_device.py`：CPU/CUDA device contract；
- `helpers.py`：共享 synthetic fixtures，测试文件之间不相互导入。

## 11. 推荐阅读顺序

首次阅读建议按以下顺序：

1. `src/lib_pano/human_fitting/staged_lm/README.md`
2. `src/script/human_fitting/staged_lm/run.py`
3. `src/script/human_fitting/staged_lm/app.py`
4. `src/lib_pano/human_fitting/staged_lm/types.py`
5. `src/lib_pano/human_fitting/staged_lm/config.py`
6. `src/lib_pano/human_fitting/staged_lm/pipeline.py`
7. `analytic_stage1.py`
8. `pose/derivatives.py`
9. `joint/derivatives.py`
10. `joint/arrowhead.py`
11. `lm/runner.py`

更详细的数学、验证状态和当前性能结论见：

```text
src/lib_pano/human_fitting/staged_lm/README.md
```

# 运行示例

python src/script/human_fitting/staged_lm/run.py \
  --frames 513 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm_cpu \
  --device cpu \
  --reproj_objective charbonnier_irls \
  --profile_timing

python src/script/human_fitting/staged_lm/run.py \
  --frames 513,514,515,516 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm_cpu_t4 \
  --device cpu \
  --reproj_objective charbonnier_irls \
  --profile_timing
