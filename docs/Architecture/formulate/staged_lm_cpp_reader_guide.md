# Staged LM C++ reader guide

```text
PREFERRED_POLICY = optimized_native_stable_p
```

本文只描述当前 O4 production path。`optimized_native_stable_p_forward_reuse`（O6）、D0–D5、transpose、blocking、attribution 和 benchmark 都是显式 experimental/diagnostic 路径，第一遍阅读可以跳过。

## 主调用链

```text
staged_lm_pipeline_reference_main
  `-- run_staged_pipeline                         pipeline.cpp
        |-- run_stage1_lm                         stage1_lm.cpp
        |     |-- evaluate_stage1_camera          stage1.cpp
        |     |-- build_stage1_normal_equations
        |     `-- run_lm                          lm.cpp
        |
        |-- Stage1(12) -> Pose(75) handoff
        |-- run_pose_lm                           pose_lm.cpp
        |     |-- Objective
        |     |     `-- evaluate_pose_camera_value (cam2/cam4)
        |     |             pose_forward.cpp
        |     |-- Normal equations
        |     |     |-- evaluate_pose_camera_with_jacobian (cam2/cam4)
        |     |     |             pose_jacobian.cpp
        |     |     `-- build_pose_normal_equations
        |     |                   pose_normal_equations.cpp
        |     `-- run_lm                          lm.cpp
        |
        |-- Pose(75) -> PoseShape(85) handoff
        `-- run_pose_shape_lm                     pose_shape_lm.cpp
              |-- evaluate_shape_value            shape.cpp
              |-- Objective
              |     `-- evaluate_pose_shape_value pose_shape_forward.cpp
              |-- Normal equations
              |     |-- evaluate_pose_shape_camera_with_jacobian
              |     |             pose_shape_jacobian.cpp
              |     `-- build_pose_shape_normal_equations
              |                   pose_shape_normal_equations.cpp
              |-- solve_pose_shape_schur          pose_shape_schur.cpp
              `-- run_lm                          lm.cpp
```

`run_lm` 的 production 调用顺序是：在当前状态构造 normal equations，计算真实 objective 和初始 lambda；每轮求解 damped system，构造 candidate，计算真实 objective、predicted/actual reduction 和 `rho`，最后 accept 或 reject。Stage wrapper 决定 value/Jacobian/normal-equation 数学，generic LM 不知道人体模型细节。

## O4 真正启用的路径

O4 使用 native/vectorized Release build、stable `selected_posedirs` ownership、value pose-offset GEMV、pose-offset derivative GEMM 和 selected-transform GEMM。它不启用 transpose、blocking 或 forward reuse：`ForwardReuseMode::kOff`，因此 `pose_lm.cpp` 与 `pose_shape_lm.cpp` 中 cache、shadow warmup、retain/drop 分支均不执行。

这些优化 kernel 目前位于 `experimental/`，但已经是 O4 验证过的 dispatch。目录名反映其来源，不表示 O4 会走 scalar reference path；不要因目录名跳过它们，只需在理解 value/Jacobian 主链后再读。

## 三类代码

### A. Production / preferred O4

- `src/pipeline.cpp`：三个 stage 和参数 handoff。
- `src/lm.cpp`：通用 LM 状态机。
- `src/stage1.cpp`、`src/stage1_lm.cpp`：刚体相机阶段。
- `src/pose_lm.cpp`、`src/pose_forward.cpp`、`src/pose_jacobian.cpp`、`src/pose_normal_equations.cpp`：Pose 阶段。
- `src/shape.cpp`、`src/pose_shape_lm.cpp`、`src/pose_shape_forward.cpp`、`src/pose_shape_jacobian.cpp`、`src/pose_shape_normal_equations.cpp`、`src/pose_shape_schur.cpp`：PoseShape 阶段。
- `src/so3.cpp`、`src/erp.cpp`：共享数学工具。
- O4 dispatch 还会进入 `experimental/value_pose_offset_gemv.cpp`、`experimental/pose_offset_gemm.cpp` 和 `experimental/selected_transform_gemm.cpp`。

### B. Reference / correctness

- `apps/*_reference_main.cpp`：导入固定 fixture、运行 reference、保存可比较结果。
- 不带 optimized policy 的 scalar/reference dispatch。
- `tests/*cpp_self_test.cpp` 与 Python checker：验证 value、Jacobian、H/g、Schur、trajectory 和跨 backend policy。

### C. Experimental / diagnostic

- `forward_reuse_cache.h`、O6 与 D0–D5：forward reuse 和 integrated-context 诊断。
- transpose/blocking dispatch 与相应 `apps/*benchmark.cpp`。
- `performance.cpp/.h` 及 attribution 开关、raw timing 输出。
- bake-off 文档和大部分专项 self-test。

这些路径均保留，但不是理解 O4 数学所必需。

## 推荐阅读顺序

1. `src/pipeline.cpp`：先掌握 12 → 75 → 85 维 handoff。
2. `src/lm.cpp`：理解 normal equations、lambda、candidate、rho 和 accept/reject。
3. `src/stage1_lm.cpp` → `src/stage1.cpp`：用最短 stage 熟悉 callback 模式。
4. `src/pose_lm.cpp`：只沿 O4 的 cache-miss/value-forward 分支阅读两个 callback。
5. `src/pose_forward.cpp`：rotation → pose offsets → FK → LBS → ERP。
6. `src/pose_jacobian.cpp`：沿同一计算链看导数。
7. `src/pose_normal_equations.cpp`：IRLS、weighted residual/J、75-D H/g。
8. `src/pose_shape_lm.cpp` → `src/shape.cpp` → `src/pose_shape_forward.cpp`。
9. `src/pose_shape_jacobian.cpp` → `src/pose_shape_normal_equations.cpp` → `src/pose_shape_schur.cpp`。
10. 最后阅读 `src/so3.cpp`、`src/erp.cpp` 和 O4 使用的三个 optimized kernel。

第一遍不要进入 forward reuse cache、D0–D5、benchmark apps、attribution instrumentation、transpose/blocking bake-off 或大部分 self-tests。
