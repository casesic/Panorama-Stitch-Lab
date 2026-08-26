# Staged-LM C++ 阅读指南

本文以当前源码为准，说明 C++ staged-LM 本身。DenseKP 端到端入口和 Python/C++ 边界见
[`../run/densekp_cuda_staged_lm_reader_guide.md`](../run/densekp_cuda_staged_lm_reader_guide.md)。

## 主调用链

```text
multiframe_staged_lm_reference_main.cpp
  -> run_multiframe_staged_pipeline          multiframe.cpp
       -> Stage1                             stage1_lm.cpp / stage1.cpp
       -> Pose LM                            pose_lm.cpp
            -> EvaluatorBackend              evaluator.cpp
            -> CUDA evaluator                cuda_evaluators.cpp
       -> shared-beta PoseShape LM            pose_shape_lm.cpp
            -> CUDA evaluator
            -> structured normal equations
            -> Schur solve                    pose_shape_schur.cpp
       -> optimized DenseKP output
```

`lm.cpp` 是通用 LM controller：初始化 lambda、构造 candidate、计算 rho、接受或拒绝
trial，并处理停止条件。人体 forward、Jacobian 和 normal equations 由各 stage/evaluator
提供，controller 不知道 SMPL-X 细节。

## 当前源码分类

- Production：`pipeline.cpp`、`multiframe.cpp`、`lm.cpp`，三个 stage，以及
  `evaluator.cpp`、`cuda_evaluators.cpp`、`cuda_forward.cu`、`densekp.cpp`、`erp.cpp`。
- Reference/parity：`apps/*_reference_main.cpp`，负责读取 fixture 和落盘可比较结果。
- Tests：`tests/*_self_test.cpp`，验证 value、Jacobian、normal equations、Schur 和 LM。

当前源码没有 `experimental/` 实现目录；阅读时直接以以上三类文件为准。

## 推荐顺序

1. `apps/multiframe_staged_lm_reference_main.cpp`
2. `src/multiframe.cpp`、`src/pipeline.cpp`
3. `src/lm.cpp`
4. `src/stage1_lm.cpp`、`src/stage1.cpp`
5. `src/pose_lm.cpp`、`src/pose_forward.cpp`
6. `src/pose_jacobian.cpp`、`src/pose_normal_equations.cpp`
7. `src/pose_shape_lm.cpp`、`src/shape.cpp`、`src/pose_shape_forward.cpp`
8. `src/pose_shape_jacobian.cpp`、`src/pose_shape_normal_equations.cpp`
9. `src/pose_shape_schur.cpp`
10. `src/evaluator.cpp`、`src/cuda_evaluators.cpp`
11. 最后阅读 `src/cuda_forward.cu`

公式和参数布局见
[`staged_lm_stage_formulas_and_jacobians_typora.md`](staged_lm_stage_formulas_and_jacobians_typora.md)。

第一遍可以跳过所有 self-test、单阶段 reference app、profiling 字段和 CUDA kernel 细节。
