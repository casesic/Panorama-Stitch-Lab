# ADR-0002：staged LM 的数值核心用 C++ 重写

- **状态**：已生效（2026-11-20 记录；决定早于此）
- **影响面**：`cpp/staged_lm/`、`tests/human_fitting/staged_lm/`
- **一句话**：SMPL-X 拟合的 SO(3)、ERP 重投影、解析 Jacobian、法方程与 Schur 求解全部自己写在 C++ 里；Python 只做编排、数据准备与可视化。

## 背景

拟合是一个逐帧的 LM 问题：Stage 1 12 维、Pose 75 维、PoseShape 85 维（75 + 10 beta latent）。参考实现先建立了 CPU 正确性基线，代价分布很集中——Jacobian 占了全部运行时间的绝大部分。

## 决定

1. 数学核心用 C++ 实现（`include/` + `src/`），不依赖自动微分框架；每阶段的 value、解析 Jacobian、`H/g` 都可单独调用与自测。
2. 每个阶段都有独立的 self-test 与 reference executable（共 11 个 CTest 用例），保证「数学正确」先于「跑得快」。
3. 在这套 C++ 之上再做 kernel 级优化（O4 policy `optimized_native_stable_p`）、多帧共享 beta、CUDA 与 TensorRT 后端。
4. Python 侧只保留编排、fixture 生成、可视化和测试对照。

## 依据

Release、frame 513 的 CPU 基线（[staged_lm_cpp_pipeline_reference](../Architecture/archive/staged_lm_cpp_pipeline_reference.md)）：

| 项 | 数值 |
|---|---|
| compute-only 三次 | 5476.507 / 5437.663 / 5433.284 ms（median 5437.663） |
| 折算吞吐 | 约 **0.184 FPS** |
| 阶段占比 | Stage1 0.67%、Pose 68.24%、PoseShape 30.97% |
| 成本构成 | **Jacobian 93.29%**、objective 4.81%、NE 1.18%、solve/control 0.60% |
| peak RSS | 约 116.8 MB |

原文据此给出下一步方向：**先分析 pose-offset derivative 与 LBS derivative，而不是 Schur 或 LM control**。这条结论直接解释了为什么后续工作集中在导数 kernel，而不是求解器结构。

优化后两次手测（见 [benchmarks/RESULTS.md](../../benchmarks/RESULTS.md)）：O4 单帧 5 FPS（2026-08-13）、100 点 loss 的 C++ 管线 25–30 FPS（2026-08-18）。

## 代价与已知限制

- **写两遍数学**：Python 参考实现与 C++ 实现并存，改动要两边对齐，靠 self-test 和 reference dump 兜底。
- **可移植性**：默认 `-march=native` + Eigen 向量化，换机器要重新编译；数值结果可能随 ISA 变化。
- **构建成本**：Eigen ≥ 3.3 必需，CUDA 可选但默认开。

## 未在文档中写明（待作者补充）

- 是否评估过 PyTorch / JAX 的自动微分方案，以及放弃的具体原因。
- 为什么选择手写 SO(3) 与解析 Jacobian，而不是用第三方库。
- Python 与 C++ 的分工边界（哪些留给 Python）的判定标准。

## 相关文档

- [staged_lm_cpp_o4_single_frame](../Architecture/run/staged_lm_cpp_o4_single_frame.md)
- [staged_lm_cpp_pipeline_reference](../Architecture/archive/staged_lm_cpp_pipeline_reference.md)
- [src/README.md](../../src/README.md)（使用说明与性能口径）
