# Staged LM C++ optimized-kernel integration bake-off

本实验第一次在同一个 host-specific C++ executable 中接入两个已经独立通过的 winner：Pose-offset derivative Eigen GEMM，以及 LBS selected-transform derivative Eigen GEMM。Reference mode 继续使用 scalar、禁用 Eigen packet 且不启用 native；Optimized mode 使用统一的 Eigen vectorization + `-march=native` policy。默认 reference 行为没有切换。

frame 513、cam2+cam4 的完整 Stage1→Pose→PoseShape 运行中，三个阶段的 trial 数、accept/reject 序列、solve backend、objective/linearization call counts 和 termination status 均一致。Optimized mode 实际调用 Pose-offset winner 16 次、LBS winner 20 次，scalar 与 fallback 都为零；正式 pipeline parity policy 全部通过，first divergence 为 NONE。

同日 Release 实测 reference compute median 为 4804.010 ms，optimized 为 420.919 ms，即 2.376 FPS 和 11.413× speedup；包含 fixture load 与输出序列化的 optimized executable wall median 为 530 ms，即 1.887 FPS。它比 618.178 ms 投影快 31.91%，主要原因是 whole-target native/vectorized build 同时加速了 Stage1、Objective、NE 和其他未替换代码；因此不能把完整 speedup 全部归因于两个 winner。完整上下文中的 winner 又分别比 isolated microbenchmark 慢，属于 cache、调用上下文和计时边界差异。

Optimized Objective 占 31.21%，真实 Top1 是 `pose_objective/cam2/value.pose_offsets`。完整 LBS 已不再是主要瓶颈，因此下一实验选择 `POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF`，本轮不实施该优化。

完整结果见 `data_process/Vis/cpp_reference_dump/staged_lm_optimized_integration/frame_000513/benchmark_summary.json`。
