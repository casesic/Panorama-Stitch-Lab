# Stable selected_posedirs integration bake-off

O4 在 `PoseCameraInput` 中用 `shared_ptr<const vector<float>>` 持有每 camera 一份 canonical immutable P。context 建立时通过 move 提升原 owning vector，不新增数据复制；Pose、PoseShape 及 pipeline 内部临时对象只复制 shared owner。owner 随 `StagedPipelineProblem` 及其阶段 problem 存活，因此任何下游 const view 都覆盖完整 fitting 生命周期，不依赖全局缓存、allocator 地址复用或跨 frame 状态。

O3 `optimized_native` 保留原 owning-copy 行为；O4 通过 `optimized_native_stable_p` 显式启用。两者使用同一 binary、编译策略和 `Y.noalias() = P * D`，未集成转置、prepack 或其他优化。完整结果见 `benchmark_summary.json`。
