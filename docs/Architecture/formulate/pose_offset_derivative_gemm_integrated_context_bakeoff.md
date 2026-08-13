# Pose-offset derivative GEMM integrated-context bake-off

本实验固定 frame 513 的真实 16-call、Q=66 corpus，只比较 pose-offset derivative 的 `P [3N,486] × D [486,66]`。Production Python、`optimized_native` 默认路径和数值容差均未修改。

当前 `PoseCameraInput source → temporary` 的值复制会复制 `selected_posedirs`。其内容按 camera 静态，但临时存储身份并不构成可依赖的稳定生命周期。A 保留逐调用值复制和原 GEMM；B 使用每 camera 一份 immutable P 并保留原 GEMM；C 在 B 基础上以零拷贝视图计算 `Yᵀ=DᵀPᵀ`。D 未执行，因为 B/C 已能回答存储身份假设，且无需引入 prepack。

结果与最终选择见 `benchmark_summary.json`。B、C 的 16 个输出均与 A exact；C 虽更快，但 B→C 未达到预定的 1.15× 采用门槛，因此选择更简单的 `STABLE_P`，下一步只做 stable-posedirs integration bake-off，不在本轮正式集成。
