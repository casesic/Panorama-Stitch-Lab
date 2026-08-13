# Pose-offset derivative GEMM blocking bake-off

在 frame 513 的真实 16-call trajectory-order workload 上，比较当前 Stable-P `Y=P*D` 与显式 M/N blocking。候选仅使用 Eigen 零拷贝 block view，不预打包、不复制完整 P、不改变 row-major `[M,66]` 输出。

四个 M 候选与四个 N 候选均未达到 1.10×；按 negative-stop 规则不执行二维 blocking。最终 BM=512 是最佳 blocking，但相对基线仍属 NEUTRAL，因而不值得集成。其局部输出与基线 bitwise equal，完整 Stage1/Pose/PoseShape trajectory 也完全一致。下一步唯一选择为 `POSE_LBS_FUSED_DOWNSTREAM_BAKEOFF`。
