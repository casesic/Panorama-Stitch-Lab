# Pose Jacobian residual performance attribution

本轮只增加计时、调用链审计和决策报告；preferred `optimized_native_stable_p` 的数学与 dispatch 均未改变。Jacobian category 包含 linearization 所需的 camera value/residual forward 以及 derivative/J construction；IRLS、weighted J/r、H/g 和 Schur 保持在 Normal Equations category。

当前 LBS downstream 在同一个内层循环中同时计算 transform term 与 vertex term，因此只报告真实的 fused exclusive timer。为了拆成两个数字而重写循环会污染本轮 measurement-only 约束。

完整结果见 `benchmark_summary.json`。Pose-offset GEMM 仍是最大的绝对、可操作机会；最大的未优化 residual family 是 scalar fused Q66 LBS downstream，但其现实节省上界小于 GEMM blocking 的机会预算，因此下一步仍选择 blocking bake-off。
