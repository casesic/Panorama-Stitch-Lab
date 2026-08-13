# LBS selected-transform vectorization bake-off

本实验固定 `DT.noalias() = W * DA`、RowMajor zero-copy layout、单线程与数值门禁，只比较三种 Release 编译策略：A 禁用 Eigen packet，B 仅启用默认 Eigen packet，C 在 B 上增加 `-march=native`。默认 scalar LBS、Direct candidate 与 production Python 均不变。

frame 513 的 cam2/cam4、Q66/Q76 六个真实 context 全部通过 dT、下游 LBS 以及既有 Python analytic parity。A→B 证明 Eigen packet vectorization 有明确收益，B→C 则证明该机器的 AVX2/AVX-512/FMA native codegen 仍有显著额外收益。C 虽设置 `-ffp-contract=off`，Eigen 显式 packet FMA 仍会生成 FMA；因此它是 host-specific build policy，不能当作通用可移植二进制。

最佳 LBS policy 与 pose-offset winner 都是 vectorized native。组合 fixed-trajectory 投影已经低于 1 秒并留有超过 10% 的 1 FPS 余量，而完整 LBS 降到 pipeline 的 10% 以下，因此本轮之后停止 LBS microbenchmark，下一步进入同一 executable 的完整 optimized-kernel integration 实测。

结果见 `data_process/Vis/cpp_reference_dump/lbs_selected_transform_vectorization/frame_000513/benchmark_summary.json`。
