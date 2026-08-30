# Pose Objective pose-offset GEMV 向量化 bake-off

本轮只比较同一份 zero-copy Eigen GEMV 的编译策略，没有修改 GEMV 数学、Python production、reference、当前 optimized-native 集成或正式容差，也没有实施 integration/cache/reuse。冻结输入是 frame 513、cam2 1023 点、cam4 927 点的 32 个真实 Objective/Linearization 状态；两轮顺序为 `A B C / C B A`，每状态预热 5 次、每轮重复 100 次。

构建隔离如下：A 为 `-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off`；B 只移除 `EIGEN_DONT_VECTORIZE`；C 只在 B 上增加 `-march=native`。A 无 Eigen packet，B 实际启用 SSE，C 实际启用 SSE/AVX/AVX2/AVX-512；Eigen 单线程。GEMV scoped assembly 中三者均无融合乘加，符合 `-ffp-contract=off`；scalar reference 则显式用 `std::fma` 累积。M/K 的非整包尾部由 Eigen 处理，cam2/cam4 均为 `M mod 16 = 13`、`K mod 16 = 6`。输入输出对齐按每个 call 记录，仍为 zero-copy、无用户侧 pack/unpack 和显式临时区。

32-state 汇总（各状态两轮样本合并后取中位数，再求和）：

| 范围 | A (ms) | B (ms) | C (ms) | A→B | B→C |
|---|---:|---:|---:|---:|---:|
| Objective | 7.147 | 3.785 | 3.493 | 1.888× | 1.083× |
| Linearization | 7.128 | 3.775 | 3.486 | 1.888× | 1.083× |
| Shared family | 14.275 | 7.559 | 6.979 | 1.888× | 1.083× |

因此单独开启 Eigen vectorization 是 strong win；`-march=native` 额外收益为 marginal，但 C 仍是最低延迟且 correctness PASS 的候选。C 相对旧 scalar shared-family 为 13.799×。A/B/C 对 scalar 和 Python 的 32-state value parity 全部通过；B/C 的完整 Pose/PoseShape Objective、linearization、residual/IRLS、H/g 与 LM trajectory shadow 均按既有 pipeline policy 通过，`PIPELINE_FIRST_DIVERGENCE=NONE`。

本轮没有整合，所以真实 pipeline 仍是 420.919 ms / 2.376 FPS。比例迁移仅作假设：Objective-only 为 331.920 ms / 3.013 FPS，shared-family 为 242.819 ms / 4.118 FPS。20 对 exact duplicate state 在 best GEMV 下的未来复用上界降为 4.363 ms，reuse 优先级随之下降。

结论：C 与当前 optimized-native 使用相同 build policy，应进入下一轮完整 pipeline integration 实测。

`POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF_READY=PASS`

`NEXT_OPTIMIZATION_EXPERIMENT=VALUE_POSE_OFFSET_GEMV_OPTIMIZED_INTEGRATION_BAKEOFF`
