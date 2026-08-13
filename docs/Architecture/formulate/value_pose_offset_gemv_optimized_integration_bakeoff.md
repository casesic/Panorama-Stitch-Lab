# Value pose-offset GEMV optimized integration bake-off

本轮在同一个 `-O3 -DNDEBUG -ffp-contract=off -march=native`、Eigen vectorization ON、Eigen 单线程二进制中比较 O2 与 O3。O2 保留两个旧 winner，并继续使用 scalar/std::fma value pose-offset；正式 `optimized_native` 更新为 O3，仅增加 native/vectorized zero-copy Eigen GEMV。Reference 保持不变，未集成 direct LBS、reuse、其它 Objective/Jacobian 优化、线程或 GPU。

真实 frame 513（cam2 1023 点、cam4 927 点）dispatch 为：O2 scalar/GEMV `32/0`；O3 `0/32`，fallback 为 0。O3 的 caller 分布为 Pose Objective 12、Pose Linearization 12、PoseShape Objective 4、PoseShape Linearization 4；两个旧 winner 仍分别调用 16 和 20 次。

O3 对 canonical Python 的 Stage1、Pose、PoseShape 参数、handoff、最终 beta 和 objective 均通过正式门禁。O2/O3 的 trial 数、accept/reject 序列、solver、objective/linearization call count 与 termination 全部一致，因此 `TRAJECTORY_EXACT=YES`，首个差异为 `NONE`。正式容差修改数为 0。

同轮实测结果：

| 模式 | Compute median | Compute FPS | Executable wall median | Wall FPS |
|---|---:|---:|---:|---:|
| O2 | 437.651 ms | 2.285 | 510 ms | 1.961 |
| O3 | 293.083 ms | 3.412 | 350 ms | 2.857 |

O2→O3 为 1.493×，节省 144.568 ms。O3 真实 compute 已超过 3 FPS，但未超过 4 FPS。上一轮 242.819 ms projection 比实测乐观 50.264 ms（20.70%）。integrated value pose-offset 为 18.852 ms，其中 Objective 9.441 ms、Linearization 9.411 ms；它是 isolated 6.979 ms 的 2.70×，但只占 pipeline 6.43%，不值得继续追该 kernel。20 对重复状态的 integrated reuse 上界为 11.782 ms，占 pipeline 4.02%，所以严格按“小于 2%”规则，`REUSE_PRIORITY_DOWNGRADED=NO`；本轮仍按范围约束不实施 reuse。

新主导类别是 Jacobian：192.477 ms / 65.67%。真实 Top1 为 `pose_jacobian/cam2/jac.pose_offset_derivative`，31.760 ms / 10.84%；Top2 为 cam4 同 kernel，29.441 ms / 10.05%。该 kernel 已是 GEMM winner，因此下一轮应先做 integrated-context attribution，解释其完整 pipeline 成本，而不是重新优化 value pose-offset。

`INTEGRATION_CLASSIFICATION=APPROVED`

`VALUE_POSE_OFFSET_GEMV_OPTIMIZED_INTEGRATION_BAKEOFF_READY=PASS`

`NEXT_OPTIMIZATION_EXPERIMENT=POSE_OFFSET_DERIVATIVE_INTEGRATED_CONTEXT_ATTRIBUTION`
