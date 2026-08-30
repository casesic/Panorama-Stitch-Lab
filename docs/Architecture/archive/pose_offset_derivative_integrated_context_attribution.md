# Pose-offset derivative integrated-context attribution

本轮只增加可关闭的 measurement instrumentation、真实 16-call input-only corpus 和 same-source replay，没有改变 optimized-native 数学、三个已集成 winner、layout、cache/reuse 或线程策略。

源码审计确认旧 `jac.pose_offset_derivative` timer 从 dispatch 前开始，到 `d_pose_offsets` 完整复制为 `d_v_posed` 后结束。它包含 output resize/allocation、Eigen Map setup、`result.noalias() = posedirs * derivative` 和一次 `[3N,66]` copy，但不包含此前独立构造的 `d_pose_feature`。PoseShape 仍复用 base Pose Jacobian 的 Q66 pose-offset derivative；Q76 只属于后续 combined LBS。

本轮 coarse full-pipeline median 为 275.228 ms / 3.633 FPS，距离 4 FPS 为 25.228 ms。Fine attribution 使用 9 个无 corpus-copy 的重复样本，16-call derivative family 为 81.810 ms：

| Exclusive 子项 | ms/frame | Family | Pipeline |
|---|---:|---:|---:|
| GEMM core | 79.723 | 97.45% | 28.97% |
| Output preparation | 0.927 | 1.13% | 0.34% |
| Postprocess copy | 0.860 | 1.05% | 0.31% |
| d_pose_feature build | 0.229 | 0.28% | 0.08% |
| Other glue | 0.007 | 0.01% | <0.01% |
| Map/setup | 0.001 | <0.01% | <0.01% |

Frame reconciliation error 为 0.077%。Map/setup 单次短于 clock overhead 的可靠界限，因此只作为“可忽略量级”诊断，不把其绝对值视为精确 wall latency。

同一 corpus、source、layout、binary policy replay 显示：warm-input core 34.203 ms，真实 trajectory-order core 81.118 ms，cache-disturbed diagnostic 33.503 ms；integrated core 79.723 ms，相对 trajectory-order isolated 为 0.983×。因此 integrated 与 apples-to-apples isolated 没有额外 slowdown；旧约 30.175 ms 主要是热 state/corpus protocol，不可作为真实 16-state trajectory 的直接基线。

每次调用都会 resize 一个新 output，16 次发生 16 次 reallocation，共申请 12,355,200 bytes；但 output preparation 只有 1.13%，不是主因。`d_pose_feature` 在同一 linearization 的双相机间逐字节一致，并被重复构造 8 次，但可避免成本只有 0.116 ms/frame。selected posedirs 内容按相机在 LM 内静态，不过现有 camera input value-copy 会形成临时副本；本轮仅记录，不修改。

GEMM core 若加速 2×，理论 pipeline 为 235.367 ms / 4.249 FPS；3× 为 222.080 ms / 4.503 FPS；5× 为 211.450 ms / 4.729 FPS；完全消除的理论上限为 195.506 ms / 5.115 FPS。GEMM core 远大于 11.782 ms Objective reuse 上界，是下一轮更优先的实验对象。

`POSE_OFFSET_DERIVATIVE_INTEGRATED_CONTEXT_ATTRIBUTION_READY=PASS`

`NEXT_OPTIMIZATION_EXPERIMENT=POSE_OFFSET_DERIVATIVE_GEMM_INTEGRATED_CONTEXT_BAKEOFF`
