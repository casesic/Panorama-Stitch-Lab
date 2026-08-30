# LM Objective/Linearization forward reuse integration bake-off

O6 `optimized_native_stable_p_forward_reuse` 在 Pose 与 PoseShape wrapper 内使用 stage-local、role-aware、byte-exact single-use slot。O4 保留；generic `run_lm` API、控制流和回调次数不变。Payload 通过 move ownership 传递，不深复制。

frame 513 上 O6 精确命中 8 个 state，camera value forward 从 32 降至 16，shape/dynamic work 也减半，三阶段 trajectory 与最终结果完全一致。然而 ABBA 30+30 的完整 pipeline 测量显示 O6 反而变慢：被 Objective 保留后再交给 Linearization 的大 payload 改变了 derivative 的 integrated cache/working-set context，约 31 ms interaction 超过约 18.8 ms forward saving。

因此 O6 correctness 成立但不应升为 preferred。下一步应对 retained payload、冷 cache 与 derivative slowdown 做 integrated-context attribution，而不是继续扩展 cache 或立即优化别的 kernel。
