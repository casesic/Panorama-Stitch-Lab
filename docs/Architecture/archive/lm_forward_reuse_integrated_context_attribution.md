# LM forward reuse integrated-context attribution

本轮只做 D0–D5 原因归因，不修改 Python production、通用 `run_lm` 或 LM 数学。O4 `optimized_native_stable_p` 继续是 preferred policy，O6 仍为 experimental。

诊断模式分别是 O4、完整双向复用、仅 Lin→Obj、仅 Obj→Lin、Obj→Lin 加丢弃式 shadow value-forward，以及保留 payload 但在线性化前销毁后走普通 O4 路径。D4 的 shadow 输出不参与数学；D5 的 payload 从不被 derivative 消费。

frame 513 的交错 Release 测量表明，O6 的主要额外时间仍集中在 derivative，尤其 pose-offset GEMM、selected-transform GEMM 和 LBS downstream。D3 能把惩罚完整定位到 Obj→Lin cached-value 路径；但 D4 没有恢复 derivative 性能，D5 也没有复现惩罚，因此“缺少普通 value forward 的一般性即时预热”和“仅仅保留大 payload 污染 working set”都不受数据支持。当前最准确的结论是：问题属于 Obj→Lin cached-result 的访问序列/数据驻留上下文，但更细的微观机制仍未确定。

完整数值、字段生命周期、buffer overlap、CPU cache 与 reconciliation 见 `data_process/Vis/cpp_reference_dump/lm_forward_reuse_integrated_context_attribution/frame_000513/benchmark_summary.json` 及同目录 CSV。

下一项且仅下一项实验为 `LM_FORWARD_REUSE_CACHED_VALUE_ACCESS_PATH_BAKEOFF`；本轮不实现 minimal payload、prefetch、warm-touch，也不正式保留 D2。
