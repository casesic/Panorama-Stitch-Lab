# Staged LM C++ CPU pipeline reference

frame 513 的 C++ pipeline 严格按 `Stage1 → Pose → Pose/Shape` 调用已经独立验证的 stage adapter。pipeline 模块只构造输入、执行 handoff、调用 stage 并汇总输出，不包含 SO(3)、ERP、Jacobian、normal equations 或 Schur 数学。

Production LM 的真实初始顺序统一记录为：initial linearization，随后根据 normal-equation 全局最大对角元初始化 lambda，最后计算 initial true objective。此前 Pose/Shape LM 汇报中的相反文字顺序属于文档矛盾，未修改 production runner 或数学。

本轮审计确认此前 Stage1 C++ reference 只覆盖 value、analytic Jacobian 与 H/g，并不包含 LM。因此新增最小 Stage1 LM adapter，复用 generic LM state machine；Pose 和 Pose/Shape 继续直接复用已有 adapter。

Stage1→Pose handoff 将 rigid 12D 原样复制，并从 body-pose anchor 初始化 63D body pose。Pose→Pose/Shape handoff将 frame 75D 原样复制，并用 `z = atanh(beta_anchor / 2.5)` 初始化 10D beta latent。C++ 内部 exact-copy invariant 与 beta reverse consistency 均通过。

跨 backend 的 stage boundary、status、trial/accepted/rejected 数、objective/linearization call count 和最终输出全部通过，`PIPELINE_FIRST_DIVERGENCE=NONE`。该结果只建立 CPU correctness baseline，不包含任何性能优化。

Release frame-513 三次 compute-only 为 5476.507、5437.663、5433.284 ms，median/min/max 为 5437.663/5433.284/5476.507 ms；对应理论单流基线约 0.184 FPS。executable wall-time median/min/max 为 5.46/5.45/5.51 s。以中位运行分解，Stage1、Pose、Pose/Shape 分别约占 0.67%、68.24%、30.97%；全部 Jacobian 合计占 93.29%，objective 4.81%，NE 1.18%，solve/control 0.60%。因此后续性能阶段首先应分析 pose-offset derivative 与 LBS derivative，而不是 Schur 或 LM control。

Release reference executable 的实测 peak RSS 约 116.8 MB，其中包括进程、静态 fixture/cache、reference bookkeeping 与输出缓冲。按 sequential stage 的最大同时存活集合而非三个 stage peak 相加，production-like working-set 保守估计约 100–117 MB；本轮未做 buffer lifetime 或内存优化。
