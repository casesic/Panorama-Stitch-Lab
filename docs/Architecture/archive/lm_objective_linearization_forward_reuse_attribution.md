# LM Objective/Linearization forward reuse attribution

`run_lm` 的初始 state 顺序为 Linearization→Objective；accepted state 为 Objective→下一次 Linearization。Pose Jacobian 内嵌完整 camera value forward，PoseShape 的两条路径还重复 shape value 与 dynamic vertices。frame 513 runtime recorder 确认 Pose 6 对、PoseShape 2 对均为 byte-exact state，common results 也一致。

本轮只观测，不实现 cache、不改变回调顺序或数学。Objective reduction、anchor 与所有 derivative-only 工作均排除在 reuse saving 之外。ERP value API 内部计算后丢弃 Jacobian 是另一项独立 overcompute，未混入 reuse 上限。

未来优先 stage-local single-slot cache，以完整参数字节和 context generation identity 作 correctness key；rejected candidate 立即失效。结果通过 move ownership 与 const view 消费，避免深复制。
