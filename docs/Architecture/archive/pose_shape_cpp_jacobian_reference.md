# Pose/Shape C++ CPU Jacobian reference

Frame 513 的组合导数空间为 `[pose/root 66 | beta latent 10] = 76`。平移不进入 FK 导数空间，而是在 ERP 后通过 `J_proj` 插入 69 维相机活动块；beta 始终是 cam2/cam4 共享的 10 维全局块。

beta 使用 `beta=2.5*tanh(z)`。它同时经由 `v_shaped → v_posed` 和 `joints → FK → Rest → relative transform` 影响 LBS。实现显式保留 `d_skin_from_vertex`、`d_skin_from_transform`，并将 `dG*Rest` 与 `G*dRest` 分开检查，避免遗漏 dynamic Rest 的导数。

最终每个相机输出 `J_active[N,2,69]` 与 `J_beta[N,2,10]`。Python dump 直接核对 canonical production 的这两个块；fixture 仅含 value/static inputs，没有导数输出。

跨 backend policy 为 SO(3)/shape 基础链 `rel_frob <= 1e-6`，组合 FK/LBS/vertex 与最终 Jacobian `<=2e-6`。frame 513 的首个差异为 `NONE`，所有 invariant 与 C++ self-test 通过。详细误差和 timing 见同一 reference 目录下的 parity JSON/Markdown。

本阶段只证明 analytic combined Jacobian 正确性，没有构造 Arrowhead normal equations、Schur 或 Pose/Shape LM。下一步可进入 `POSE_SHAPE_NORMAL_EQUATIONS_CPP_CPU`。
