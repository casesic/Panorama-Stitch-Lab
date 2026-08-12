# Pose/Shape C++ CPU structured normal equations

Production 链为：point-major raw residual → frozen Charbonnier IRLS → FP32 row scaling → `weighted r/J_active/J_beta` → FP64 batched contractions → 69D active scatter 到 75D frame → 跨相机/帧累加共享 beta 块。块方向为 `A[75,75]`、`B[75,10]`（frame-to-beta）、`C[10,10]`。

Pose prior 是 `s_p(q_body-q_anchor)`，只贡献 `A[12:75,12:75]` 与 `g_frame[12:75]`。Shape prior 作用在 physical beta：`r_s=s_s(2.5*tanh(z)-beta_anchor)`，Jacobian 为 `s_s*diag(d beta/dz)`，只贡献 `C` 与 `g_beta`。

FP64 boundary 位于 row scaling 之后；H/g 不在 FP32 累积。validation 将 structured 块重建为 85D dense H/g，并与同一批 weighted rows 构造的 dense reference 对比，以检查 B 方向、相机 rigid scatter、共享 body 与全局 beta。

frame 513 的 UV 跨 backend 绝对误差小于 0.001 px，但小 residual 经 IRLS 倒数平方根放大。因此正式 Pose/Shape NE policy 保留 absolute upstream gates，并对 weighted J 使用 `rel<=2e-5,max_abs<=1e-3`、中间及最终 H 使用 `rel<=5e-6`、gradient 使用 `rel<=2e-4,max_abs<=1e-3`。这是记录过原因的 cross-backend policy，不涉及修改 production 数学。

本阶段只构造 A/B/C 与梯度，没有 Schur、damping、solve 或 LM。下一步可进入 `POSE_SHAPE_SCHUR_CPP_CPU`。
