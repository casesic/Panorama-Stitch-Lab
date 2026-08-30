# Pose/Shape C++ CPU Schur reference

Production 接收外部 scalar damping，并使用 `Aλ=A+λI`、`Cλ=C+λI`；B 与已经进入 A/C/g 的 priors 不变。系统符号是 `Hδ=-g`。每帧通过一次 Cholesky 联合求解 `[g_t,B_t]`，得到 `y=Aλ⁻¹g` 与 `Y=Aλ⁻¹B`，随后构造 `S=Cλ-ΣBᵀY`、`rhs=-g_beta+ΣBᵀy`，求 beta 后执行 `δ_t=-y-Yδ_beta`。

Production structured solver 的 local/reduced Cholesky 任一失败即返回 `None`；dense fallback 位于上层 solve dispatch，而非 Schur kernel 内。C++ reference 保持这一分层和 failure flag，并仅把 dense 85D solve作为诊断。

验证分两层：kernel fixture 合法输入仅为 FP64 A/B/C/g 和三个外部 λ，不包含 delta/Y/S；integration 使用已验证的 C++ NE blocks。两层 first divergence 均为 NONE。另有 T=3、B=0、g=0、符号和 large-damping synthetic tests。

本阶段不初始化或更新 λ，不计算 candidate/objective/rho，也不执行 LM trial。下一步可进入 `POSE_SHAPE_LM_CPP_CPU`。
