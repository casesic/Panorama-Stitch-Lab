# Pose objective value pose-offset GEMV bake-off

`evaluate_pose_camera_value()` 是 Objective 与 Jacobian linearization 共享的 value pose-offset 边界。完整 frame513 trajectory 实测共有 32 个 camera-call：Pose Objective/linearization 各 12 次，PoseShape Objective/linearization 各 4 次。

当前 scalar 以 `vertex → xyz → feature` 顺序，在 FP32 中计算 `offset[v,x]=Σ_f feature[f]·posedirs[f,v,x]`。`selected_posedirs` 的 feature-major `[486,N,3]` 存储可零拷贝解释为 ColMajor `[3N,486]`，因此 candidate 直接执行 `Y.noalias() = P * X`，没有 pack、unpack、persistent prepack 或跨 call cache。

在禁用 Eigen vectorization、无 `-march=native` 的 reference policy 下，Objective frame-weighted scalar/GEMV 为 47.922/7.100 ms，即 6.750×；Objective+linearization shared family 为 96.309/14.341 ms，即 6.716×。完整 GEMV shadow 通过既有 pipeline parity，first divergence 为 NONE。

当前 optimized-native 的真实 full pipeline 仍是 420.919 ms、2.376 FPS；本实验没有集成 native GEMV。仅作 ratio-transfer 时，Objective-only 为假设的 339.145 ms/2.949 FPS，shared family 为假设的 257.496 ms/3.884 FPS。32 个 calls 中发现 20 个 Objective↔linearization exact-state pair，但本轮只记录，不实施复用。

Candidate 为 STRONG_WIN，下一步是独立的 `POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF`。
