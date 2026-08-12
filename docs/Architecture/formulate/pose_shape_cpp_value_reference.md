# Pose/Shape value C++ CPU reference

单帧参数为 85D：frame Pose `[0:75]` 与共享 beta latent `[75:85]`。初值严格来自 production Stage1 LM → Pose LM → `build_pose_shape_stage_definition`，其中 latent 为 `atanh(beta_anchor/2.5)`。

Value 链先在 FP32 中计算 `beta=2.5*tanh(latent)`，用 selected beta shapedirs 构造两相机动态 shaped vertices，并用共享 `d_joints_d_beta` basis 构造唯一一份 dynamic joints。`d_joints_d_beta` 虽有 derivative 名称，但在这里是计算关节 value 的静态线性模型输入，不是本轮 Jacobian 输出。

动态 vertices/joints 随后传入现有 Pose common forward，复用 axis-angle、rotation、pose blend、FK、LBS、scale/translation 与 ERP。Rest translation严格为 `-dynamic_joints`，未误用 Pose 阶段的 fixed joints。

True objective由 Charbonnier reprojection、Pose anchor和physical-beta shape anchor组成。本轮不包含 beta/combined Jacobian、Arrowhead、Schur、Pose/Shape LM、CUDA或优化。完整结果和 timing 见 `pose_shape_cpp_parity.json`；通过后下一步为 Pose/Shape Jacobian C++ CPU。
