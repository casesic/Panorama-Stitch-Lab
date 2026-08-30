# Pose LM C++ CPU reference

## Python production state machine

Pose 从 Stage1 handoff 的 FP32 75D 参数开始。production 首先构造一次 frozen-weight linearization 和 FP64 H/g，然后令 `lambda=1e-3*max(diag(H))`，再计算一次 initial true objective。每个 trial 使用 Cholesky 求解 `(H+lambda I) delta=-g`，失败才回退 general dense solve。

成功求解得到的 FP64 delta 先 cast FP32，再执行 FP32 candidate update。candidate acceptance 使用 value-only Pose forward 的真实 Charbonnier objective与 pose anchor，而不是 frozen IRLS quadratic。Predicted reduction在 FP64中计算；actual、rho 和 lambda 是 Python/C++ double scalar。

接受条件严格为 predicted、actual、rho 全部大于零。接受时 rho 大于 0.75 将 lambda 除以 3，小于 0.25 乘以 3；拒绝乘以 10，并 clamp 到 `[1e-10,1e12]`。Rejected trial 复用 H/g；accepted candidate 在下一 outer 前重新线性化。即使第五个 accepted step耗尽 budget，production 仍进行最后一次 relinearization，然后返回 `MAX_ACCEPTED_STEPS`。

收敛检查顺序为 scaled gradient、step norm、relative objective。无 accepted trial 时保留 legacy classification：accepted steps 大于 4 且 scaled gradient 小于 `1e-4` 才记作 gradient convergence，否则 `LM_NO_ACCEPTED_TRIAL`。

## Objective lifecycle and trajectory

Initial objective只计算一次，每个真正 evaluated candidate只计算一次；accepted candidate objective成为 current objective，下一 outer 和 final 都直接复用。Frame 513 轨迹为 6 trials，accept/reject `[A,A,A,R,A,A]`，5 accepted、1 rejected、6 H/g calls、7 objective calls，final status `MAX_ACCEPTED_STEPS`。

C++ 直接复用已验证的 Pose value、Jacobian、IRLS/normal-equation模块。Branch、lambda、convergence、status和counters全部匹配，`LM_FIRST_DIVERGENCE=NONE`。最小 rho branch margin远大于 Python/C++ rho差异，因此轨迹分支稳定。

本阶段不包含 Stage1 LM、Pose/Shape、beta、Schur、CUDA或性能优化。下一步可以进入 Pose/Shape value C++ CPU。
