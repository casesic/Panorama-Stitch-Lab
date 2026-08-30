# Pose/Shape C++ CPU LM reference

状态机复用 generic C++ LM runner，并通过 solve callback 接入 structured Schur。Production 的真实初始顺序是先 linearize，再以 `1e-3*max(max diag A_t,max diag C)` 初始化 Python-float/double λ，随后只计算一次 current true objective。rejected trials 复用当前 linearization；accepted point 才重新线性化，即使 accepted-step budget 随后耗尽也完成最后一次 relinearization。

Schur 失败由上层 dispatch 负责 dense fallback。正常 frame 513 全程使用 Schur。FP64 delta 先转 FP32，再与 FP32 current parameters 相加；beta latent 不额外 clamp。true objective 是 normalized Charbonnier reprojection、body-pose anchor 和 physical-beta anchor 之和。

Predicted reduction 使用未 damping 的 structured quadratic model，等价于 `-gᵀδ-0.5δᵀHδ`；actual 为 current objective 减 candidate objective。仅当 predicted、actual、rho 都大于零才接受。接受后 rho>0.75 除以3、rho<0.25乘3，其余不变；拒绝乘10。收敛顺序是 scaled gradient、step norm、relative objective。

frame 513 为一个 accepted trial 后达到 `MAX_ACCEPTED_STEPS`。Python/C++ solve、accept、lambda branch、status 和 counters 全部一致。本阶段未进入完整 staged pipeline，也未做性能优化。
