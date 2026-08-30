# CUDA Pose normal equations backend

## 架构

`EvaluatorBackend` 和 CPU `run_lm` 接口保持不变。`CudaPoseEvaluator::linearize`
把 residual 与 `J_active[2N,69]` 留在两个相机各自的 persistent workspace；随后
`CudaPoseNormalEquationBuilder` 在 GPU 上执行：

```text
raw residual + J_active (GPU resident)
  -> Charbonnier IRLS point weight (float32, production formula)
  -> row scaling (float32)
  -> H_active = weighted_J^T weighted_J (FP64 accumulation)
  -> g_active = weighted_J^T weighted_r (FP64 accumulation)
  -> D2H: two 69x69 H blocks + two 69-vectors g
  -> CPU unchanged 69-to-75 scatter + pose prior
  -> CPU solver and LM controller
```

完整 Jacobian 不再回传 CPU。没有修改 Jacobian kernel、loss、参数布局、lambda
更新、accept/reject 或 termination。CPU backend 继续调用原
`build_pose_normal_equations`；CPU-only build 中 CUDA builder 使用明确的 unavailable
stub。

## 数值策略

GPU kernel 严格沿用 CPU production 的顺序语义：先以 float32 计算 IRLS weight
和 row scale，再形成 float32 weighted J/r，最后以 FP64 顺序累积 H/g。由于 CUDA
`sqrtf`、FMA 和 Eigen reduction 顺序不同，H/g 不要求跨 backend bitwise exact，
但 LM 控制决策必须一致，且未修改 tolerance。

