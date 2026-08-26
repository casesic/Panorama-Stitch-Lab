# Staged-LM evaluator backend boundary

## 目标

本次只把 Pose 与 PoseShape 的 CPU 数值计算从 LM controller 中抽离，给后续
CUDA evaluator 留出稳定边界。当前 production policy 仍为
`optimized_native_stable_p`，没有 CUDA kernel，也没有改变 objective、Jacobian、
normal equations、Schur、参数布局或 LM 控制数学。

```text
run_pose_lm / run_pose_shape_lm
              |
              v
       CPU LM controller (run_lm)
       - damping / trial
       - solve
       - accept / reject
       - convergence
              |
              v
       EvaluatorBackend
       - forward(parameters)
       - evaluate_residual()
       - linearize(parameters)
       - build_normal_equations()
              |
       +------+------------------+
       |                         |
CpuPoseEvaluator       CpuPoseShapeEvaluator
       |                         |
existing CPU value / Jacobian / NE / Schur implementation
```

## 调用契约

Objective evaluation 使用一对调用：

```text
forward(parameters)
evaluate_residual()
```

Linearization 使用另一对调用：

```text
linearize(parameters)
build_normal_equations()
```

CPU evaluator 在两次调用之间持有对应的 value 或 Jacobian 中间结果。计算顺序仍为
cam2 后 cam4，随后按原实现构造 objective 或 normal equations。`run_lm` 新增接收
`EvaluatorBackend&` 的入口，内部 damping、求解和状态机保持原样。旧 callback 入口
继续供 Stage1、multi-frame 现有实现和轻量 self-test 使用。

## CUDA 后续边界

后续 CUDA migration 只需新增实现同一接口的 backend。第一版仍可让 CPU
`run_lm` 管理状态和小型求解，而 evaluator 在设备侧完成 forward、residual、
Jacobian 与 normal-equation reduction。此次没有加入 CUDA 占位分支或 runtime
dispatch，避免尚未实现的路径进入 production。

## Production policy

```text
preferred_policy=optimized_native_stable_p
evaluator_backend=cpu
cuda_kernel_count=0
tolerance_change_count=0
```
