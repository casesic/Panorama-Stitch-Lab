# C++ staged-LM 多帧参数与正规方程

第一版多帧 fitting 保持 Stage1、Pose 和 PoseShape 的单帧数学不变，只改变执行范围与 PoseShape 的共享体型参数布局。

## Pipeline

```text
对每帧依次运行 Stage1（12D）
          ↓
对每帧依次运行 Pose（75D）
          ↓
所有帧联合运行 PoseShape，共享一个 beta
```

当前实现为顺序 CPU 执行，不包含线程、GPU 或时间平滑损失。

## 参数布局

对于 `T` 帧：

```text
x = [frame_0(75), frame_1(75), ..., frame_{T-1}(75), z_beta(10)]
```

总维数为：

```text
75 * T + 10
```

每帧的 75D 参数为两台相机的旋转和平移，以及 63D body pose。整个参数向量末尾只有一个共享的 10D latent beta：

```text
beta = 2.5 * tanh(z_beta)
```

最终结果按帧提供 Stage1、Pose 和 PoseShape 75D 参数，并单独提供共享的 latent beta 与 physical beta。

## PoseShape 正规方程

每帧独立生成：

```text
A_t       [75, 75]
B_t       [75, 10]
C_t       [10, 10]
g_frame_t [75]
g_beta_t  [10]
```

聚合后的 Arrowhead 系统为：

```text
        x0      x1      x2      beta
x0      A0                       B0
x1              A1              B1
x2                      A2       B2
beta    B0^T    B1^T    B2^T     C
```

其中：

```text
C      = sum(C_t) + shared shape anchor Hessian
g_beta = sum(g_beta_t) + shared shape anchor gradient
```

Pose anchor 分别加入每帧的 `A_t` 和 `g_frame_t`。Shape anchor 只对共享 beta 添加一次，不随帧数重复或放大。系统直接交给已有的 `solve_pose_shape_schur()`，并设置：

```text
frame_count = T
frame_dim   = 75
beta_dim    = 10
```

## 帧间独立性

给定共享 beta 后，第 `t` 帧的 residual 和 Jacobian 不依赖其他帧的 pose 或 camera 参数。因此不同帧之间的 pose block 为零，帧间只通过共享 beta 的 `B_t` 产生耦合。

本版本未实现以下时间项：

```text
pose smoothness
translation smoothness
velocity / acceleration loss
temporal beta
```

这些损失会引入 frame-frame 非对角块，并改变当前 Arrowhead 结构。

## T=1 兼容性

`T=1` 直接调用原单帧 PoseShape O4 路径，从而保持 objective、正规方程、LM trajectory、最终参数和 beta 完全一致，不修改任何数值容差。

