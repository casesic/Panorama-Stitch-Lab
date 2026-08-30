# Multi-frame CUDA Pose batched memory/reduction

## 实现

Multi-frame Pose batch 新增长期驻留的
`CudaMultiFramePoseBatchWorkspace`。每次 active batch 将以下数据组织成一个连续
buffer 后上传：

```text
[B, parameters75 + cam2(axis165, translation3)
                  + cam4(axis165, translation3)]
```

Pose value/Jacobian 直接读取对应 device slice，不再为每帧、每相机分别上传
axis-angle 和 translation。固定模型、手部姿态和 body-pose anchor 由 evaluator
workspace 长期持有。

两相机的 69 维 active normal equations 保持在各自 GPU workspace 中。新增
batch consolidation kernel，在 GPU 上完成：

```text
cam2 active H/g + cam4 active H/g
  -> scatter 到每帧 75 维 H/g
  -> 加 pose anchor H/g
  -> contiguous [B, H75x75, g75]
```

最终 H/g 使用一次 D2H 返回 CPU。CPU LM controller、每帧 lambda/trial/state、
Cholesky/dense solver、loss、Jacobian、参数布局与 tolerance 均未改变。

## Transfer 结果

统计覆盖完整 Pose LM 中的动态参数上传和 normal-equation 输出：

| 配置 | H2D count before -> after | H2D bytes before -> after | D2H count before -> after | D2H bytes before -> after |
|---|---:|---:|---:|---:|
| T=4 | 244 -> 26 | 97,536 -> 80,752 | 96 -> 6 | 1,854,720 -> 1,094,400 |
| T=8 | 484 -> 26 | 193,728 -> 159,856 | 192 -> 6 | 3,709,440 -> 2,188,800 |

H2D 次数下降来自每个 evaluator batch 只进行 packed-data 与 active-index 两次
上传；D2H 次数下降来自每次 linearization 只回传一次连续 H/g。

## Correctness

T=1 继续复用既有单帧 CUDA Pose。T=4（513--516）与 T=8（704--711）
相对同一二进制的 sequential CUDA 诊断路径：每帧 Pose trajectory、Pose final
parameters、后续 PoseShape trajectory 均 bitwise exact。

```text
CONTROL_DECISIONS_EXACT=YES
TRAJECTORY_EXACT=YES
FINAL_PARAMETERS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```

## Wall time

单次同配置测量：

```text
                 优化前 batch    优化后 batch
T=4 Pose          259.89 ms        259.07 ms
T=8 Pose          307.67 ms        306.27 ms
```

传输调用已经显著合并，但 wall-time 只基本持平。这说明当前 Pose batch 的主要
剩余成本不再是 H/g 回传次数；默认流上的逐帧 kernel launch、同步 value output
以及 kernel 本身仍占主导。该结论不影响本轮 memory/reduction 优化的正确性，
但后续不能仅继续压缩 H/g copy 来期待明显 FPS 提升。
