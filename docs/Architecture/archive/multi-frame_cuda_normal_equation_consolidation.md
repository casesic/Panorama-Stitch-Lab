# Multi-frame CUDA PoseShape normal-equation consolidation

## 实现边界

每个frame-camera仍使用原有FP64 IRLS normal-equation kernels。新增GPU合并层：

```text
cam2/cam4 active H/g、pose-beta B、beta C/g
  -> GPU scatter到每帧 A_t/B_t/g_pose_t
  -> GPU生成每帧 C_t/g_beta_t
  -> 按frame顺序归约shared C/shared g_beta
  -> D2H最终Schur blocks
  -> 原CPU Schur solver和LM controller
```

pose/shape anchor的数值仍按原CPU表达式生成，然后作为小型prior blocks上传，避免改变
transcendental和float-to-double的计算语义。没有修改参数布局、loss、IRLS、Schur、
lambda/rho更新或tolerance。

## D2H before/after

T=8，704--711，每次LM运行包含两次normal-equation construction：

| 指标 | Before | After | 变化 |
|---|---:|---:|---:|
| D2H calls | 160 | 10 | -93.75% |
| D2H bytes | 1,441,280 B | 827,360 B | -42.60% |
| Nsight GPU copy time | 0.300 ms | 0.096 ms | -67.9% |
| normal-equation NVTX wall | 15.03 ms | 12.18 ms | -19.0% |

After每次construction只回传5个连续tensor：

| Tensor | T=8单次bytes | 两次调用数 |
|---|---:|---:|
| `A[T,75,75]` | 360,000 | 2 |
| `B[T,75,10]` | 48,000 | 2 |
| `C[10,10]` | 800 | 2 |
| `g_pose[T,75]` | 4,800 | 2 |
| `g_beta[10]` | 80 | 2 |

GPU端新增每帧consolidation kernel和一个shared-beta reduction kernel，没有做kernel
fusion。

## Pipeline benchmark

| Fixture | PoseShape before | PoseShape after |
|---|---:|---:|
| T=4 / 513--516 | 37.10 ms | 36.05 ms |
| T=8 / 704--711 | 79.78 ms | 73.77 ms |

单次运行会受GPU频率和前序Pose阶段影响；Nsight隔离的normal-equation range更直接地
反映本轮收益。

## Parity report

GPU consolidation改变了FP64 block的加法位置，但没有改变参与求和的项或顺序定义。
实际LM结果：

| 指标 | T=4 | T=8 |
|---|---:|---:|
| initial lambda | exact | exact |
| candidate objective | exact | exact |
| actual reduction | exact | exact |
| accepted/rejected | exact | exact |
| solver/status/trial count | exact | exact |
| predicted reduction abs diff | 1.09e-8 | 2.58e-8 |
| final parameter max abs | 0 | 7.45e-9 |
| differing FP32 parameters | 0 | 3 / 610 |

T=8的3个参数差异属于FP64归约末位传播到FP32后的1 ULP量级，没有触发任何LM控制
分支变化。这里的`FIRST_DIVERGENCE=NONE`指控制trajectory无分歧，不声称所有浮点
诊断字段byte-exact。

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```
