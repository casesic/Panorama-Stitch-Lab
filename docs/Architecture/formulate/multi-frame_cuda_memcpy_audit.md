# Multi-frame CUDA PoseShape memory-transfer audit

## 范围

本报告审计 persistent-workspace 优化后的 T=8 PoseShape evaluator，fixture 为
704--711，设备为 RTX 3090。统计来自 Nsight Systems，并由以下 NVTX ranges 隔离：

```text
multiframe_cuda_pose_shape_workspace
multiframe_cuda_pose_shape_forward
multiframe_cuda_pose_shape_linearize
multiframe_cuda_pose_shape_normal_equations
```

只分析 H2D/D2H；没有修改 kernel、solver、数学、tolerance 或参数布局。

## 总览

### LM runtime，不含 workspace construction

| 方向 | 调用数 | 总字节 | GPU copy time | blocking API wall |
|---|---:|---:|---:|---:|
| H2D | 512 | 597,760 B | 0.273 ms | 2.397 ms |
| D2H | 576 | 2,025,600 B | 0.885 ms | 5.632 ms |
| 合计 | 1088 | 2,623,360 B | 1.158 ms | 8.029 ms |

传输数据量本身不大，但由 1088 次同步 `cudaMemcpy` 组成，所以 API wall 远高于实际
DMA 时间。历史 38.25 ms 是 persistent-workspace 之前的统计；当前 LM runtime copy
API 已降至约 8.03 ms。

### Workspace construction

| 方向 | 调用数 | 总字节 | GPU copy time | blocking API wall |
|---|---:|---:|---:|---:|
| H2D |约 384 | 40,624,608 B | 5.605 ms | 11.531 ms |

这里上传的是固定模型和每帧相机资产，已经保持 GPU resident，LM 迭代中不再重复上传。

## Runtime H2D audit

下表调用数均为整个 T=8 PoseShape LM trajectory，两次 forward、两次 linearization。

| Tensor | 单次 bytes | 调用数 | 调用位置 | 能否保持 resident |
|---|---:|---:|---|---|
| latent beta10 | 40 | 16 forward + 16 linearize | `evaluate_pose_shape_value_cuda_with_workspace` | 参数由 CPU LM 更新，不能永久固定；可与 pose/camera 打包为每帧一次上传 |
| camera translation3 | 12 | 32 forward + 64 linearize | `evaluate_pose_camera_value_cuda` | 动态参数；可直接从一块 GPU parameter buffer 读取 |
| axis angles165 | 660 | 32 forward + 96 linearize | camera value及Pose Jacobian入口 | 动态，但当前在CPU重复组装；可在GPU从参数块展开 |
| shaped joints55x3 | 660 | 32 forward + 64 linearize | `update_cuda_pose_workspace_shape` | 可以；它刚由GPU shape kernel生成，目前发生GPU→CPU→GPU回传 |
| dynamic support vertices204x3 | 2448 | 32 forward + 64 linearize | `update_cuda_pose_workspace_shape` | 可以；刚由GPU dynamic-vertices kernel生成，应共享device pointer |
| `d_beta/d_latent` | 40 | 32 linearize | beta Jacobian入口 | 可以在GPU计算，或每帧只生成一次供cam2/cam4共享 |
| `d_joints/d_latent` 165x10 | 6600 | 32 linearize | `evaluate_pose_shape_beta_jacobian_cuda_with_workspace` | 可以；cam2/cam4内容相同，目前每相机上传一次 |

源码位置：

- `cpp/staged_lm/src/cuda_forward.cu`：persistent shape forward、camera value、shape
  workspace更新和beta Jacobian；
- `cpp/staged_lm/src/cuda_evaluators.cpp`：PoseShape forward/linearize编排。

最值得消除的H2D不是字节最大的某一块，而是以下回环：

```text
GPU shape/dynamic kernel
  -> D2H joints + dynamic vertices
  -> H2D 到 value workspace
  -> H2D 到 linearization workspace
```

应让shape输出直接成为后续camera evaluator的GPU-resident输入。

## Runtime D2H audit

### Forward与linearize中的value输出

两类range各产生完全相同的292,160 B D2H：

| Tensor | 单次 bytes | 每类range调用数 | CPU是否需要 | 建议 |
|---|---:|---:|---|---|
| `tanh_latent` | 40 | 16 | linearize计算链式导数需要 | 可在GPU继续计算导数；objective不需要下载 |
| physical beta | 40 | 16 | CPU shape prior当前需要 | 可只回传最终objective标量，或暂时保留 |
| shaped joints | 660 | 16 | 当前用于构造CPU `PoseModelInput` | GPU camera路径不需要CPU副本，可resident |
| cam2/cam4 dynamic vertices | 2448 | 32 | 当前用于更新camera workspace | 不需要CPU副本，可resident |
| final support vertices | 2448 | 32 | objective不使用 | 可resident，当前属于多余下载 |
| DenseKP 3D observation points | 1656 | 32 | objective不使用 | 可resident，当前属于多余下载 |
| projected UV138x2 | 1104 | 32 | objective只使用residual | 可resident，当前属于多余下载 |
| raw residual138x2 | 1104 | 32 | CPU objective当前使用 | 若objective reduction迁GPU，只回传一个标量 |

特别是linearize开头调用的完整PoseShape value forward，其camera value结果随后没有被
linearization消费；这部分不仅产生D2H，也重复执行forward。该问题属于后续
forward-reuse范畴，本轮仅记录、不修改。

### Normal-equation blocks

每个tensor调用32次，即 `2 linearizations x 8 frames x 2 cameras`：

| Tensor | 单次 bytes | 总字节 | CPU是否需要 | 建议 |
|---|---:|---:|---|---|
| active H 69x69 FP64 | 38,088 | 1,218,816 | CPU当前用于scatter到A | 可先在GPU scatter/双相机合并，再回传每帧A |
| active gradient69 FP64 | 552 | 17,664 | CPU当前用于scatter | 可先在GPU合并 |
| pose-beta cross69x10 FP64 | 5,520 | 176,640 | CPU当前用于B | 可先在GPU scatter/合并 |
| beta Hessian10x10 FP64 | 800 | 25,600 | shared C需要 | 可跨相机、跨帧GPU累积后只回传一份shared C |
| beta gradient10 FP64 | 80 | 2,560 | shared gradient需要 | 可跨相机、跨帧GPU累积后只回传一份shared g_beta |

CPU Schur最终确实需要 `A_t/B_t/g_pose_t/shared C/shared g_beta`，但不需要每个camera的
active中间块。把scatter和跨camera/frame累积留在GPU，可显著减少160次小D2H调用和
约1.44 MB中间传输，同时不需要修改Schur数学。

## Workspace-construction H2D

主要固定资产如下：

| Tensor | 典型单次 bytes | T=8调用数 | 状态 |
|---|---:|---:|---|
| selected posedirs204x3x486 | 1,189,728 | 32 | 已resident；当前value/linearize、cam2/cam4各有副本 |
| selected LBS weights204x55 | 44,880 | 32 | 已resident，但存在workspace间重复副本 |
| beta shapedirs204x3x10 |约24,480 |约32 | 已resident，但shape/value/linearize间有副本 |
| selected shaped vertices204x3 | 2,448 | 48 | 初值resident，之后动态更新 |
| joints55x3 | 660 | 40 | resident，但多个camera workspace重复 |
| joint basis165x10 | 6,600 | 8 | 每帧一份，实际上可由所有frame共享一份 |
| parents55 int64 | 440 | 32 | 固定且重复，可全局共享 |
| DenseKP targets138x2 | 1,104 | 32 | 每frame-camera固定，已resident |
| DenseKP sparse ids/offsets/indices/weights | 552/556/816 B | 多份 | 已resident，可在同模型/相机间共享只读资产 |

这些传输不发生在LM迭代内，但构造成本仍计入当前单次pipeline。长期复用整个
multi-frame evaluator可以摊薄；进一步共享只读model storage则可减少约40.6 MB启动上传。

## 优先级结论

1. 让GPU shape joints和dynamic vertices直接供camera value/Jacobian使用，消除
   GPU→CPU→GPU回环；
2. objective在GPU reduction，只回传objective标量，避免vertices、DenseKP 3D、UV和
   residual整块D2H；
3. GPU端完成cam2/cam4 scatter及shared-beta累积，只回传Schur最终blocks；
4. 将pose、camera、beta打包成少量H2D，而不是数百次12--6600 B小copy；
5. 多帧共享posedirs、LBS、parents、regressor等只读model storage，降低构造上传。

本轮只完成审计。profiling前后的trajectory和final parameters保持byte-exact。

第3项已在后续实现，结果见
[`multi-frame_cuda_normal_equation_consolidation.md`](multi-frame_cuda_normal_equation_consolidation.md)。

```text
CONTROL_DECISIONS_EXACT=YES
FIRST_DIVERGENCE=NONE
TOLERANCE_CHANGE_COUNT=0
```
