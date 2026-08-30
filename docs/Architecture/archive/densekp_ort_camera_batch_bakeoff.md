# DenseKP ORT camera-batch bake-off

## 当前接口

ONNX graph支持动态首维：

```text
input  img             [batch, 3, 256, 256] float32
output pred_keypoints  [batch, 138, 3]       float32
```

production 当前仍使用 `batch_size=1`，cam2、cam4 分别执行一次 ORT inference。

## 受控 A/B

在同一个 ORT CUDA session、相同 CUDA ERP image和相同 detector中，对 frame 513
比较：

| 路径 | ORT calls | items | inference |
|---|---:|---:|---:|
| sequential | 2 | 2 | 70.98 ms |
| camera batch=2 | 1 | 2 | 132.67 ms |

batch=2 反而慢 `86.9%`，没有达到 10% 收益门槛。同时：

```text
valid rules exact       yes
ERP_ROTATE max_abs      0.201171875 px
ERP_ROTATE mean_abs     0.05346746 px
ERP_ROTATE bitwise      no
```

动态 batch改变了 ORT/CUDA execution plan与浮点计算顺序，因此输出不再 exact。

## 决策

不保留 camera-batch wrapper，production继续采用两个 `batch=1` ORT调用。仅保留
`densekp_ort_call_count` 和 `densekp_ort_batch_item_count` profiling计数，便于未来
更换模型或ORT版本后重新评估。

```text
DENSEKP_ORT_CAMERA_BATCH=REJECTED
reason=SLOWER_AND_NOT_EXACT
```
