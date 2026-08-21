# DenseKP persistent runtime inference profiling

## 计时边界

Persistent runtime 每帧分别统计 cam2、cam4，然后在该帧 profile 中求和：

- `preprocess_ms`：人体 bbox crop、缩放到 256×256、normalize 和 NCHW FP32 tensor 准备；
- `detector_ms`：person detector 的 session inference；
- `bbox_postprocess_ms`：detector 输出 bbox 的收集、scale 和 center 计算；
- `densekp_model_ms`：DenseKP ONNX session inference；
- `decode_ms`：DenseKP 输出解码及 crop 坐标到原 ERP 坐标的仿射变换；
- `coordinate_ms`：138 个点从 ERP 到 ERP_ROTATE 的坐标变换。

输出文件为每帧的：

```text
result/densekp_inference_profile.json
```

完整的 `densekp_runtime_profile.json` 额外保留 preprocess 子项：
`densekp_crop_resize_ms`、`densekp_normalize_ms` 和
`densekp_tensor_prepare_ms`。所有 runner 计时均采用帧前后累计值之差，不包含
runtime startup，也不会累计之前帧的时间。

## CPU 实测

frame 513/514、cam2+cam4 合计：

| 阶段 | frame 513 | frame 514 |
|---|---:|---:|
| preprocess | 3.50 ms | 2.73 ms |
| detector | 134.77 ms | 113.51 ms |
| bbox postprocess | 0.11 ms | 0.09 ms |
| DenseKP model | 505.34 ms | 501.00 ms |
| decode | 0.51 ms | 0.49 ms |
| ERP→ERP_ROTATE | 0.20 ms | 0.20 ms |

frame 513 的 preprocess 进一步拆分为：crop/resize `1.76 ms`、normalize
`0.54 ms`、tensor prepare `1.08 ms`。因此下一步主要优化对象是 DenseKP ONNX
model inference，其次是 detector inference；Python preprocess、decode 和坐标变换
都不是当前瓶颈。

## 正确性

- profiling 前后 frame 513/514 DenseKP JSON bitwise identical；
- frame 513 C++ LM input 和 optimized DenseKP output bitwise identical；
- frame 513 CPW JPEG bitwise identical。

```text
DENSEKP_INFERENCE_PROFILING_READY=PASS
```
