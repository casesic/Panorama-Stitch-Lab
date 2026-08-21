# DenseKP runtime session persistence

## 目标与边界

DenseKP delivery 使用长生命周期 `DenseKPRuntime`：启动时只创建一次 ONNX
Runtime session、加载一次模型并初始化一次 detector，之后连续处理多帧。DenseKP
模型、ERP_ROTATE 坐标、observation JSON、C++ LM 和 CPW 均未改变。

```text
startup
  -> DenseKPRuntime.load()
     -> ORT session + DenseKP weights
     -> detector + detector weights

per frame
  -> run_pipeline(args, runtime=runtime)
     -> fisheye -> ERP
     -> runtime.infer(cam2)
     -> runtime.infer(cam4)
     -> ERP -> ERP_ROTATE point transform
     -> DenseKP JSON

shutdown
  -> DenseKPRuntime.close()
```

单帧 `cli.py` 保持兼容；只有在同一 Python 进程中复用同一个 runtime，才能消除
跨帧初始化。`run_pipeline` 的 runtime profiling 使用累计计数器的帧间差值，因此
持久化路径的每帧 `ort_init_ms` 和 `detector_init_ms` 均为 0。

## 实测结果

CPU、frame 513/514、calibration cache hit：

| 项目 | 旧逐帧初始化 | persistent runtime |
|---|---:|---:|
| frame 513 wall | 3645.81 ms | 1132.43 ms |
| frame 514 wall | 3076.77 ms | 905.43 ms |
| persistent startup（一次） | - | 1575.64 ms |
| ORT session 创建次数 | 每帧 1 次 | 全部帧共 1 次 |
| detector 初始化次数 | 每帧 1 次 | 全部帧共 1 次 |

正确性验证：

- frame 513/514 DenseKP JSON bitwise equal，points max error 为 0；
- frame 513 C++ optimized DenseKP JSON bitwise equal；
- frame 513 CPW JPEG bitwise equal；
- `keypoint_value` 的语义和使用方式未变，仍不参与 LM loss/weight。

验证脚本：

```bash
conda run -n camerahmr python \
  src/script/validation/densekp_runtime_session_persistence.py \
  --frames 513 514 \
  --fisheye_root data_process/Output/20260311_riding/image/fisheye \
  --onnx_path Circular_Video/01_recv/20260708/CameraHMR/deploy/densekp_onnx/artifacts/densekp_b_256x256.onnx \
  --output /tmp/densekp_session_persistence_513_514 \
  --calibration_cache_dir /tmp/densekp_calibration_cache_new \
  --device cpu --camera old
```

验收结论：

```text
DENSEKP_RUNTIME_SESSION_PERSISTENCE_READY=PASS
```
