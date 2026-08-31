# DenseKP production inference backends

The production delivery runtime supports two explicit DenseKP inference
backends. `ort` remains the default and numerical reference. `trt` loads the
accepted TensorRT FP16 engine when the caller explicitly selects it and gives
an engine path. The runtime never selects a backend by probing files and never
silently falls back from TensorRT to ORT.

## Shared delivery path

`ORTRunner` owns the detector and the backend-independent DenseKP path:

```text
detector or guided bbox
  -> crop/resize
  -> BGR to RGB
  -> ImageNet normalization
  -> contiguous FP32 NCHW tensor
  -> backend inference seam (ORT or TensorRT)
  -> (prediction xy + 0.5) * 256
  -> affine restoration to the source ROI/image
  -> existing ERP restoration, validity, and matching
```

`TensorRTRunner` specializes only model initialization and inference. Raw
third-channel values are preserved as model output; they are not interpreted
as validity or confidence. Geometry-based validity rules are unchanged.

## TensorRT contract and lifecycle

The production runner targets TensorRT 10.7 in the Python 3.10 `camerahmr`
environment. It accepts an already-built fixed-B1 engine with this contract:

| Direction | Name | Shape | I/O dtype |
| --- | --- | --- | --- |
| input | `img` | `[1, 3, 256, 256]` | FP32 |
| output | `pred_keypoints` | `[1, 138, 3]` | FP32 |

The accepted engine uses FP16 internally and FP32 I/O. Production code does
not build or rebuild engines. It deserializes the supplied engine once per
`DenseKPRuntime`, validates all I/O names, shapes, and dtypes, and creates one
persistent execution context, CUDA stream, and pair of device buffers. FULL
and TRACK_GUIDED delivery share that runtime. Every enqueue is followed by an
explicit stream synchronization before CPU decode/postprocess, preserving the
serial production semantics.

Missing engines, dependency failures, deserialization failures, contract
mismatches, allocation failures, and execution failures are explicit errors.
They include the selected backend, engine path, or expected contract as
appropriate. There is no implicit ORT fallback.

## Configuration

The sequence entry keeps ORT as its default:

```bash
# Default/reference backend
--densekp-backend ort --onnx-path /path/to/densekp.onnx

# Accepted fixed-B1 TensorRT backend
--densekp-backend trt --trt-engine-path /path/to/densekp.plan
```

The standalone delivery CLI uses the equivalent `--backend` selection and
`--trt_engine_path` spelling used by its existing configuration interface.
TensorRT requires CUDA and `batch_size=1`. The production C++ staged-LM
evaluator remains CPU by default; backend selection does not alter fitting,
tracking, identity, detector, or matching policy.

## Numerical and performance expectations

TensorRT FP16 is functionally accepted, not bitwise equivalent to ORT FP32.
Normal floating-point differences can change a small number of strict
geometry-validity bits. Acceptance therefore uses task-level delivery,
identity, tracking, LM, and optimized-geometry checks rather than raw tensor
identity.

The integrated backend's real-crop fixed-B1 smoke (20 warmups and 100 measured
calls) measured 5.498 ms p50 and 5.534 ms p90 on the acceptance host. Sequence
throughput must still be reported separately for single- and multi-person
workloads because detector and CPU fitting costs remain significant.
