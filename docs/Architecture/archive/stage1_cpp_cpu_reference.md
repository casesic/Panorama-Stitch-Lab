# Stage 1 C++ CPU reference

This is the first standalone C++ implementation of staged LM. Its scope is one-frame Stage 1 rigid value/Jacobian/normal-equation evaluation only; it contains no SMPL-X, LM loop, Pose, Pose/Shape, CUDA, or performance optimisation.

The formulas remain documented in [cpp_port_computation_graph.md](cpp_port_computation_graph.md) and [staged_lm_stage_formulas_and_jacobians_typora.md](staged_lm_stage_formulas_and_jacobians_typora.md). Python production and the generated reference dump are authoritative.

## Layout and data flow

The implementation is under `cpp/staged_lm`: `so3` implements the canonical axis-angle kernels, `erp` implements unified ERP projection/Jacobian, `stage1` constructs residuals and rigid Jacobians, and `binary_io` handles checked raw-array I/O.

The fixture contains only `parameters`, local vertices, rotation centers, and targets for cam2/cam4. C++ computes every reference output itself. All geometry, residual, IRLS, and weighted-J/r operations stay in float32. Only after cam2 rows followed by cam4 rows are concatenated are J/r cast to float64 and reduced by one Eigen matrix multiplication into H/g.

## Build and run

```bash
python src/script/validation/staged_lm/export_stage1_cpp_fixture.py \
  --reference_dir data_process/Vis/cpp_reference_dump/stage1/frame_000513

/usr/bin/cmake -S cpp/staged_lm -B build/staged_lm -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build build/staged_lm -j

./build/staged_lm/stage1_reference_main \
  --fixture data_process/Vis/cpp_reference_dump/stage1/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_reference_dump/stage1/frame_000513/cpp_output

python src/script/validation/staged_lm/check_stage1_cpp_reference.py \
  --reference data_process/Vis/cpp_reference_dump/stage1/frame_000513/stage1_reference.npz \
  --cpp_output data_process/Vis/cpp_reference_dump/stage1/frame_000513/cpp_output
```

## Current numerical result

SO(3) `R/Jr`, transformed points, and `dpoint_dorient` are bitwise identical for frame 513. All prescribed float32 geometry, ERP, Jacobian, residual, IRLS, and weighted-array gates pass. ERP first differs by at most 0.000244141 pixels because PyTorch's AVX2 vector math and the standalone C++ standard math library round `atan2/asin` differently.

That float32 difference propagates to authoritative H/g at roughly `1e-7` relative error. The approved cross-backend policy therefore uses `1e-6` as the H/g relative pass gate and `5e-7` as an early-warning gate. This is an explicit policy rather than an unrecorded tolerance relaxation: float64 accumulation cannot remove differences already present in float32 weighted J/r. Frame 513 reaches H `1.33e-7` and g `6.79e-8`, both below the warning line, so `STAGE1_CPP_CPU_REFERENCE_READY=PASS`.

The next permitted scope is Pose value-only C++ CPU (`R → pose feature → pose offsets → FK → LBS → vertices → ERP`). Pose Jacobians, LM and CUDA remain out of scope.
