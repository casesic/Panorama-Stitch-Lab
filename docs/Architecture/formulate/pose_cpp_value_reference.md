# Pose value C++ CPU reference

This reference covers frame 513, CPU, cam2 and cam4, and the value chain only. It does not contain Pose derivatives, Jacobians, normal equations, LM, Pose/Shape or CUDA.

## Inputs and flow

The input-only fixture stores the real Stage1-to-Pose 75D handoff parameters, frozen-beta joints, parents, frozen hand poses and scale, selected shaped vertices, selected posedirs, selected LBS weights, and targets. It never stores reference rotations, transforms, vertices or ERP outputs.

The standalone C++ flow is:

```text
axis angles → rotations → pose feature → pose offsets → v_posed
→ local/global FK → production rest correction → selected LBS
→ scale/translation → ERP → residual
```

All matrices are serialized in canonical row-major order by explicit loops; Eigen's internal column-major storage is never written directly. Small matrix products retain the Python parent/child and weighted-transform operation grouping, with fixed float32 FMA accumulation.

## Commands

```bash
python src/script/validation/staged_lm/dump_pose_cpp_reference.py ...
python src/script/validation/staged_lm/export_pose_cpp_fixture.py \
  --reference_dir data_process/Vis/cpp_reference_dump/pose/frame_000513
/usr/bin/cmake -S cpp/staged_lm -B build/staged_lm -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build build/staged_lm -j
./build/staged_lm/pose_reference_main \
  --fixture data_process/Vis/cpp_reference_dump/pose/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_reference_dump/pose/frame_000513/cpp_output
python src/script/validation/staged_lm/check_pose_cpp_reference.py \
  --reference data_process/Vis/cpp_reference_dump/pose/frame_000513/pose_reference.npz \
  --cpp_output data_process/Vis/cpp_reference_dump/pose/frame_000513/cpp_output
```

## Current parity

Axis angles, rotations, pose feature, pose offsets, v_posed, FK, rest correction, selected LBS, final vertices, ERP uv and residual all pass their prescribed gates. UV requires both relative norm `<= 1e-6` and max absolute error `<= 1e-3 px`. Residual requires max absolute error `<= 1e-3 px` and a passing upstream UV gate; its relative norm remains diagnostic because `uv - target` can amplify relative error through subtractive cancellation.

Frame 513 has max UV/residual absolute error `4.88e-4 px`, below half of the pixel-space limit, and reports `FIRST_DIVERGENCE=NONE`. No reference output is consumed by C++, and no SLEEF/LibTorch dependency is introduced. `POSE_VALUE_CPP_CPU_REFERENCE_READY=PASS`.

## Production details

Serial Pose value rotation uses `_aa_to_rotmat`, while the batched derivative path uses `geometry.so3`; C++ reuses the validated Stage1 SO(3) kernel. Installed SMPL-X rest correction uses subtraction of the transformed rest joint rather than directly multiplying a Rest matrix, and C++ follows that exact grouping.
