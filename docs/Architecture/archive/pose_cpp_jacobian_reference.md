# Pose analytic Jacobian C++ CPU reference

## Scope and parameter spaces

This reference extends the validated Pose value path through the raw analytic 2D Jacobian only. It contains no row weighting, H/g, anchors, LM, Pose/Shape or CUDA.

- Internal 66: camera root orientation `[0:3]`, shared body pose `[3:66]`.
- Active 69: camera orientation `[0:3]`, camera translation `[3:6]`, body pose `[6:69]`.
- Frame 75: cam2 rigid `[0:6]`, cam4 rigid `[6:12]`, shared body pose `[12:75]`.

The derivative flow is `Jr → dR → d feature → d offsets/v_posed → d local/global/rest-corrected transforms → d selected LBS → d vertices → ERP J → J_uv_rotpose → J_active → J_frame`. Root orientation does not enter the non-root pose feature, so `d_pose_feature[:,0:3]` is exactly zero, while its final FK/ERP columns are generally nonzero.

## Storage and fixture

The input fixture is unchanged from Pose value-only and contains no derivative or Jacobian reference output. Every high-dimensional C++ binary is serialized explicitly in canonical row-major logical axis order. `cpp_output_meta.txt` records shapes and axes such as `joint,row,col,param` and `vertex,row,col,param`.

## Parity

Frame 513 uses cam2 1023 points and cam4 927 points. All analytic checkpoints pass, with relative errors from roughly `2e-8` through `1.7e-7`. Final J_frame relative error is about `1.04e-7` for cam2 and `1.05e-7` for cam4. `FIRST_DIVERGENCE=NONE`.

The finite-difference sweep is diagnostic only. Across eight cam2/cam4 points and sampled root/body columns, `eps=1e-2` is best (`9.18e-5` cam2, `1.59e-4` cam4 relative); smaller epsilon becomes noisier in float32, with no sign or column-order anomaly.

## Build and check

```bash
/usr/bin/cmake -S cpp/staged_lm -B build/staged_lm -DCMAKE_BUILD_TYPE=Release
/usr/bin/cmake --build build/staged_lm -j
./build/staged_lm/pose_reference_main \
  --fixture data_process/Vis/cpp_reference_dump/pose/frame_000513/cpp_fixture \
  --output data_process/Vis/cpp_reference_dump/pose/frame_000513/cpp_output
python src/script/validation/staged_lm/check_pose_cpp_jacobian.py \
  --reference data_process/Vis/cpp_reference_dump/pose/frame_000513/pose_reference.npz \
  --cpp_output data_process/Vis/cpp_reference_dump/pose/frame_000513/cpp_output
```

Timing and derivative-intermediate memory are recorded in `pose_jacobian_cpp_parity.json`. They are baseline observations only; no optimisation was performed.
