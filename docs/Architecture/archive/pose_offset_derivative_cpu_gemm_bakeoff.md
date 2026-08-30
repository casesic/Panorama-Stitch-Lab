# Pose-offset derivative CPU GEMM bakeoff

本实验保持默认 scalar pipeline 不变，只比较真实 scalar contraction 与 experimental Eigen GEMM。编译策略继续使用 `-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off`，没有 `-march=native`、fast-math、LTO 或多线程；`Eigen::nbThreads()==1`。

真实 scalar 数学为 `O[row,q] = Σ_f P[f,row] D[f,q]`。`selected_posedirs` 物理顺序是 `[feature,N,xyz]`，可零拷贝映射为 column-major `[3N,486]`；`d_pose_feature` 是 parameter-contiguous row-major `[486,66]`；canonical output 是 row-major `[3N,66]`。候选使用 `result.noalias() = posedirs * derivative`，无需显式 input pack 或 output unpack，只分配与 scalar 相同的 canonical output。

四个真实 context 均通过现有 feature tolerance。candidate 对 scalar/Python 的最大绝对误差为 `2.794e-9`，relative 约 `6.7e-8`。20 次 interleaved 测量中，cam2/cam4 的 scalar 中位数约为 `136.3/122.9 ms`，GEMM total 中位数约为 `23.6/21.4 ms`；四 context 的 call-weighted frame cost 从约 `2075.0 ms` 降至 `360.4 ms`，约 `5.76x`。

固定 trajectory 投影得到 pipeline `2902.9 ms`、约 `0.344 FPS`。新主热点变为 LBS derivative；但按决策规则，本轮只证明 GEMM expression 有效，默认 kernel 不替换。下一轮唯一实验为 `POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF`。
