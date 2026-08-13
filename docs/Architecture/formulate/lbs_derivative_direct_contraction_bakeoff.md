# LBS derivative direct-contraction bake-off

Frame 513 的实验保持 reference CPU 编译策略、FP32、55 个完整关节、原始
weight/transform 布局和单线程不变。默认路径继续使用 materialized kernel；direct
candidate 仅存在于 experimental benchmark。

当前实现先计算 `[N,16,Q]` 的 weighted transform derivative，再产生 transform
term、vertex term 和最终 `[N,3,Q]`。Direct 按 vertex → parameter → xyz → joint →
homogeneous component 顺序，直接计算前三个输出行并累计到 canonical `[N,3,Q]`；
benchmark 不保存 selected-transform 或两个 term，debug parity 模式才保存 term。

两者数学上均计算 `d(T h) = dT h + T dh`。Direct 仍遍历全部 55 个 joint，未跳过
零 weight，未融合 scale/ERP，也未使用 SIMD、native、blocking 或稀疏近似。它跳过
不会进入 xyz 输出的 homogeneous bottom row，并将每个 vertex/parameter 的估算操作
从 1811 降到 1671，约减少 7.73%。

六个真实 context 的 speedup 均约 1.25x；frame-weighted LBS 从 1992.16 ms 降到
1590.69 ms，节省 401.47 ms。显式 materialization 从每 context 的 `[N,25,Q]`
float 降到 `[N,3,Q]`，减少 88%。最大 current intermediate 是 PoseShape cam2 Q76
的 `[1023,16,76]`，约 4.745 MiB。

PoseShape base66 是 required production work：nested Pose 结果提供 `J_active`，并向
combined Q76 链提供 pose/FK derivative 的前 66 列。它与 combined76 存在未来去重
机会，但本轮未删除或共享。

正确性使用既有 2e-6 layer policy；local、debug terms、Python analytic、Pose
`J_active`、PoseShape `J_active/J_beta` 全部 PASS。

`LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF_READY=PASS`

`NEXT_OPTIMIZATION_EXPERIMENT=LBS_DERIVATIVE_SELECTED_TRANSFORM_GEMM_BAKEOFF`
