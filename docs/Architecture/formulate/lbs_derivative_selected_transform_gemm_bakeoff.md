# LBS selected-transform GEMM bake-off

本实验只将 materialized LBS 中的 55-joint scalar reduction 替换为 Eigen GEMM，
其余 `(dT)h`、`T(dh)`、term buffers 和 `d_skinned` downstream 完全共用 current
实现。默认 scalar LBS 与上一轮 direct candidate 均未改变。

物理布局天然支持 zero-copy：weights 是 row-major `[N,55]`；dA 是
`[55,16,Q]` 且 Q 连续，可视为 `[55,16Q]`；dT 是 `[N,16,Q]` 且 Q 连续，
可视为 `[N,16Q]`。Candidate 使用 `DT.noalias() = W * DA`，没有显式
pack、unpack、transpose 或额外 output buffer，仍完整 materialize dT。

Reference policy 下六个 context 的 selected-transform speedup 为 5.46–6.04x，
完整 materialized LBS speedup 为 5.32–5.41x。Frame-weighted selected transform
与 full LBS 的详细数值见 benchmark summary。GEMM 后 selected transform 自身仍
明显大于 aggregate downstream，因此 residual bottleneck 尚未转移到 `(dT)h`；
下一轮应先隔离测试 Eigen vectorization/native，而不是直接融合 Direct。

所有 dT、term、d_skinned、Python analytic 与 downstream Jacobian 使用既有
tolerance 通过。Vertex term 与 current bitwise exact。

`LBS_DERIVATIVE_SELECTED_TRANSFORM_GEMM_BAKEOFF_READY=PASS`

`NEXT_OPTIMIZATION_EXPERIMENT=LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF`
