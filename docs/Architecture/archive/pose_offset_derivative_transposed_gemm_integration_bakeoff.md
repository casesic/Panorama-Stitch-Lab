# Transposed pose-offset GEMM integration bake-off

O5 `optimized_native_stable_p_transposed` reuses the prior Candidate C implementation. Relative to O4 it changes only the derivative GEMM orientation from `Y=P*D` to the zero-copy expression `Yᵀ=DᵀPᵀ`; stable-P ownership, external output layout, value GEMV, LBS, LM and build policy remain unchanged.

Correctness and LM trajectory remain exact, but the balanced 20+20 full-pipeline measurement shows no system win: O4 median 193.408 ms and O5 median 194.053 ms. Integrated GEMM changes only from 66.827 to 66.651 ms, far below the micro-bake-off projection. Therefore O4 remains the preferred production-like policy; O5 is retained only as an explicit experimental comparison mode.
