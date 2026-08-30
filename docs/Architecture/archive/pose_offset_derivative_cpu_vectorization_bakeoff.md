# Pose-offset derivative CPU vectorization bake-off

This experiment keeps the accepted FP32 Eigen GEMM expression and its maps,
storage, allocation, and single-thread policy unchanged. It compares three
separate Release builds: Eigen packets disabled (A), packets enabled with the
generic compiler ISA (B), and packets enabled with `-march=native` (C). The
default production-like scalar path remains unchanged.

On frame 513, the call-weighted pose-offset cost is 365.30 ms for A, 99.43 ms
for B, and 30.17 ms for C. Thus Eigen packet vectorization adds 3.67x over the
already accepted non-vectorized GEMM, while native ISA targeting adds another
3.29x. The projected fixed-trajectory pipeline is respectively 2907.87,
2641.99, and 2572.74 ms (0.344, 0.379, and 0.389 FPS).

The generic B build exposes Eigen SSE packets. The native C build exposes SSE,
AVX, AVX2, and AVX-512 packets on this host. Although every build retains
`-ffp-contract=off`, disassembly shows FMA in C because Eigen's native packet
implementation explicitly uses fused intrinsics; this is not scalar-expression
contraction by the compiler. All current formal numerical gates still pass,
without tolerance changes.

All buffers are at least 16-byte aligned, but their 32/64-byte alignment varies;
the unchanged unaligned `Eigen::Map` policy is therefore appropriate for this
experiment. For float packets, B has width 4; C's widest AVX-512 packet has
width 16. With C, `(M,K,Q)` tails are cam2 `(13,6,2)` and cam4 `(13,6,2)` modulo
16. No padding, aligned maps, persistent packing, or user-side temporary beyond
the canonical output was introduced.

With C, pose-offset derivative falls to 1.17% of projected pipeline time, while
the measured LBS derivative family becomes the clear top hotspot at 1877.90 ms
(72.99%). Therefore pose-offset is no longer the next P0.

`POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF_READY=PASS`

`NEXT_OPTIMIZATION_EXPERIMENT=LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF`
