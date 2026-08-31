# Persistent C++ fitting worker

## Purpose and execution selection

Production fitting has two explicit execution backends:

```text
subprocess
  materialize runtime fixture -> reference binary -> parse file outputs

persistent_worker
  binary request -> resident CPU worker -> binary response
```

`subprocess` remains the default and the detailed correctness/debug oracle.
`persistent_worker` is selected explicitly with
`--cpp-execution-backend persistent_worker` and an explicit worker binary.
There is no automatic selection, retry, or fallback between them.

## Lifecycle and independence

The sequence entry starts one worker before processing its first frame and
closes it in a `finally` block at sequence exit. The worker loads the immutable
model bundle once, then handles person requests serially:

```text
sequence start
  -> READY / static contract validation
  -> Track 0 request: independent T=1 problem and beta
  -> Track 1 request: independent T=1 problem and beta
  -> ...
  -> SHUTDOWN / ACK
sequence end
```

The worker is not multi-frame shared-beta fitting and is not multi-person
batching. Python `TrackState` remains the sole owner of previous-LM state.
Every request supplies its complete initial frame state, physical beta, frozen
hand/scale state, DenseKP IDs, and camera targets. Lambda, accepted-step state,
solver state, frame parameters, and beta never persist between requests.

## Static and dynamic data

Startup loads the DenseKP regressor/sparse operator, base joints, beta joint
basis, parents, camera posedirs, LBS weights, shape directions, and immutable
solver configuration. The READY response identifies the canonical static
fixture and verifies CPU evaluation, protocol version 1, frame dimension 75,
beta dimension 10, and DenseKP dimension 138.

Requests contain only dynamic arrays. The normal path creates no runtime
fixture directory and writes no per-request fitting files. An explicit debug
dump directory can record binary requests/responses for reference replay; it
is off by default.

## Protocol and failures

stdin/stdout carry a little-endian length-prefixed binary protocol:

```text
uint32 magic
uint32 protocol_version
uint32 command
uint32 payload_size
payload bytes
```

stdout is protocol-only. A background thread drains stderr so diagnostics
cannot fill the child pipe. Well-framed request validation errors, such as an
invalid point count, are recoverable. Timeout, EOF, broken pipe, child death,
bad magic/version, and READY/static-model mismatch are fatal: the current call
raises, the worker is marked dead, and production does not retry, restart, or
fall back silently. Normal close sends SHUTDOWN, waits for ACK, and reaps the
child; close is idempotent.

## Correctness and performance acceptance

Reference replay for frames 704, 705, 711 and both frame-2214 tracks is bitwise
exact for final frame parameters, physical beta, optimized DenseKP pairs, and
valid IDs, with exact solver statuses and trial/acceptance counts. Request
order, A-B-A beta isolation, malformed-request recovery, and repeated-request
tests also pass.

With real HMR2 cold start, TensorRT DenseKP, CPU staged-LM and K=3, the final
serial Run1-3 measurement was 87.445 ms/frame (11.436 FPS) for frames 705-711.
The 2214-2221 mixed workload measured 102.175 ms/frame (9.787 FPS); that mixed
result is not generalized as a 10 FPS pass. Worker startup and static model
load are reported separately from steady frame timing.
