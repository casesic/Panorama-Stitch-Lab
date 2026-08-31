"""Experimental serial persistent-process adapter for staged-LM bake-offs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections import deque
import select
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from lib_pano.fitting_io.densekp_runtime_export import PersonFrameInitialState
from .matched_person_tracks import MatchedPerson, OptimizedPersonState
from .per_person_cpp import projection_extent_ratio


REQUEST_MAGIC = 0x31514D4C
RESPONSE_MAGIC = 0x31524D4C
PROTOCOL_VERSION = 1
COMMAND_SOLVE = 1
COMMAND_SHUTDOWN = 2
HEADER = struct.Struct("<4I")


@dataclass(frozen=True)
class PersistentCppWorkerConfig:
    executable: Path
    static_fixture: Path
    timeout_seconds: float = 30.0
    debug_dump_dir: Path | None = None


class WorkerRequestError(RuntimeError):
    """Recoverable, well-framed request validation failure."""


class WorkerFatalError(RuntimeError):
    """Fatal process, timeout, transport, or protocol failure."""


class _Reader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def unpack(self, format_: str):
        parser = struct.Struct("<" + format_)
        if self.offset + parser.size > len(self.payload):
            raise RuntimeError("worker response truncated")
        value = parser.unpack_from(self.payload, self.offset)
        self.offset += parser.size
        return value[0] if len(value) == 1 else value

    def array(self, dtype, count: int) -> np.ndarray:
        dtype = np.dtype(dtype)
        size = dtype.itemsize * count
        if self.offset + size > len(self.payload):
            raise RuntimeError("worker response array truncated")
        value = np.frombuffer(
            self.payload, dtype=dtype, count=count, offset=self.offset,
        ).copy()
        self.offset += size
        return value

    def string(self) -> str:
        size = self.unpack("I")
        if self.offset + size > len(self.payload):
            raise RuntimeError("worker response string truncated")
        value = self.payload[self.offset:self.offset + size].decode("utf-8")
        self.offset += size
        return value


class PersistentCppFittingWorker:
    """One child process, one immutable static cache, serial T=1 requests."""

    def __init__(self, config: PersistentCppWorkerConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.request_count = 0
        self.profile_rows: list[dict[str, Any]] = []
        self.start_count = 0
        self.restart_count = 0
        self.static_model_load_count = 0
        self.worker_startup_ms = 0.0
        self.static_model_load_ms = 0.0
        self._failed = False
        self._closed = False
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=200)

    def start(self) -> "PersistentCppFittingWorker":
        if self.process is not None:
            if self.process.poll() is not None:
                self._failed = True
                raise WorkerFatalError("persistent C++ worker exited unexpectedly")
            return self
        if self._failed:
            raise WorkerFatalError(
                "persistent C++ worker is marked dead; automatic restart is disabled")
        if self._closed:
            raise WorkerFatalError("persistent C++ worker is closed")
        executable = self.config.executable.resolve()
        fixture = self.config.static_fixture.resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"persistent C++ worker binary missing: {executable}")
        if not fixture.is_dir():
            raise FileNotFoundError(f"persistent C++ static fixture missing: {fixture}")
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                [str(executable), "--static-fixture", str(fixture)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
            self.start_count += 1
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, name="staged-lm-worker-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            ready = _Reader(self._receive_response())
            self.static_model_load_count = ready.unpack("I")
            protocol_version = ready.unpack("I")
            evaluator = ready.string()
            frame_dimension = ready.unpack("I")
            beta_dimension = ready.unpack("I")
            densekp_dimension = ready.unpack("I")
            static_fixture = Path(ready.string()).resolve()
            self.worker_startup_ms = ready.unpack("d")
            self.static_model_load_ms = ready.unpack("d")
            if ready.offset != len(ready.payload):
                raise WorkerFatalError("worker READY payload has trailing bytes")
            actual = (
                protocol_version, evaluator, frame_dimension,
                beta_dimension, densekp_dimension, static_fixture,
            )
            expected = (PROTOCOL_VERSION, "cpu", 75, 10, 138, fixture)
            if actual != expected or self.static_model_load_count != 1:
                raise WorkerFatalError(
                    f"worker READY contract mismatch: expected={expected}, "
                    f"actual={actual}, static_loads={self.static_model_load_count}")
        except Exception as error:
            self._mark_failed()
            if isinstance(error, (FileNotFoundError, WorkerFatalError)):
                raise
            raise WorkerFatalError(f"persistent C++ worker startup failed: {error}") from error
        self.worker_startup_ms = max(
            self.worker_startup_ms, (time.perf_counter() - started) * 1000.0)
        return self

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in iter(process.stderr.readline, b""):
            self._stderr_lines.append(line.decode(errors="replace").rstrip())

    @property
    def recent_stderr(self) -> tuple[str, ...]:
        return tuple(self._stderr_lines)

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def _request_arrays(person: MatchedPerson, initial: PersonFrameInitialState):
        initial.validate()
        ids = np.ascontiguousarray(person.point_ids[person.valid], dtype="<i4")
        pairs = np.ascontiguousarray(person.pairs[person.valid], dtype="<f4")
        if not 0 < len(ids) <= 138:
            raise ValueError("worker request requires 1..138 valid DenseKP points")
        return len(ids), [
            np.asarray(initial.frame_parameters, dtype="<f4"),
            np.asarray(initial.beta, dtype="<f4"),
            np.asarray(initial.cam2_left_hand_pose, dtype="<f4"),
            np.asarray(initial.cam2_right_hand_pose, dtype="<f4"),
            np.asarray(initial.cam4_left_hand_pose, dtype="<f4"),
            np.asarray(initial.cam4_right_hand_pose, dtype="<f4"),
            np.asarray([initial.cam2_scale, initial.cam4_scale], dtype="<f4"),
            ids, np.ascontiguousarray(pairs[:, :2], dtype="<f4"),
            np.ascontiguousarray(pairs[:, 2:], dtype="<f4"),
        ]
    @staticmethod
    def _serialize_request(point_count: int, arrays) -> bytes:
        return struct.pack("<I", point_count) + b"".join(
            value.tobytes() for value in arrays)

    @classmethod
    def encode_request(cls, person: MatchedPerson,
                       initial: PersonFrameInitialState) -> bytes:
        point_count, arrays = cls._request_arrays(person, initial)
        return cls._serialize_request(point_count, arrays)

    def _read_exact(self, size: int) -> bytes:
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + self.config.timeout_seconds
        chunks = []
        remaining = size
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or not select.select(
                [self.process.stdout], [], [], max(0.0, timeout),
            )[0]:
                raise TimeoutError("persistent C++ worker response timeout")
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                raise RuntimeError(
                    "persistent C++ worker closed stdout; stderr="
                    + " | ".join(self.recent_stderr[-10:]))
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _receive_response(self) -> bytes:
        magic, version, status, size = HEADER.unpack(self._read_exact(HEADER.size))
        if magic != RESPONSE_MAGIC or version != PROTOCOL_VERSION:
            raise WorkerFatalError(
                f"persistent C++ worker response header mismatch: "
                f"magic={magic:#x} version={version}")
        response = self._read_exact(size)
        if status:
            raise WorkerRequestError(_Reader(response).string())
        return response

    def _exchange(self, command: int, payload: bytes) -> bytes:
        self.start()
        assert self.process is not None and self.process.stdin is not None
        if self.process.poll() is not None:
            raise RuntimeError("persistent C++ worker is not running")
        try:
            self.process.stdin.write(HEADER.pack(
                REQUEST_MAGIC, PROTOCOL_VERSION, command, len(payload)))
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            return self._receive_response()
        except WorkerRequestError:
            raise
        except Exception as error:
            self._mark_failed()
            if isinstance(error, WorkerFatalError):
                raise
            raise WorkerFatalError(
                f"persistent C++ worker transport failed: {error}") from error

    def fit(self, person: MatchedPerson,
            initial: PersonFrameInitialState) -> OptimizedPersonState:
        outer_started = time.perf_counter()
        build_started = time.perf_counter()
        point_count, arrays = self._request_arrays(person, initial)
        build_ms = (time.perf_counter() - build_started) * 1000.0
        serialization_started = time.perf_counter()
        request = self._serialize_request(point_count, arrays)
        serialization_ms = (time.perf_counter() - serialization_started) * 1000.0
        if self.config.debug_dump_dir is not None:
            self.config.debug_dump_dir.mkdir(parents=True, exist_ok=True)
            (self.config.debug_dump_dir / f"request_{self.request_count:06d}.bin").write_bytes(
                request)
        ipc_started = time.perf_counter()
        response = self._exchange(COMMAND_SOLVE, request)
        ipc_ms = (time.perf_counter() - ipc_started) * 1000.0
        if self.config.debug_dump_dir is not None:
            (self.config.debug_dump_dir / f"response_{self.request_count:06d}.bin").write_bytes(
                response)
        parse_started = time.perf_counter()
        reader = _Reader(response)
        static_load_count = reader.unpack("I")
        timing_names = (
            "cpp_decode_ms", "cpp_dynamic_setup_ms", "runtime_preparation_ms",
            "stage1_ms", "pose_ms", "pose_shape_ms", "optimized_evaluation_ms",
            "cpp_lm_ms", "cpp_request_before_encode_ms",
        )
        timing = {name: reader.unpack("d") for name in timing_names}
        initial_objective = reader.unpack("d")
        final_objective = reader.unpack("d")
        runs = []
        for _ in range(3):
            runs.append({
                "accepted": reader.unpack("i"),
                "rejected": reader.unpack("i"),
                "trials": reader.unpack("i"),
                "status": reader.string(),
                "convergence": reader.string(),
            })
        frame = reader.array("<f4", 75)
        beta = reader.array("<f4", 10)
        point_count = reader.unpack("I")
        all_pairs = reader.array("<f4", point_count * 4).reshape(point_count, 4)
        output_valid = reader.array("u1", point_count)
        timing["cpp_response_encode_ms"] = reader.unpack("d")
        if reader.offset != len(response):
            raise RuntimeError("persistent C++ worker response has trailing bytes")
        parse_ms = (time.perf_counter() - parse_started) * 1000.0
        state_started = time.perf_counter()
        state = OptimizedPersonState(
            frame_parameters=frame, beta=beta,
            optimized_pairs=np.ascontiguousarray(all_pairs[person.point_ids]),
            point_ids=person.point_ids.copy(), valid=person.valid.copy(),
            success=bool(np.isfinite(frame).all() and np.isfinite(beta).all()
                         and np.isfinite(all_pairs).all()),
            status=runs[2]["status"], final_objective=final_objective,
            cpp_preparation_success=True,
            cpp_prepare_ms=timing["runtime_preparation_ms"],
            initial_objective=initial_objective,
            lm_iterations=sum(run["trials"] for run in runs),
            cpp_lm_ms=timing["cpp_lm_ms"], warm_state_exact=None,
            projection_extent_ratio=projection_extent_ratio(
                np.ascontiguousarray(all_pairs[person.point_ids]),
                person.pairs, person.valid),
        )
        state_ms = (time.perf_counter() - state_started) * 1000.0
        outer_ms = (time.perf_counter() - outer_started) * 1000.0
        self.profile_rows.append({
            "request_build_ms": build_ms, "ipc_ms": ipc_ms,
            "request_serialization_ms": serialization_ms,
            "python_response_parse_ms": parse_ms,
            "state_construction_ms": state_ms, "worker_outer_ms": outer_ms,
            "static_load_count": float(static_load_count), **timing,
            "stage1_accepted": float(runs[0]["accepted"]),
            "stage1_rejected": float(runs[0]["rejected"]),
            "pose_accepted": float(runs[1]["accepted"]),
            "pose_rejected": float(runs[1]["rejected"]),
            "pose_shape_accepted": float(runs[2]["accepted"]),
            "pose_shape_rejected": float(runs[2]["rejected"]),
            "output_valid_count": float(output_valid.sum()),
            "stage1_trials": runs[0]["trials"],
            "pose_trials": runs[1]["trials"],
            "pose_shape_trials": runs[2]["trials"],
            "stage1_status": runs[0]["status"],
            "pose_status": runs[1]["status"],
            "pose_shape_status": runs[2]["status"],
            "stage1_convergence": runs[0]["convergence"],
            "pose_convergence": runs[1]["convergence"],
            "pose_shape_convergence": runs[2]["convergence"],
        })
        self.request_count += 1
        return state

    def send_malformed_request(self) -> str:
        try:
            self._exchange(COMMAND_SOLVE, struct.pack("<I", 0))
        except WorkerRequestError as error:
            return str(error)
        raise AssertionError("malformed request unexpectedly succeeded")

    def _mark_failed(self) -> None:
        self._failed = True
        process = self.process
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def close(self) -> None:
        process = self.process
        if process is None:
            self._closed = True
            return
        if process.poll() is None and not self._failed:
            try:
                self._exchange(COMMAND_SHUTDOWN, b"")
            except WorkerFatalError:
                pass
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=self.config.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            self._failed = True
        self.process = None
        self._closed = True
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

    def __enter__(self) -> "PersistentCppFittingWorker":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class PersistentCppPerPersonFitter:
    """Experimental production-shape callable; subprocess backend is untouched."""

    def __init__(self, worker: PersistentCppFittingWorker):
        self.worker = worker
        self._frozen_by_track: dict[int, PersonFrameInitialState] = {}
        self.call_count = 0

    def __call__(self, person: MatchedPerson, initialization: object,
                 track_id: int) -> OptimizedPersonState:
        if isinstance(initialization, PersonFrameInitialState):
            self._frozen_by_track[track_id] = initialization
            initial = initialization
            warm_exact = None
        elif isinstance(initialization, OptimizedPersonState):
            frozen = self._frozen_by_track.get(track_id)
            if frozen is None:
                raise RuntimeError("warm start has no frozen hand/scale state for track")
            initial = PersonFrameInitialState(
                frame_parameters=np.asarray(initialization.frame_parameters, np.float32),
                beta=np.asarray(initialization.beta, np.float32),
                cam2_left_hand_pose=frozen.cam2_left_hand_pose,
                cam2_right_hand_pose=frozen.cam2_right_hand_pose,
                cam4_left_hand_pose=frozen.cam4_left_hand_pose,
                cam4_right_hand_pose=frozen.cam4_right_hand_pose,
                cam2_scale=frozen.cam2_scale, cam4_scale=frozen.cam4_scale,
            )
            initial.validate()
            warm_exact = bool(
                np.array_equal(initial.frame_parameters, initialization.frame_parameters)
                and np.array_equal(initial.beta, initialization.beta))
        else:
            raise TypeError("unsupported persistent worker initialization")
        result = self.worker.fit(person, initial)
        self.call_count += 1
        return replace(result, warm_state_exact=warm_exact)

    def close(self) -> None:
        self.worker.close()
