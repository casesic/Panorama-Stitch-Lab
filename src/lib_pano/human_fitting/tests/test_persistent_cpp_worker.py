from pathlib import Path
from unittest.mock import patch

import numpy as np

from lib_pano.fitting_io.densekp_runtime_export import PersonFrameInitialState
from lib_pano.human_fitting.matched_person_tracks import MatchedPerson
from lib_pano.human_fitting.persistent_cpp_worker import (
    PersistentCppFittingWorker,
    PersistentCppWorkerConfig,
    _Reader,
    WorkerFatalError,
)
from lib_pano.human_fitting.per_person_cpp import (
    CppPerPersonFitter,
    CppPersonFitterConfig,
    create_cpp_per_person_fitter,
)
import pytest


def _initial(beta=0.0):
    return PersonFrameInitialState(
        frame_parameters=np.zeros(75, np.float32),
        beta=np.full(10, beta, np.float32),
        cam2_left_hand_pose=np.zeros(45, np.float32),
        cam2_right_hand_pose=np.zeros(45, np.float32),
        cam4_left_hand_pose=np.zeros(45, np.float32),
        cam4_right_hand_pose=np.zeros(45, np.float32),
        cam2_scale=1.0, cam4_scale=2.0,
    )


def test_fixture_free_request_encoding_contains_only_dynamic_arrays():
    person = MatchedPerson(
        person_id=0, bbox_cam2=np.zeros(4), bbox_cam4=np.zeros(4),
        pairs=np.arange(16, dtype=np.float32).reshape(4, 4),
        point_ids=np.array([2, 5, 7, 9], np.int32),
        valid=np.array([True, False, True, False]),
    )
    payload = PersistentCppFittingWorker.encode_request(person, _initial(0.25))
    reader = _Reader(payload)
    assert reader.unpack("I") == 2
    np.testing.assert_array_equal(reader.array("<f4", 75), 0.0)
    np.testing.assert_array_equal(reader.array("<f4", 10), 0.25)
    reader.array("<f4", 45 * 4)
    np.testing.assert_array_equal(reader.array("<f4", 2), [1.0, 2.0])
    np.testing.assert_array_equal(reader.array("<i4", 2), [2, 7])
    np.testing.assert_array_equal(
        reader.array("<f4", 4).reshape(2, 2), person.pairs[[0, 2], :2])
    np.testing.assert_array_equal(
        reader.array("<f4", 4).reshape(2, 2), person.pairs[[0, 2], 2:])
    assert reader.offset == len(payload)


def test_worker_construction_is_lazy_and_does_not_fallback():
    worker = PersistentCppFittingWorker(PersistentCppWorkerConfig(
        executable=Path("missing-worker"), static_fixture=Path("missing-fixture")))
    assert worker.process is None
    assert not worker.alive


def _fake_worker(tmp_path: Path, behavior: str = "normal") -> tuple[Path, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    script = tmp_path / "worker.py"
    script.write_text(f'''#!/usr/bin/env python3
import pathlib, struct, sys, time
H=struct.Struct("<4I")
def text(x):
 b=x.encode(); return struct.pack("<I",len(b))+b
fixture=pathlib.Path(sys.argv[sys.argv.index("--static-fixture")+1]).resolve()
version={{"normal":1,"bad_version":99,"bad_fixture":1,"exit":1,"timeout":1}}["{behavior}"]
identity=pathlib.Path("/wrong/static/model") if "{behavior}"=="bad_fixture" else fixture
p=struct.pack("<II",1,version)+text("cpu")+struct.pack("<III",75,10,138)+text(str(identity))+struct.pack("<dd",0.1,0.1)
sys.stdout.buffer.write(H.pack(0x31524d4c,1,0,len(p))+p);sys.stdout.buffer.flush()
if "{behavior}"=="exit": sys.exit(0)
while True:
 h=sys.stdin.buffer.read(H.size)
 if not h: break
 magic,v,command,n=H.unpack(h);sys.stdin.buffer.read(n)
 if "{behavior}"=="timeout": time.sleep(10); continue
 sys.stdout.buffer.write(H.pack(0x31524d4c,1,0,0));sys.stdout.buffer.flush()
 if command==2: break
''')
    script.chmod(0o755)
    return script, fixture


def test_ready_handshake_and_close_are_idempotent(tmp_path):
    executable, fixture = _fake_worker(tmp_path)
    worker = PersistentCppFittingWorker(PersistentCppWorkerConfig(
        executable, fixture, timeout_seconds=1.0)).start()
    assert worker.start_count == 1
    assert worker.static_model_load_count == 1
    assert worker.alive
    worker.close()
    worker.close()
    assert not worker.alive


def test_protocol_version_mismatch_is_fatal(tmp_path):
    executable, fixture = _fake_worker(tmp_path, "bad_version")
    worker = PersistentCppFittingWorker(PersistentCppWorkerConfig(
        executable, fixture, timeout_seconds=1.0))
    with pytest.raises(WorkerFatalError, match="READY contract mismatch"):
        worker.start()
    worker.close()


def test_static_model_identity_mismatch_is_fatal(tmp_path):
    executable, fixture = _fake_worker(tmp_path, "bad_fixture")
    worker = PersistentCppFittingWorker(PersistentCppWorkerConfig(
        executable, fixture, timeout_seconds=1.0))
    with pytest.raises(WorkerFatalError, match="READY contract mismatch"):
        worker.start()
    worker.close()


def test_unexpected_worker_death_is_explicit_and_never_restarts(tmp_path):
    executable, fixture = _fake_worker(tmp_path, "exit")
    worker = PersistentCppFittingWorker(PersistentCppWorkerConfig(
        executable, fixture, timeout_seconds=1.0)).start()
    worker.process.wait(timeout=1.0)
    with pytest.raises(WorkerFatalError):
        worker._exchange(1, b"")
    assert worker.start_count == 1
    assert worker.restart_count == 0
    worker.close()


def test_worker_timeout_marks_process_dead(tmp_path):
    executable, fixture = _fake_worker(tmp_path, "timeout")
    worker = PersistentCppFittingWorker(PersistentCppWorkerConfig(
        executable, fixture, timeout_seconds=0.05)).start()
    with pytest.raises(WorkerFatalError, match="timeout"):
        worker._exchange(1, b"")
    assert not worker.alive
    worker.close()


def test_execution_backend_selection_retains_subprocess(tmp_path):
    config = CppPersonFitterConfig(
        tmp_path, tmp_path / "reference", tmp_path / "output")
    assert isinstance(create_cpp_per_person_fitter(config), CppPerPersonFitter)
    worker_config = CppPersonFitterConfig(
        tmp_path, tmp_path / "reference", tmp_path / "output",
        execution_backend="persistent_worker",
        worker_executable=tmp_path / "worker")
    with patch(
        "lib_pano.human_fitting.persistent_cpp_worker.PersistentCppFittingWorker.start",
        autospec=True,
    ) as start:
        start.side_effect = lambda value: value
        selected = create_cpp_per_person_fitter(worker_config)
    assert selected.worker.config.static_fixture == tmp_path
