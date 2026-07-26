"""Compare staged-LM artifacts before and after the Pose module split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VOLATILE_KEY_PARTS = ("_ms", "timing", "wall", "fps", "output_dir")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json(value: Any, path: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_json(item, f"{path}.{key}")
            for key, item in value.items()
            if not any(part in key.lower() for part in VOLATILE_KEY_PARTS)
        }
    if isinstance(value, list):
        return [_stable_json(item, path) for item in value]
    return value


def _compare_npz(before: Path, after: Path) -> dict[str, Any]:
    with np.load(before, allow_pickle=False) as lhs, np.load(after, allow_pickle=False) as rhs:
        keys_equal = tuple(lhs.files) == tuple(rhs.files)
        mismatches = [
            key for key in lhs.files
            if key not in rhs.files
            or lhs[key].shape != rhs[key].shape
            or lhs[key].dtype != rhs[key].dtype
            or not np.array_equal(lhs[key], rhs[key])
        ]
    return {"keys_exact": keys_equal, "mismatches": mismatches, "exact": keys_equal and not mismatches}


def compare_trees(before_root: Path, after_root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for before_json in sorted(before_root.rglob("staged_stage2b_beta.json")):
        relative = before_json.relative_to(before_root)
        after_json = after_root / relative
        before_npz = before_json.with_suffix(".npz")
        after_npz = after_json.with_suffix(".npz")
        lhs = json.loads(before_json.read_text())
        rhs = json.loads(after_json.read_text())
        json_keys_exact = set(lhs) == set(rhs)
        stable_exact = _stable_json(lhs) == _stable_json(rhs)
        cases[str(relative.parent)] = {
            "json_key_set_exact": json_keys_exact,
            "json_stable_fields_exact": stable_exact,
            "npz": _compare_npz(before_npz, after_npz),
            "before_json_sha256": _hash_file(before_json),
            "after_json_sha256": _hash_file(after_json),
        }
    exact = bool(cases) and all(
        case["json_key_set_exact"] and case["json_stable_fields_exact"] and case["npz"]["exact"]
        for case in cases.values()
    )
    return {"exact": exact, "cases": cases}


def build_manifest(root: Path) -> dict[str, Any]:
    """Record compact artifact hashes and stable solver outcomes."""
    cases: dict[str, Any] = {}
    for json_path in sorted(root.rglob("staged_stage2b_beta.json")):
        relative = json_path.relative_to(root)
        payload = json.loads(json_path.read_text())
        pose = payload["stage2"]["pose_warmup"]["lm"]
        joint = payload["stage2"]["joint_refinement"]["lm"]
        cases[str(relative.parent)] = {
            "json_sha256": _hash_file(json_path),
            "npz_sha256": _hash_file(json_path.with_suffix(".npz")),
            "pose_status": pose["status"],
            "pose_accepted_steps": pose["accepted_steps"],
            "pose_rejected_trials": pose["rejected_trials"],
            "pose_objective": pose["obj"],
            "joint_status": joint["status"],
            "joint_objective": joint["obj"],
        }
    return {"root": str(root), "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_trees(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output.parent / "baseline_manifest.json").write_text(
        json.dumps(build_manifest(args.before), indent=2, sort_keys=True) + "\n"
    )
    (args.output.parent / "refactored_manifest.json").write_text(
        json.dumps(build_manifest(args.after), indent=2, sort_keys=True) + "\n"
    )
    print("POSE_MODULE_REFACTOR_PARITY_PASS" if report["exact"] else "POSE_MODULE_REFACTOR_PARITY_FAIL")
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
