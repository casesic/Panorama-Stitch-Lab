#!/usr/bin/env python3
"""Summarise the staged-LM fixed surface-sampling bake-off."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib_pano.human3d.crop_upright_coord import resolve_dataset_rotate_rules
from lib_pano.human3d.smplx_region_sampling import assign_smplx_vertex_regions
from lib_pano.human_fitting.projection import project_cam2
from lib_pano.human_fitting.smplx_sequence import (
    forward_smplx_tensor_batch, get_smplx_batch,
)
from lib_pano.human_fitting.staged_lm.data_loading import load_frame_observations
from lib_pano.human_fitting.staged_lm.geometry.so3 import axis_angle_to_matrix

OUT = ROOT / "data_process/Vis/sam3d_surface_point_sampling_bakeoff"
RUNS = OUT / "runs"
DATASET = ROOT / "data_process/Output/20260311_riding"
SEGMENTS = ((513, 518), (704, 711), (1217, 1225))
POLICIES = ("S1024", "S50", "S100", "S138", "S200")


def load_seq(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def vertices(model, seq) -> np.ndarray:
    with torch.no_grad():
        value, _, _ = forward_smplx_tensor_batch(
            model, seq.orient, seq.body_pose, seq.left_hand_pose,
            seq.right_hand_pose, seq.beta, seq.transl,
            seq.scale.expand(seq.num_batch),
        )
    return value.cpu().numpy()


def geodesic(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    ra, rb = axis_angle_to_matrix(a.reshape(-1, 3)), axis_angle_to_matrix(b.reshape(-1, 3))
    cosine = ((torch.diagonal(ra.transpose(-1, -2) @ rb, dim1=-2, dim2=-1).sum(-1) - 1) / 2)
    angle = torch.acos(cosine.clamp(-1, 1))
    angle = torch.where(torch.linalg.norm(a.reshape(-1, 3) - b.reshape(-1, 3), dim=1) == 0,
                        torch.zeros_like(angle), angle)
    return angle.cpu().numpy()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    model = get_smplx_batch("neutral").eval()
    regions = assign_smplx_vertex_regions(model.lbs_weights.cpu().numpy())
    region_index = np.empty(10475, np.int32)
    region_names = list(regions)
    for index, name in enumerate(region_names):
        region_index[regions[name]] = index
    rotate_rules = resolve_dataset_rotate_rules(
        ROOT / "configs/rotate_upright_rules.json",
        DATASET / "image/equirect_crop_rotate",
    )
    sampling_rows, quality_rows, stability_rows, parameter_rows = [], [], [], []
    aggregate_errors: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    aggregate_parameters: dict[tuple[str, str], list[float]] = defaultdict(list)
    visual_payload = None

    for start, end in SEGMENTS:
        segment = f"{start}_{end}"
        frames = list(range(start, end + 1))
        observations = [load_frame_observations(
            frame, DATASET / "sam3dbody/smplx_dense",
            DATASET / "image/equirect_crop_rotate", 5760, 2880, rotate_rules,
        ) for frame in frames]
        baseline_seq = [load_seq(RUNS / "S1024" / camera / f"seg_{segment}_seq.pkl")
                        for camera in ("cam2", "cam4")]
        baseline_vertices = [vertices(model, seq) for seq in baseline_seq]

        for policy in POLICIES:
            run_root = RUNS / policy
            seqs = [load_seq(run_root / camera / f"seg_{segment}_seq.pkl")
                    for camera in ("cam2", "cam4")]
            verts = [vertices(model, seq) for seq in seqs]
            with (run_root / segment / "staged_stage2b_beta.json").open() as file:
                result = json.load(file)
            stages = [result["stage1"]["lm"], result["stage2"]["pose_warmup"]["lm"],
                      result["stage2"]["pose_shape_refinement"]["lm"]]
            terms = stages[0]["aggregation_spec"]["terms"]
            counts = np.asarray([term["n_valid"] for term in terms.values()])
            nmax = int(counts.max())
            requested = 1024 if policy == "S1024" else int(policy[1:])
            sampling_rows.append({
                "policy": policy, "segment": segment, "frames": len(frames),
                "requested_per_group": requested, "groups": len(counts),
                "valid_min": int(counts.min()), "valid_mean": float(counts.mean()),
                "valid_max": nmax, "batch_nmax": nmax,
                "padding_ratio": float(1 - counts.sum() / (len(counts) * nmax)),
            })
            profiles = [stage.get("profile_timing", {}) for stage in stages]
            stability_rows.append({
                "policy": policy, "segment": segment, "frames": len(frames),
                "fitting_wall_ms": result["pipeline_timing"]["fitting_total_ms"],
                "lm_ms": sum(stage.get("wall_time_ms", 0) for stage in stages),
                "ms_per_frame": result["pipeline_timing"]["fitting_total_ms"] / len(frames),
                "fps": 1000 * len(frames) / result["pipeline_timing"]["fitting_total_ms"],
                "normal_equations_ms": sum(p.get("normal_equations_ms", 0) for p in profiles),
                "objective_ms": sum(p.get("true_objective_ms", 0) for p in profiles),
                "linear_solve_ms": sum(p.get("linear_solve_ms", 0) for p in profiles),
                "linearization_count": sum(stage["linearization_count"] for stage in stages),
                "solve_attempt_count": sum(stage["solve_attempt_count"] for stage in stages),
                "accepted_steps": sum(stage["accepted_steps"] for stage in stages),
                "rejected_trials": sum(stage["rejected_trials"] for stage in stages),
                "final_objective": stages[-1]["obj"], "status": stages[-1]["status"],
            })

            all_errors, by_region = [], defaultdict(list)
            for camera_index, camera in enumerate(("cam2", "cam4")):
                projected = project_cam2(torch.from_numpy(verts[camera_index]), 5760, 2880).numpy()
                for position, obs_pair in enumerate(observations):
                    obs = obs_pair[camera_index]
                    valid = obs.valid
                    error = np.linalg.norm(projected[position, valid] - obs.points_2d[valid], axis=1)
                    all_errors.append(error)
                    valid_regions = region_index[valid]
                    for region_id, name in enumerate(region_names):
                        by_region[name].append(error[valid_regions == region_id])
                if visual_payload is None and policy == "S1024" and segment == "704_711" and camera == "cam2":
                    visual_payload = (frames, observations, {})
                if visual_payload is not None and segment == "704_711" and camera == "cam2":
                    visual_payload[2][policy] = projected
            error = np.concatenate(all_errors)
            quality_rows.append({
                "policy": policy, "segment": segment, "region": "all", "count": error.size,
                "rms_px": float(np.sqrt(np.mean(error ** 2))), "mean_px": float(error.mean()),
                "p95_px": float(np.percentile(error, 95)), "max_px": float(error.max()),
            })
            aggregate_errors[(policy, "all")].append(error)
            for name in region_names:
                values = np.concatenate(by_region[name])
                quality_rows.append({
                    "policy": policy, "segment": segment, "region": name, "count": values.size,
                    "rms_px": float(np.sqrt(np.mean(values ** 2))), "mean_px": float(values.mean()),
                    "p95_px": float(np.percentile(values, 95)), "max_px": float(values.max()),
                })
                aggregate_errors[(policy, name)].append(values)

            orient = np.concatenate([geodesic(seqs[c].orient, baseline_seq[c].orient) for c in range(2)])
            translation = np.concatenate([np.linalg.norm(
                (seqs[c].transl - baseline_seq[c].transl).numpy(), axis=1) for c in range(2)])
            pose = geodesic(seqs[0].body_pose, baseline_seq[0].body_pose)
            beta = np.linalg.norm((seqs[0].beta - baseline_seq[0].beta).numpy())
            mpve = np.concatenate([np.linalg.norm(verts[c] - baseline_vertices[c], axis=2).reshape(-1)
                                   for c in range(2)])
            parameter_rows.append({
                "policy": policy, "segment": segment,
                "global_orient_deg_mean": float(np.degrees(orient).mean()),
                "translation_l2_mean": float(translation.mean()),
                "body_pose_joint_deg_mean": float(np.degrees(pose).mean()),
                "beta_l2": float(beta), "vertex_mpve": float(mpve.mean()),
            })

    aggregate_quality = []
    for (policy, region), chunks in aggregate_errors.items():
        error = np.concatenate(chunks)
        aggregate_quality.append({
            "policy": policy, "segment": "ALL", "region": region, "count": error.size,
            "rms_px": float(np.sqrt(np.mean(error ** 2))), "mean_px": float(error.mean()),
            "p95_px": float(np.percentile(error, 95)), "max_px": float(error.max()),
        })
    quality_rows.extend(aggregate_quality)
    write_csv(OUT / "sampling_summary.csv", sampling_rows)
    write_csv(OUT / "holdout_quality.csv", quality_rows)
    write_csv(OUT / "lm_stability.csv", stability_rows)
    write_csv(OUT / "parameter_deltas_vs_s1024.csv", parameter_rows)

    speed_quality = []
    for policy in POLICIES:
        runs = [row for row in stability_rows if row["policy"] == policy]
        quality = next(row for row in aggregate_quality if row["policy"] == policy and row["region"] == "all")
        params = [row for row in parameter_rows if row["policy"] == policy]
        total_frames = sum(row["frames"] for row in runs)
        total_wall = sum(row["fitting_wall_ms"] for row in runs)
        speed_quality.append({
            "policy": policy, "frames": total_frames,
            "fitting_ms_per_frame": total_wall / total_frames,
            "fitting_fps": 1000 * total_frames / total_wall,
            "full_holdout_rms_px": quality["rms_px"], "full_holdout_p95_px": quality["p95_px"],
            "orient_deg_mean": float(np.mean([p["global_orient_deg_mean"] for p in params])),
            "translation_l2_mean": float(np.mean([p["translation_l2_mean"] for p in params])),
            "pose_joint_deg_mean": float(np.mean([p["body_pose_joint_deg_mean"] for p in params])),
            "beta_l2_mean": float(np.mean([p["beta_l2"] for p in params])),
            "vertex_mpve_mean": float(np.mean([p["vertex_mpve"] for p in params])),
        })
    write_csv(OUT / "speed_quality.csv", speed_quality)
    with (OUT / "benchmark_summary.json").open("w") as file:
        json.dump({"segments": [f"{a}_{b}" for a, b in SEGMENTS], "speed_quality": speed_quality}, file, indent=2)

    # One representative full-ERP overlay plus anatomy crops for every policy.
    frames, observations, projections = visual_payload
    position, frame = frames.index(708), 708
    image = cv2.imread(str(DATASET / f"image/equirect_rotate/cam2/{frame:05d}.jpg"))
    obs = observations[position][0]
    visual_dir = OUT / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for policy, projected in projections.items():
        canvas = image.copy()
        valid_ids = np.flatnonzero(obs.valid)
        shown = valid_ids[::10]
        for vertex_id in shown:
            target = tuple(np.rint(obs.points_2d[vertex_id]).astype(int))
            fit = tuple(np.rint(projected[position, vertex_id]).astype(int))
            cv2.circle(canvas, target, 3, (0, 255, 0), -1)
            cv2.circle(canvas, fit, 2, (0, 0, 255), -1)
        small = cv2.resize(canvas, (1440, 720))
        cv2.imwrite(str(visual_dir / f"frame708_cam2_{policy}_full.jpg"), small)
        for crop_name, names in {
            "head_torso": ("head_neck", "torso"),
            "arms_hands": ("left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_hand", "right_hand"),
            "legs_feet": ("left_upper_leg", "right_upper_leg", "left_lower_leg", "right_lower_leg", "left_foot", "right_foot"),
        }.items():
            ids = np.concatenate([regions[name] for name in names])
            ids = ids[obs.valid[ids]]
            points = obs.points_2d[ids]
            x0, y0 = np.maximum(np.floor(points.min(0) - 80).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(points.max(0) + 80).astype(int), (5760, 2880))
            if x1 > x0 and y1 > y0:
                cv2.imwrite(str(visual_dir / f"frame708_cam2_{policy}_{crop_name}.jpg"), canvas[y0:y1, x0:x1])

    print(json.dumps(speed_quality, indent=2))


if __name__ == "__main__":
    main()
