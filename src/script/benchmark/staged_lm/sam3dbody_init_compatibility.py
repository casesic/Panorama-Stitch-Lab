"""SAM 3D Body parameter and coordinate compatibility audit.

The audit is intentionally conservative: unavailable raw parameters or an
unclosed coordinate conversion disables the initialization bake-off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lib_pano.human3d.crop_upright_coord import upright_points_to_equirect_rotate
from lib_pano.human3d.sam3dbody_initialization import (
    audit_crop_projection, basis_diagnostics, classify_body_pose,
    orthonormal_basis_from_rays,
)
from lib_pano.human3d.sam3dbody_io import load_sam3dbody_recon


ESTIMATOR_SELECTED_KEYS = [
    "bbox", "focal_length", "pred_keypoints_3d", "pred_keypoints_2d",
    "pred_vertices", "pred_cam_t", "pred_pose_raw", "global_rot",
    "body_pose_params", "hand_pose_params", "scale_params", "shape_params",
    "expr_params", "mask", "pred_joint_coords", "pred_global_rots",
    "mhr_model_params", "lhand_bbox", "rhand_bbox",
]


def _metric(delta: np.ndarray) -> dict[str, float]:
    value = np.asarray(delta, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(value))), "rms": float(np.sqrt(np.mean(value * value)))}


def _erp_ray(u: float, v: float, width: int, height: int) -> np.ndarray:
    lon = (u / width - 0.5) * (2.0 * np.pi)
    lat = (0.5 - v / height) * np.pi
    camera = np.array([np.sin(lon) * np.cos(lat), -np.sin(lat), np.cos(lon) * np.cos(lat)])
    # Inverse of production camera_points = world_points @ ROTATION_MATRIX.T.
    rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    return camera @ rotation


def _basis_for_bbox(bbox: list[int], width: int, height: int) -> dict[str, Any]:
    x = 0.5 * (bbox[0] + bbox[2]); y = 0.5 * (bbox[1] + bbox[3])
    center = _erp_ray(x, y, width, height)
    right = _erp_ray(x + 1.0, y, width, height)
    down = _erp_ray(x, y + 1.0, width, height)
    basis = orthonormal_basis_from_rays(center, right, down)
    return {"matrix": basis.tolist(), "center_forward": center.tolist(), **basis_diagnostics(basis)}


def _mapping_closure(recon: Any, dense_path: Path, mapping_path: Path) -> dict[str, Any]:
    if not mapping_path.exists():
        return {"status": "MISSING_MHR_TO_SMPLX_MAPPING", "passed": False}
    if recon.faces is None or not dense_path.exists():
        return {"status": "MISSING_FACES_OR_DENSE_OBSERVATION", "passed": False}
    with np.load(mapping_path, allow_pickle=False) as mapping:
        triangle_ids = mapping["triangle_ids"].astype(np.int64)
        bary = mapping["baryc_coords"].astype(np.float64)
    triangles = recon.faces[triangle_ids]
    vertices_cam = recon.pred_vertices.astype(np.float64) + recon.pred_cam_t.astype(np.float64)[None, :]
    surface = np.sum(vertices_cam[triangles] * bary[..., None], axis=1)
    height, width = recon.image_shape[:2]
    mapped = np.stack((
        recon.focal_length * surface[:, 0] / surface[:, 2] + width / 2.0,
        recon.focal_length * surface[:, 1] / surface[:, 2] + height / 2.0,
    ), axis=1)
    with np.load(dense_path, allow_pickle=False) as dense:
        expected = dense["points_2d"].astype(np.float64)
    error = _metric(mapped - expected)
    return {"status": "PASS" if error["max_abs"] <= 1e-4 else "FAIL",
            "passed": error["max_abs"] <= 1e-4, "crop_uv_error": error,
            "point_count": int(mapped.shape[0]), "mapping_keys": ["triangle_ids", "baryc_coords"]}


def source_semantics(repo_root: Path) -> dict[str, Any]:
    return {
        "selected_dict": {"path": "tools/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py",
                          "function": "SAM3DBodyEstimator.process_one_image", "keys": ESTIMATOR_SELECTED_KEYS},
        "global_rot": {"path": "tools/sam-3d-body/sam_3d_body/models/heads/mhr_head.py",
                       "function": "MHRHead.forward", "network_representation": "6D rotation",
                       "exported_representation": "ZYX Euler radians [3]", "coordinate": "MHR body in crop camera convention",
                       "included_in_pred_vertices": True},
        "body_pose_params": {"path": "tools/sam-3d-body/sam_3d_body/models/modules/mhr_utils.py",
                             "function": "compact_cont_to_model_params_body", "shape": [133],
                             "semantics": "MHR model parameters: mixed 3-DOF Euler rotations, 1-DOF rotations, and 1-DOF translations",
                             "smplx_axis_angle_63": False},
        "shape_params": {"path": "tools/sam-3d-body/sam_3d_body/models/heads/mhr_head.py",
                         "function": "MHRHead.forward/mhr_forward", "shape": [45],
                         "model_space": "MHR", "direct_smplx_beta10": False},
        "pred_cam_t": {"path": "tools/sam-3d-body/sam_3d_body/models/heads/camera_head.py",
                       "function": "CameraHead.forward", "semantics": "perspective crop/original-input camera translation (tx+cx,ty+cy,tz), tz=2f/(bbox_size*scale)"},
        "postprocess": {"path": "tools/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py",
                        "functions": ["forward_hand", "replace_hands", "mhr_forward"],
                        "note": "hand replacement/IK path may update pose/shape before final pred_vertices"},
    }


def audit_compatibility(*, frames: list[int], recon_dir: Path, dense_dir: Path,
                        bbox_dir: Path, mapping_path: Path, erp_width: int = 5760,
                        erp_height: int = 2880) -> dict[str, Any]:
    records=[]; exact_available={name:True for name in ("global_rot","body_pose_params","shape_params")}
    mapping_pass=True; projection_pass=True
    bboxes={cam:json.loads((bbox_dir/f"{cam}_bboxes.json").read_text()) for cam in ("cam2","cam4")}
    for frame in frames:
        for cam in ("cam2","cam4"):
            path=recon_dir/cam/f"{frame:05d}.npz"
            if not path.exists():
                records.append({"frame":frame,"cam":cam,"status":"MISSING_RECON"}); projection_pass=False; continue
            recon=load_sam3dbody_recon(path)
            with np.load(path,allow_pickle=False) as raw:
                cache_keys=list(raw.files)
            closure=audit_crop_projection(recon);projection_pass &= closure.passed
            for name in exact_available: exact_available[name] &= getattr(recon,name) is not None
            bbox=[int(v) for v in bboxes[cam][str(frame)]]
            full_points=upright_points_to_equirect_rotate(recon.vertices_2d,bbox,frame)
            mapping=_mapping_closure(recon,dense_dir/cam/f"{frame:05d}.npz",mapping_path)
            mapping_pass &= bool(mapping["passed"])
            records.append({"frame":frame,"cam":cam,"status":"OK","cache_keys":cache_keys,
                "optional_parameters":{name:(None if getattr(recon,name) is None else {
                    "shape":list(getattr(recon,name).shape),"dtype":str(getattr(recon,name).dtype)}) for name in exact_available},
                "crop_projection":closure.__dict__,"mhr_to_smplx_mapping":mapping,
                "full_erp_uv_range":{"min":full_points.min(0).tolist(),"max":full_points.max(0).tolist()},
                "crop_to_erp_basis":_basis_for_bbox(bbox,erp_width,erp_height)})
    pose_class=(classify_body_pose(next((load_sam3dbody_recon(recon_dir/c/f"{f:05d}.npz").body_pose_params
        for f in frames for c in ("cam2","cam4") if (recon_dir/c/f"{f:05d}.npz").exists()),None)).value)
    capability={
        "sam_global_rot_available":"PASS" if exact_available["global_rot"] else "FAIL",
        "sam_global_rot_reconstructable":"UNKNOWN" if exact_available["global_rot"] else "NOT_APPLICABLE",
        "sam_global_rot_representation":"ZYX_EULER_RADIANS" if exact_available["global_rot"] else "KNOWN_FROM_SOURCE_NOT_EXPORTED",
        "sam_global_rot_coordinate_known":"FAIL",
        "sam_body_pose_available":"PASS" if exact_available["body_pose_params"] else "FAIL",
        "sam_body_pose_reconstructable":"UNKNOWN" if exact_available["body_pose_params"] else "NOT_APPLICABLE",
        "sam_body_pose_direct_smplx_compatible":"FAIL",
        "sam_body_pose_conversion_available":"FAIL",
        "sam_shape_available":"PASS" if exact_available["shape_params"] else "FAIL",
        "sam_shape_direct_smplx_compatible":"FAIL",
        "sam_pred_cam_t_available":"PASS",
        "sam_translation_erp_conversion_valid":"FAIL",
        "mhr_to_smplx_mapping_available":"PASS" if mapping_path.exists() else "FAIL",
        "geometry_rigid_initializer_available":"FAIL",
        "official_mhr_mesh_to_smplx_conversion":"PASS" if mapping_path.exists() else "FAIL",
        "official_conversion_input_from_existing_cache":"PASS",
    }
    direct_fields=all(exact_available.values())
    failures=[]
    if not direct_fields: failures.append("RAW_PARAMETER_FIELDS_NOT_EXPORTED")
    if not mapping_pass: failures.append("MHR_TO_SMPLX_MAPPING_CLOSURE_FAILED")
    failures.extend(["MHR_BODY_POSE_IS_NOT_SMPLX_AXIS_ANGLE_63","MHR_SHAPE_IS_NOT_SMPLX_BETA10",
                     "CROP_CAMERA_TO_UNIFIED_CAM2_RIGID_TRANSFORM_NOT_CLOSED"])
    bakeoff_allowed=False
    classification="MHR_PARAMETERS_NOT_DIRECTLY_SMPLX_COMPATIBLE"
    return {"raw_schema":{"records":records,"source_selected_keys":ESTIMATOR_SELECTED_KEYS},
        "source_code_semantics":source_semantics(Path.cwd()),
        "parameter_reconstruction":{"status":"NOT_RUN_MISSING_RAW_PARAMETERS" if not direct_fields else "REQUIRES_MHR_MODEL_INTEGRATION",
                                    "gate":"SAM_RAW_PARAMETER_RECONSTRUCTION_FAIL"},
        "body_pose_compatibility":pose_class,"shape_compatibility":"MHR_SHAPE_NOT_DIRECTLY_COMPATIBLE_WITH_SMPLX_BETA",
        "crop_projection_consistency":{"passed":projection_pass},
        "cam4_to_cam2_conversion":{"status":"UNRESOLVED","closed_loop":False},
        "translation_candidates":{"status":"NOT_RUN_COORDINATE_GATE_FAILED"},
        "geometry_mapping":{"available":mapping_path.exists(),"all_sample_closures_pass":mapping_pass},
        "official_mesh_to_smplx_conversion":{
            "available_with_existing_cache": bool(mapping_path.exists()),
            "required_inputs": ["pred_vertices", "pred_cam_t"],
            "input_formula": "100 * (pred_vertices + pred_cam_t[:,None,:])",
            "implementation": "tools/MHR/tools/mhr_smpl_conversion/Conversion.convert_sam3d_output_to_smpl",
            "status": "READY_FOR_CONVERSION_VALIDATION" if mapping_path.exists() else "MISSING_MAPPING",
            "local_runtime_blocker": "PYMOMENTUM_IMPORT_MISSING",
            "local_api_note": "This checkout np.stack()s every optional SAM key before selecting pred_vertices; a thin adapter must call convert_mhr2smpl on the documented vertex formula or avoid empty optional stacks.",
        },
        "capability_table":capability,"failed_capabilities":failures,"bakeoff_allowed":bakeoff_allowed,
        "classification":classification,"recommendation":"RUN_OFFICIAL_MESH_TO_SMPLX_CONVERSION_BEFORE_BAKEOFF",
        "next_action":"CONVERTED_POSE_ONLY_VALIDATION"}
