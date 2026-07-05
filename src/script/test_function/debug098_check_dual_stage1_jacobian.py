"""Stage 1 dual-view Jacobian correctness verification: autograd vs central finite differences.

Verifies the raw reprojection residual Jacobian (not scalar loss gradient) for the
12-D Stage 1 parameter vector [cam2_orient(3), cam2_transl(3), cam4_orient(3), cam4_transl(3)]
against central-difference approximations.

Only supports sam3d_dense primary observation source, single frame, no optimization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.func import jacfwd

import config_script
from lib_pano.fitting_io.hmr_init import (
    build_hmr_init_for_frame,
    build_segment_smplx_init,
    try_load_smplx_init,
)
from lib_pano.fitting_io.sam3d_dense_io import load_sam3d_dense_result
from lib_pano.human3d.crop_upright_coord import (
    resolve_dataset_rotate_rules,
    resolve_rotate_rule,
)
from lib_pano.human_fitting import DualCamERPFitter
from lib_pano.human_fitting.constraints.sam3d_dense import (
    select_valid_sam3d_dense_points,
)
from lib_pano.human_fitting.smplx_sequence import forward_smplx_tensor_batch
from lib_pano.pipeline.bbox_io import load_bbox_dict as load_bbox_json

R_CAM4 = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)
PARAM_DIM = 12
MIN_VALID_POINTS = 16
DEFAULT_EPS_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
THRESHOLD = 5e-3


def _elapsed_ms_since(t0: float) -> float:
    """Return elapsed wall time since *t0* in milliseconds, with CUDA sync if needed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for single-frame Jacobian check."""
    parser = argparse.ArgumentParser(
        description="Stage 1 dual-view Jacobian correctness verification (sam3d_dense only)",
    )
    parser.add_argument("--frame_idx", type=int, required=True, help="Single frame index.")
    parser.add_argument(
        "--hmr_dir", type=str, required=True,
        help="HMR pkl root directory containing cam2/ and cam4/ subdirectories.",
    )
    parser.add_argument(
        "--sam3d_dense_dir", type=str, required=True,
        help="SAM 3D Body dense observation root (cam2/ & cam4/).",
    )
    parser.add_argument(
        "--bbox_dir", type=str, required=True,
        help="BBox json directory containing cam2_bboxes.json and cam4_bboxes.json.",
    )
    parser.add_argument("--image_width", type=int, default=5760, help="Equirect image width.")
    parser.add_argument("--image_height", type=int, default=2880, help="Equirect image height.")
    parser.add_argument(
        "--autograd_mode", type=str, choices=["forward", "reverse"], default="forward",
        help="Autograd mode: forward (jacfwd, faster) or reverse (standard jacobian).",
    )
    parser.add_argument(
        "--eps_list", type=float, nargs="+", default=DEFAULT_EPS_LIST,
        help="Space-separated list of central-difference step sizes.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    return parser.parse_args()


def load_observations(
    frame_idx: int,
    sam3d_dense_dir: Path,
    bbox_dir: Path,
    image_width: int,
    image_height: int,
    rotate_rules: list[dict],
) -> tuple[Any, Any]:
    """Load SAM3D dense observations for cam2 and cam4."""
    cam2_bboxes = load_bbox_json(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_json(bbox_dir / "cam4_bboxes.json")
    rotate_str = resolve_rotate_rule(frame_idx, rotate_rules)

    obs_c2 = load_sam3d_dense_result(
        frame_idx, sam3d_dense_dir, "cam2",
        image_width, image_height,
        crop_bbox=cam2_bboxes.get(str(frame_idx)),
        rotate_rule=rotate_str,
    )
    if obs_c2 is None:
        raise RuntimeError(f"cam2 SAM3D dense observation missing for frame {frame_idx}")

    obs_c4 = load_sam3d_dense_result(
        frame_idx, sam3d_dense_dir, "cam4",
        image_width, image_height,
        crop_bbox=cam4_bboxes.get(str(frame_idx)),
        rotate_rule=rotate_str,
    )
    if obs_c4 is None:
        raise RuntimeError(f"cam4 SAM3D dense observation missing for frame {frame_idx}")

    if not obs_c2.valid.any():
        raise RuntimeError(f"cam2 SAM3D dense observation has no valid points for frame {frame_idx}")
    if not obs_c4.valid.any():
        raise RuntimeError(f"cam4 SAM3D dense observation has no valid points for frame {frame_idx}")
    return obs_c2, obs_c4


def build_frozen_params(
    smpl_c2: Any,
    smpl_c4: Any,
    shared_beta: torch.Tensor,
    shared_body_pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Detach all non-Stage1 params into a frozen dict."""
    return {
        "shared_beta": shared_beta.detach().clone(),
        "shared_body_pose": shared_body_pose.detach().clone(),
        "lhand_c2": smpl_c2.left_hand_pose.detach().clone(),
        "rhand_c2": smpl_c2.right_hand_pose.detach().clone(),
        "lhand_c4": smpl_c4.left_hand_pose.detach().clone(),
        "rhand_c4": smpl_c4.right_hand_pose.detach().clone(),
        "scale_c2": smpl_c2.scale.detach().clone().reshape(-1)[:1],
        "scale_c4": smpl_c4.scale.detach().clone().reshape(-1)[:1],
    }


def precompute_valid_points(
    obs_c2: Any,
    obs_c4: Any,
    sample_ids: np.ndarray,
) -> tuple[dict, dict]:
    """Pre-compute valid vertex ids and target 2D points for both cameras."""
    def _prepare(obs: Any, camera: str) -> dict:
        valid_ids, target_2d = select_valid_sam3d_dense_points(obs, sample_ids)
        num_valid = int(valid_ids.shape[0])
        if num_valid < MIN_VALID_POINTS:
            raise RuntimeError(
                f"{camera}: only {num_valid} valid SAM3D dense points, "
                f"need at least {MIN_VALID_POINTS}"
            )
        return {"valid_ids": valid_ids, "target_2d": target_2d, "num_valid": num_valid}

    info_c2 = _prepare(obs_c2, "cam2")
    info_c4 = _prepare(obs_c4, "cam4")
    return info_c2, info_c4


def make_residual_fn(
    fitter: DualCamERPFitter,
    model: Any,
    frozen: dict[str, torch.Tensor],
    info_c2: dict,
    info_c4: dict,
    erp_w: int,
    erp_h: int,
) -> Any:
    """Return a closure residual_fn(x: Tensor[12]) -> Tensor[M]."""
    shared_beta = frozen["shared_beta"]
    shared_body_pose = frozen["shared_body_pose"]
    lhand_c2 = frozen["lhand_c2"]
    rhand_c2 = frozen["rhand_c2"]
    lhand_c4 = frozen["lhand_c4"]
    rhand_c4 = frozen["rhand_c4"]
    scale_c2 = frozen["scale_c2"]
    scale_c4 = frozen["scale_c4"]

    valid_ids_c2_t = torch.as_tensor(info_c2["valid_ids"].astype(np.int64))
    target_c2_t = torch.as_tensor(info_c2["target_2d"].astype(np.float32))
    valid_ids_c4_t = torch.as_tensor(info_c4["valid_ids"].astype(np.int64))
    target_c4_t = torch.as_tensor(info_c4["target_2d"].astype(np.float32))

    def residual_fn(x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype

        cam2_orient = x[0:3].view(1, 3)
        cam2_transl = x[3:6].view(1, 3)
        cam4_orient = x[6:9].view(1, 3)
        cam4_transl = x[9:12].view(1, 3)

        verts_c2, _, _ = forward_smplx_tensor_batch(
            smplx_model=model,
            global_orient=cam2_orient,
            body_pose=shared_body_pose.unsqueeze(0).to(device, dtype=dtype),
            left_hand_pose=lhand_c2.to(device, dtype=dtype),
            right_hand_pose=rhand_c2.to(device, dtype=dtype),
            betas=shared_beta.unsqueeze(0).to(device, dtype=dtype),
            transl=cam2_transl,
            scale=scale_c2.expand(1).to(device, dtype=dtype),
        )
        verts_c4, _, _ = forward_smplx_tensor_batch(
            smplx_model=model,
            global_orient=cam4_orient,
            body_pose=shared_body_pose.unsqueeze(0).to(device, dtype=dtype),
            left_hand_pose=lhand_c4.to(device, dtype=dtype),
            right_hand_pose=rhand_c4.to(device, dtype=dtype),
            betas=shared_beta.unsqueeze(0).to(device, dtype=dtype),
            transl=cam4_transl,
            scale=scale_c4.expand(1).to(device, dtype=dtype),
        )

        proj_c2 = fitter._project_cam2(verts_c2, erp_w, erp_h).squeeze(0)  # [10475, 2]
        proj_c4 = fitter._project_cam2(verts_c4, erp_w, erp_h).squeeze(0)  # [10475, 2]

        idx_c2 = valid_ids_c2_t.to(device)
        idx_c4 = valid_ids_c4_t.to(device)
        pred_c2 = proj_c2.index_select(0, idx_c2)
        pred_c4 = proj_c4.index_select(0, idx_c4)

        tgt_c2 = target_c2_t.to(device, dtype=dtype)
        tgt_c4 = target_c4_t.to(device, dtype=dtype)

        r_c2 = (pred_c2 - tgt_c2).reshape(-1)
        r_c4 = (pred_c4 - tgt_c4).reshape(-1)
        return torch.cat([r_c2, r_c4])

    return residual_fn


def check_no_nan_inf(tensor: torch.Tensor, name: str) -> None:
    """Raise if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise RuntimeError(f"{name} contains NaN or Inf")


def compute_autograd_jacobian(
    residual_fn: Any,
    x0: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, float]:
    """Compute full Jacobian via forward-mode (jacfwd) or reverse-mode (jacrev/jacobian)."""
    x_input = x0.detach().clone().requires_grad_(True)
    t0 = time.perf_counter()

    if mode == "forward":
        try:
            J = jacfwd(residual_fn)(x_input)
        except Exception as exc:
            raise RuntimeError(
                f"Forward-mode autograd (jacfwd) failed on residual_fn. "
                f"Try --autograd_mode reverse.\n"
                f"  Original error: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        J = torch.autograd.functional.jacobian(residual_fn, x_input)

    ms = _elapsed_ms_since(t0)
    J = J.detach().cpu()
    check_no_nan_inf(J, "J_auto")
    return J, ms


def compute_fd_jacobian(
    residual_fn: Any,
    x0: torch.Tensor,
    eps: float,
    x0_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Compute full Jacobian via central finite differences column by column (no-grad)."""
    if x0_ref is None:
        x0_ref = x0.detach().clone()
    M = -1  # will be set after first column
    J_fd = torch.zeros(0, PARAM_DIM, dtype=torch.float64)
    t0 = time.perf_counter()

    for j in range(PARAM_DIM):
        x_plus = x0_ref.clone()
        x_minus = x0_ref.clone()
        x_plus[j] += eps
        x_minus[j] -= eps
        with torch.no_grad():
            r_plus = residual_fn(x_plus).cpu()
            r_minus = residual_fn(x_minus).cpu()

        if M < 0:
            M = int(r_plus.shape[0])
            J_fd = torch.zeros(M, PARAM_DIM, dtype=torch.float64)
        J_fd[:, j] = ((r_plus - r_minus) / (2.0 * eps)).to(torch.float64)

    ms = _elapsed_ms_since(t0)
    check_no_nan_inf(J_fd, "J_fd")
    return J_fd, ms


def compute_metrics(
    J_auto: torch.Tensor,
    J_fd: torch.Tensor,
) -> dict[str, float]:
    """Compute global comparison metrics."""
    diff = J_auto.to(torch.float64) - J_fd
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    norm_auto = J_auto.norm("fro").item()
    norm_fd = J_fd.norm("fro").item()
    denom = max(norm_auto, norm_fd, 1e-12)
    rel_frob = float(diff.norm("fro").item() / denom)
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_frobenius_error": rel_frob,
    }


def compute_block_metrics(
    J_auto: torch.Tensor,
    J_fd: torch.Tensor,
    block_slices: dict[str, slice],
) -> dict[str, dict[str, float]]:
    """Compute per-block comparison metrics."""
    out = {}
    J_a = J_auto.to(torch.float64)
    J_f = J_fd
    for name, sl in block_slices.items():
        diff_block = J_a[:, sl] - J_f[:, sl]
        max_abs = float(diff_block.abs().max().item())
        norm_a = J_a[:, sl].norm("fro").item()
        norm_f = J_f[:, sl].norm("fro").item()
        denom = max(norm_a, norm_f, 1e-12)
        rel_frob = float(diff_block.norm("fro").item() / denom)
        out[name] = {
            "relative_frobenius_error": rel_frob,
            "max_abs_error": max_abs,
        }
    return out


def get_device_info(x0: torch.Tensor, model: Any) -> tuple[str, str, str, str]:
    """Return (x0_device, x0_dtype, model_device, model_dtype)."""
    model_device = "unknown"
    model_dtype = "unknown"
    for attr in ("v_template", "shapedirs", "lbs_weights"):
        v = getattr(model, attr, None)
        if v is not None:
            model_device = str(v.device)
            model_dtype = str(v.dtype).split(".")[-1]
            break
    if model_device == "unknown":
        params = list(model.parameters())
        if params:
            model_device = str(params[0].device)
            model_dtype = str(params[0].dtype).split(".")[-1]
    return (
        str(x0.device),
        str(x0.dtype).split(".")[-1],
        model_device,
        model_dtype,
    )


def main() -> None:
    args = parse_args()

    frame_idx = args.frame_idx
    eps_list = sorted(args.eps_list)
    autograd_mode = args.autograd_mode
    erp_w = args.image_width
    erp_h = args.image_height

    hmr_dir = Path(args.hmr_dir)
    bbox_dir = Path(args.bbox_dir)
    sam3d_dense_dir = Path(args.sam3d_dense_dir)
    output_dir = Path(args.output_dir)

    print("[INFO] Both cam2 and cam4 residuals are evaluated in cam2 ERP coordinates via _project_cam2().")

    # === Load data ===
    rotate_rules_config = Path(config_script.PROJECT_ROOT) / "configs" / "rotate_upright_rules.json"
    rotate_rules = resolve_dataset_rotate_rules(rotate_rules_config, bbox_dir)

    obs_c2, obs_c4 = load_observations(
        frame_idx, sam3d_dense_dir, bbox_dir, erp_w, erp_h, rotate_rules,
    )

    hmr_invalid: set[int] = set()
    direct_hmr_c2 = {
        frame_idx: try_load_smplx_init(frame_idx, hmr_dir, hmr_invalid, "cam2"),
    }
    direct_hmr_c4 = {
        frame_idx: try_load_smplx_init(frame_idx, hmr_dir, hmr_invalid, "cam4"),
    }
    if direct_hmr_c2[frame_idx] is None and direct_hmr_c4[frame_idx] is None:
        raise RuntimeError(f"No HMR init available for frame {frame_idx}")

    smplx_c2, init_c2 = build_hmr_init_for_frame(frame_idx, [frame_idx], direct_hmr_c2)
    smplx_c4, init_c4 = build_hmr_init_for_frame(frame_idx, [frame_idx], direct_hmr_c4)
    smplx_init_c2, smplx_init_c4 = build_segment_smplx_init([smplx_c2], [smplx_c4])

    # === Create fitter and initialize params ===
    fitter = DualCamERPFitter(
        primary_observation_source="sam3d_dense",
        gender="neutral",
        R_cam4=R_CAM4,
    )

    smpl_c2, smpl_c4, shared_beta, shared_body_pose = fitter._init_params(
        smplx_init_c2=smplx_init_c2,
        smplx_init_c4=smplx_init_c4,
        primary_observations_c2=[obs_c2],
        primary_observations_c4=[obs_c4],
        image_size=(erp_w, erp_h),
    )

    # === Build x0 from init params ===
    x0 = torch.cat([
        smpl_c2.orient.detach().reshape(-1),
        smpl_c2.transl.detach().reshape(-1),
        smpl_c4.orient.detach().reshape(-1),
        smpl_c4.transl.detach().reshape(-1),
    ]).clone()

    # === Freeze non-Stage1 params ===
    frozen = build_frozen_params(smpl_c2, smpl_c4, shared_beta, shared_body_pose)

    # === Pre-compute valid points ===
    sample_ids = fitter._sam3d_dense_sample_ids
    if sample_ids is None:
        raise RuntimeError("fitter._sam3d_dense_sample_ids is None; ensure source is sam3d_dense")

    info_c2, info_c4 = precompute_valid_points(obs_c2, obs_c4, sample_ids)
    n_c2 = info_c2["num_valid"]
    n_c4 = info_c4["num_valid"]
    residual_dim = 2 * (n_c2 + n_c4)

    print(f"[INFO] residual_dim={residual_dim}, cam2_valid_points={n_c2}, cam4_valid_points={n_c4}")

    # === Build residual function and model ===
    model = fitter._get_model()
    residual_fn = make_residual_fn(fitter, model, frozen, info_c2, info_c4, erp_w, erp_h)

    # === Print device/dtype info ===
    x0_dev, x0_dt, model_dev, model_dt = get_device_info(x0, model)
    print(f"[INFO] x0 device={x0_dev} dtype={x0_dt}")
    print(f"[INFO] model device={model_dev} dtype={model_dt}")
    if x0_dev != model_dev or x0_dt != model_dt:
        raise RuntimeError(
            f"x0 (device={x0_dev}, dtype={x0_dt}) does not match "
            f"model (device={model_dev}, dtype={model_dt})"
        )

    # === Check initial residual ===
    r_test = residual_fn(x0).detach().cpu()
    check_no_nan_inf(r_test, "initial residual")
    assert int(r_test.shape[0]) == residual_dim, f"residual shape {r_test.shape[0]} != {residual_dim}"

    # === Time a single residual forward for reference ===
    t0 = time.perf_counter()
    residual_fn(x0)
    rf_ms = _elapsed_ms_since(t0)
    print(f"[TIME] residual_forward_ms={rf_ms:.1f}")

    # === Autograd Jacobian ===
    print(f"[JAC] Computing autograd Jacobian (mode={autograd_mode}) ...")
    J_auto, auto_ms = compute_autograd_jacobian(residual_fn, x0, autograd_mode)
    print(f"[TIME] autograd_jacobian_ms={auto_ms:.1f}")

    # === Finite-difference Jacobian for each eps ===
    block_slices = {
        "cam2_orient": slice(0, 3),
        "cam2_transl": slice(3, 6),
        "cam4_orient": slice(6, 9),
        "cam4_transl": slice(9, 12),
    }

    eps_metrics: dict[str, Any] = {}
    best_eps: float | None = None
    best_rel_frob: float = float("inf")
    best_J_fd: torch.Tensor | None = None
    x0_ref = x0.detach().clone()

    for eps in eps_list:
        key = f"{eps:.0e}" if eps >= 1e-3 else f"{eps:.1e}"
        print(f"[JAC] Computing central-difference Jacobian (eps={key}) ...")
        J_fd_cur, fd_ms = compute_fd_jacobian(residual_fn, x0, eps, x0_ref)
        print(f"[TIME] fd_eps={key} ms={fd_ms:.1f}")

        gm = compute_metrics(J_auto, J_fd_cur)
        bm = compute_block_metrics(J_auto, J_fd_cur, block_slices)
        eps_metrics[key] = {
            "global_metrics": gm,
            "block_metrics": bm,
            "fd_ms": round(fd_ms, 3),
        }

        print(f"[EPS] eps={key} rel_frob={gm['relative_frobenius_error']:.8f} "
              f"max_abs={gm['max_abs_error']:.8f} mean_abs={gm['mean_abs_error']:.8f}")

        if gm["relative_frobenius_error"] < best_rel_frob:
            best_rel_frob = gm["relative_frobenius_error"]
            best_eps = eps
            best_J_fd = J_fd_cur

    print(f"[EPS] best_eps={best_eps:.0e}" if best_eps is not None and best_eps >= 1e-3
          else f"[EPS] best_eps={best_eps:.1e}",
          f"best_rel_frob={best_rel_frob:.8f}")
    assert best_J_fd is not None

    # === Report using best eps ===
    J_fd = best_J_fd
    global_metrics = eps_metrics[f"{best_eps:.1e}" if best_eps < 1e-3 else f"{best_eps:.0e}"]["global_metrics"]
    block_metrics = eps_metrics[f"{best_eps:.1e}" if best_eps < 1e-3 else f"{best_eps:.0e}"]["block_metrics"]

    print(f"\n[JAC] shape: residual_dim={residual_dim}, param_dim={PARAM_DIM}")
    g = global_metrics
    print(f"[JAC] all: rel_frob={g['relative_frobenius_error']:.8f}, "
          f"max_abs={g['max_abs_error']:.8f}, mean_abs={g['mean_abs_error']:.8f}")
    for name in ["cam2_orient", "cam2_transl", "cam4_orient", "cam4_transl"]:
        b = block_metrics[name]
        print(f"[JAC] {name}: rel_frob={b['relative_frobenius_error']:.8f}, "
              f"max_abs={b['max_abs_error']:.8f}")

    # === Threshold check ===
    passed = global_metrics["relative_frobenius_error"] <= THRESHOLD
    status = "PASS" if passed else "FAIL"
    print(f"[JAC] {status}: relative_frobenius_error={g['relative_frobenius_error']:.8f} "
          f"(threshold={THRESHOLD})")

    if not passed:
        print("[WARN] Global relative Frobenius error exceeds threshold.")
        print("[DIAG] This check validates derivatives of the implemented residual only.")
        print("[DIAG] Large mismatch usually indicates finite-difference eps/dtype issues,")
        print("[DIAG]        graph detach/in-place operations, or non-smooth projection branches.")
        print("[DIAG] Coordinate semantics such as cam4 double rotation and target/vertex")
        print("[DIAG]        correspondence require separate forward reprojection validation.")

    for name, bm_val in block_metrics.items():
        if bm_val["relative_frobenius_error"] > THRESHOLD * 10:
            print(f"[WARN] Block {name} rel_frob={bm_val['relative_frobenius_error']:.8f} "
                  f"is unusually large.")

    # === Save artifacts ===
    out_frame_dir = output_dir / str(frame_idx)
    out_frame_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(out_frame_dir / "jacobian_check.npz"),
        x0=x0.cpu().numpy(),
        residual=r_test.cpu().numpy(),
        J_auto=J_auto.cpu().numpy(),
        J_fd=J_fd.cpu().numpy(),
        sample_ids=np.asarray(sample_ids),
        valid_ids_cam2=info_c2["valid_ids"],
        valid_ids_cam4=info_c4["valid_ids"],
    )

    with (out_frame_dir / "jacobian_check.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "frame_idx": frame_idx,
                "autograd_mode": autograd_mode,
                "best_eps": best_eps,
                "residual_dim": residual_dim,
                "param_dim": PARAM_DIM,
                "cam2_valid_points": n_c2,
                "cam4_valid_points": n_c4,
                "global_metrics": global_metrics,
                "block_metrics": block_metrics,
                "eps_metrics": eps_metrics,
                "passed": passed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[SAVE] Artifacts saved to {out_frame_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
