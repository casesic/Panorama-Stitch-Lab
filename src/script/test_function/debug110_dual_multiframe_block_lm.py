"""Block-Structured LM: unified temporal/per-frame paths with reduced boilerplate."""

from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
from torch.func import jacfwd

import config_script
from lib_pano.fitting_io.hmr_init import (build_hmr_init_for_frame, build_segment_smplx_init, try_load_smplx_init)
from lib_pano.fitting_io.sam3d_dense_io import load_sam3d_dense_result
from lib_pano.human3d.crop_upright_coord import resolve_dataset_rotate_rules, resolve_rotate_rule
from lib_pano.human_fitting import DualCamERPFitter
from lib_pano.human_fitting.constraints.sam3d_dense import select_valid_sam3d_dense_points
from lib_pano.human_fitting.smplx_sequence import forward_smplx_tensor_batch
from lib_pano.pipeline.bbox_io import load_bbox_dict as load_bbox_json

R_CAM4 = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)
MIN_VALID_POINTS = 16; S1_PF = 12; POSE_PF = 63; BETA_ATANH_EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Block-Structured LM Phase 1")
    p.add_argument("--frames", type=str, required=True)
    p.add_argument("--hmr_dir", type=str, required=True)
    p.add_argument("--sam3d_dense_dir", type=str, required=True)
    p.add_argument("--bbox_dir", type=str, required=True)
    p.add_argument("--image_width", type=int, default=5760)
    p.add_argument("--image_height", type=int, default=2880)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--charbonnier_eps", type=float, default=1.0)
    p.add_argument("--pose_anchor_weight", type=float, default=1.0)
    p.add_argument("--shape_anchor_weight", type=float, default=10.0)
    p.add_argument("--beta_abs_limit", type=float, default=2.5)
    p.add_argument("--stop_after_stage", choices=["stage1", "stage2a", "stage2b"], default="stage2a")
    p.add_argument("--validate_irls_first_order", action="store_true", dest="validate_irls_first_order", default=True)
    p.add_argument("--no_validate_irls_first_order", action="store_false", dest="validate_irls_first_order")
    p.add_argument("--validate_irls_each_accepted", action="store_true", default=False)
    p.add_argument("--irls_gradient_rel_tol", type=float, default=5e-4)
    p.add_argument("--stage1_lm_max_acc", type=int, default=30)
    p.add_argument("--stage2a_lm_max_acc", type=int, default=60)
    p.add_argument("--stage2b_lm_max_acc", type=int, default=60)
    p.add_argument("--lm_max_trials", type=int, default=8)
    p.add_argument("--rel_obj_tol", type=float, default=1e-7)
    p.add_argument("--scaled_grad_tol", type=float, default=1e-7)
    p.add_argument("--step_tol", type=float, default=1e-7)
    p.add_argument("--profile_timing", action="store_true", default=False)
    p.add_argument("--progress_every_accepted", type=int, default=0)
    p.add_argument("--dense_reference_npz", type=str, default=None)
    p.add_argument("--dense_reference_json", type=str, default=None)
    p.add_argument("--linearization_impl", choices=["legacy", "fused"], default="fused")
    p.add_argument("--view_forward_impl", choices=["legacy", "batch2"], default="batch2")
    p.add_argument("--temporal_jacobian_impl", choices=["per_frame", "batched_jvp"], default="batched_jvp")
    p.add_argument("--endpoint_reference_npz", type=str, default=None)
    return p.parse_args()


# ============================================================
# Utilities
# ============================================================
def elapsed_ms(t0, ref=None):
    if ref is not None and ref.is_cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def load_observations(fidx, sam3d_d, bbox_d, W, H, rotate_rules):
    cb = load_bbox_json(bbox_d / "cam2_bboxes.json")
    c4b = load_bbox_json(bbox_d / "cam4_bboxes.json")
    rr = resolve_rotate_rule(fidx, rotate_rules)
    o2 = load_sam3d_dense_result(fidx, sam3d_d, "cam2", W, H, crop_bbox=cb.get(str(fidx)), rotate_rule=rr)
    o4 = load_sam3d_dense_result(fidx, sam3d_d, "cam4", W, H, crop_bbox=c4b.get(str(fidx)), rotate_rule=rr)
    if o2 is None or o4 is None: raise RuntimeError(f"Missing SAM3D dense for frame {fidx}")
    return o2, o4


def build_frozen_params(sc2, sc4, t):
    s2 = sc2.scale.detach().clone().reshape(-1)
    s4 = sc4.scale.detach().clone().reshape(-1)
    return {"lhand_c2": sc2.left_hand_pose.detach()[t:t + 1].clone(),
            "rhand_c2": sc2.right_hand_pose.detach()[t:t + 1].clone(),
            "lhand_c4": sc4.left_hand_pose.detach()[t:t + 1].clone(),
            "rhand_c4": sc4.right_hand_pose.detach()[t:t + 1].clone(),
            "scale_c2": s2[t].reshape(1) if t < s2.shape[0] else s2[0].reshape(1),
            "scale_c4": s4[t].reshape(1) if t < s4.shape[0] else s4[0].reshape(1)}


def precompute_valid_points(obs_c2, obs_c4, sample_ids):
    def _p(obs, cam):
        vid, tgt = select_valid_sam3d_dense_points(obs, sample_ids)
        if int(vid.shape[0]) < MIN_VALID_POINTS: raise RuntimeError(f"{cam}: {vid.shape[0]} < {MIN_VALID_POINTS}")
        return {"valid_ids": vid, "target_2d": tgt, "num_valid": int(vid.shape[0])}
    return _p(obs_c2, "cam2"), _p(obs_c4, "cam4")


def raw_reproj_metrics(r_raw):
    r2d = r_raw.reshape(-1, 2); n = torch.norm(r2d, dim=1).cpu(); sq = n ** 2
    return {"rms": float(torch.sqrt(sq.mean())), "median": float(n.median()),
            "p95": float(n.kthvalue(max(1, int(n.numel() * 0.95))).values)}


# ============================================================
# Temporal JVP helpers
# ============================================================
def _prepare_padded_targets(valid_ids, target_2d, N_fixed, all_sample_ids):
    vid = valid_ids.to(torch.int64); sid = all_sample_ids.to(torch.int64)
    valid_lookup = {int(v): i for i, v in enumerate(vid)}
    mask = torch.zeros(N_fixed, dtype=torch.bool); padded = torch.zeros(N_fixed, 2)
    for i, s in enumerate(sid):
        s_int = int(s)
        if s_int in valid_lookup: mask[i] = True; padded[i] = target_2d[valid_lookup[s_int]]
    return padded, mask


def _temporal_jvp_local(fn, L, local_dim, T):
    r = fn(L).detach()
    J_blocks = [torch.zeros(r.shape[1], local_dim, dtype=torch.float32, device=L.device) for _ in range(T)]
    for d in range(local_dim):
        tangent = torch.zeros_like(L); tangent[:, d] = 1.0
        _, dr = torch.func.jvp(fn, (L,), (tangent,))
        for t in range(T): J_blocks[t][:, d] = dr[t].detach()
    return r, J_blocks, local_dim


def _temporal_jvp_beta(fn, L, z, beta_dim, T):
    J_beta_blocks = [torch.zeros(fn(L, z).shape[1], beta_dim, dtype=torch.float32, device=L.device) for _ in range(T)]
    for k in range(beta_dim):
        tangent = torch.zeros(beta_dim, device=z.device); tangent[k] = 1.0
        _, dr = torch.func.jvp(lambda z_: fn(L, z_), (z,), (tangent,))
        for t in range(T): J_beta_blocks[t][:, k] = dr[t].detach()
    return J_beta_blocks, beta_dim


def refresh_temporal_irls_weights(L, z, stage, raw_groups_global_fn, static_spec, eps,
                                   N, masks_c2, masks_c4, sqrt_ws_c2, sqrt_ws_c4, T):
    xg = L.reshape(-1) if stage in ("S1", "S2A") else torch.cat([L.reshape(-1), z.detach()])
    groups = raw_groups_global_fn(xg)
    weights = compute_irls_point_weights(groups, eps, static_spec)
    sqrt_ws = [(torch.sqrt(w) if w.numel() > 0 else w).detach() for w in weights]
    new_c2 = torch.zeros(T, N, device=sqrt_ws_c2.device); new_c4 = torch.zeros(T, N, device=sqrt_ws_c4.device)
    for t in range(T):
        w_c2 = sqrt_ws[2 * t]; w_c4 = sqrt_ws[2 * t + 1]
        if w_c2.numel() > 0: new_c2[t][masks_c2[t]] = w_c2.to(new_c2.device)
        if w_c4.numel() > 0: new_c4[t][masks_c4[t]] = w_c4.to(new_c4.device)
    sqrt_ws_c2.copy_(new_c2.to(sqrt_ws_c2.device, sqrt_ws_c2.dtype))
    sqrt_ws_c4.copy_(new_c4.to(sqrt_ws_c4.device, sqrt_ws_c4.dtype))


def _build_batched_residual(T, W, H, device, bp_anchors_tensor, beta_tensor,
                             lhand_2T, rhand_2T, scale_2T, targets_c2_pad, masks_c2, sqrt_ws_c2,
                             targets_c4_pad, masks_c4, sqrt_ws_c4, model, fitter, sample_ids,
                             N, local_dim, pa_scale, stage, beta_abs_limit=None):
    has_pose_anchor = (stage != "S1"); is_s2b = (stage == "S2B")

    def batched_residual(L, z=None):
        T_act = L.shape[0]
        o2 = L[:, 0:3]; t2 = L[:, 3:6]; o4 = L[:, 6:9]; t4 = L[:, 9:12]
        orient_b = torch.cat([o2, o4], dim=0); transl_b = torch.cat([t2, t4], dim=0)
        if stage == "S1":
            bp_b = torch.cat([bp_anchors_tensor, bp_anchors_tensor], dim=0)
        else:
            bp_b = torch.cat([L[:, 12:local_dim], L[:, 12:local_dim]], dim=0)
        beta_b = (beta_abs_limit * torch.tanh(z)).expand(2 * T_act, -1) if is_s2b else beta_tensor.expand(2 * T_act, -1)
        v_b = forward_smplx_tensor_batch(model, orient_b, bp_b, lhand_2T, rhand_2T, beta_b, transl_b, scale_2T)[0]
        proj = fitter._project_cam2(v_b, W, H)
        p2 = proj[:T_act]; p4 = proj[T_act:]
        p2s = p2[:, sample_ids, :]; p4s = p4[:, sample_ids, :]
        r2 = torch.where(masks_c2[:T_act, :, None], (p2s - targets_c2_pad[:T_act]) * sqrt_ws_c2[:T_act, :, None],
                         torch.zeros_like(p2s))
        r4 = torch.where(masks_c4[:T_act, :, None], (p4s - targets_c4_pad[:T_act]) * sqrt_ws_c4[:T_act, :, None],
                         torch.zeros_like(p4s))
        parts = [r2.reshape(T_act, -1), r4.reshape(T_act, -1)]
        if has_pose_anchor: parts.append((L[:, 12:local_dim] - bp_anchors_tensor[:T_act]) * pa_scale)
        return torch.cat(parts, dim=1)
    return batched_residual


# ============================================================
# Per-frame reprojection
# ============================================================
def make_raw_reproj_for_frame(x_t, fidx, vid_c2, tgt_c2, vid_c4, tgt_c4,
                               frozen_t, model, fitter, W, H, stride, beta,
                               bp_frozen=None, view_forward_impl="batch2"):
    device, dtype = x_t.device, x_t.dtype
    o2 = x_t[0:3].view(1, 3); t2 = x_t[3:6].view(1, 3)
    o4 = x_t[6:9].view(1, 3); t4 = x_t[9:12].view(1, 3)
    bp = bp_frozen.to(device, dtype).view(1, -1) if bp_frozen is not None else x_t[12:stride].view(1, -1)

    if view_forward_impl == "batch2":
        orient_b = torch.cat([o2, o4], dim=0); transl_b = torch.cat([t2, t4], dim=0)
        body_pose_b = torch.cat([bp, bp], dim=0)
        beta_b = torch.cat([beta.unsqueeze(0).to(device, dtype), beta.unsqueeze(0).to(device, dtype)], dim=0)
        lh_b = torch.cat([frozen_t["lhand_c2"].to(device, dtype), frozen_t["lhand_c4"].to(device, dtype)], dim=0)
        rh_b = torch.cat([frozen_t["rhand_c2"].to(device, dtype), frozen_t["rhand_c4"].to(device, dtype)], dim=0)
        sc_b = torch.cat([frozen_t["scale_c2"].expand(1).to(device, dtype), frozen_t["scale_c4"].expand(1).to(device, dtype)], dim=0)
        p2, p4 = fitter._project_cam2(forward_smplx_tensor_batch(model, orient_b, body_pose_b, lh_b, rh_b, beta_b, transl_b, sc_b)[0], W, H)
    else:
        p2 = fitter._project_cam2(forward_smplx_tensor_batch(model, o2, bp,
            frozen_t["lhand_c2"].to(device, dtype), frozen_t["rhand_c2"].to(device, dtype),
            beta.unsqueeze(0).to(device, dtype), t2, frozen_t["scale_c2"].expand(1).to(device, dtype))[0], W, H).squeeze(0)
        p4 = fitter._project_cam2(forward_smplx_tensor_batch(model, o4, bp,
            frozen_t["lhand_c4"].to(device, dtype), frozen_t["rhand_c4"].to(device, dtype),
            beta.unsqueeze(0).to(device, dtype), t4, frozen_t["scale_c4"].expand(1).to(device, dtype))[0], W, H).squeeze(0)

    r2d_c2 = p2.index_select(0, vid_c2.to(device)) - tgt_c2.to(device, dtype)
    r2d_c4 = p4.index_select(0, vid_c4.to(device)) - tgt_c4.to(device, dtype)
    return [{"frame_idx": fidx, "camera": "cam2", "r2d": r2d_c2, "n_valid": int(r2d_c2.shape[0])},
            {"frame_idx": fidx, "camera": "cam4", "r2d": r2d_c4, "n_valid": int(r2d_c4.shape[0])}]


def flatten_groups(groups):
    return torch.cat([g["r2d"].reshape(-1) for g in groups])


# ============================================================
# Aggregation / Charbonnier / IRLS
# ============================================================
def build_primary_aggregation_spec(groups):
    frames_by_cam = {"cam2": set(), "cam4": set()}
    for g in groups: frames_by_cam[g["camera"]].add(g["frame_idx"])
    terms = {}
    for g in groups:
        n_frames = len(frames_by_cam[g["camera"]])
        alpha = 1.0 / (n_frames * g["n_valid"]) if g["n_valid"] > 0 and n_frames > 0 else 0.0
        terms[f"{g['frame_idx']}|{g['camera']}"] = {"n_valid": g["n_valid"], "valid_frame_count": n_frames,
                                                      "camera_weight": 1.0, "alpha_per_point": alpha}
    return {"terms": terms, "source": "sam3d_dense.py:272,135 + erp_dual.py:393"}


def compute_primary_charbonnier_tensor(groups, eps, spec):
    total = torch.tensor(0.0, device=groups[0]["r2d"].device, dtype=groups[0]["r2d"].dtype)
    for g in groups:
        if g["n_valid"] == 0: continue
        alpha = spec["terms"][f"{g['frame_idx']}|{g['camera']}"]["alpha_per_point"]
        if alpha <= 0: continue
        total = total + alpha * torch.sum(torch.sqrt(torch.norm(g["r2d"], dim=1) ** 2 + eps ** 2) - eps)
    return total


def compute_irls_point_weights(groups, eps, spec):
    weights = []
    for g in groups:
        if g["n_valid"] == 0: weights.append(torch.tensor([])); continue
        alpha = spec["terms"][f"{g['frame_idx']}|{g['camera']}"]["alpha_per_point"]
        if alpha <= 0: weights.append(torch.tensor([])); continue
        weights.append((alpha / torch.sqrt(torch.norm(g["r2d"], dim=1) ** 2 + eps ** 2)).detach())
    return weights


# ============================================================
# IRLS linearization
# ============================================================
def make_irls_linearization(raw_fn, anchor_fn, x, eps, spec, z=None):
    groups = raw_fn(x) if z is None else raw_fn(x, z)
    weights = compute_irls_point_weights(groups, eps, spec)
    flat_w = torch.cat([w for w in weights if w.numel() > 0])
    w_stats = {"min": float(flat_w.min()) if flat_w.numel() else 0, "mean": float(flat_w.mean()) if flat_w.numel() else 0,
               "median": float(flat_w.median()) if flat_w.numel() else 0, "max": float(flat_w.max()) if flat_w.numel() else 0,
               "n_points": int(flat_w.numel())}
    if z is None:
        def surrogate_fn(y):
            parts = []; gy = raw_fn(y)
            for i, g in enumerate(gy):
                if g["n_valid"] == 0 or i >= len(weights) or weights[i].numel() == 0: parts.append(g["r2d"].reshape(-1))
                else: parts.append((g["r2d"] * torch.sqrt(weights[i].to(g["r2d"].device)).unsqueeze(1)).reshape(-1))
            if anchor_fn is not None: parts.append(anchor_fn(y))
            return torch.cat(parts)
    else:
        def surrogate_fn(l, z_):
            parts = []; gy = raw_fn(l, z_)
            for i, g in enumerate(gy):
                if g["n_valid"] == 0 or i >= len(weights) or weights[i].numel() == 0: parts.append(g["r2d"].reshape(-1))
                else: parts.append((g["r2d"] * torch.sqrt(weights[i].to(g["r2d"].device)).unsqueeze(1)).reshape(-1))
            if anchor_fn is not None: parts.append(anchor_fn(l))
            return torch.cat(parts)
    return {"surrogate_fn": surrogate_fn, "weight_stats": w_stats}


# ============================================================
# Linearization primitives
# ============================================================
def _linearize_single(surrogate_fn, x, impl):
    if impl == "legacy":
        J = jacfwd(surrogate_fn)(x.detach().clone().requires_grad_(True)).detach()
        r = surrogate_fn(x).detach()
    else:
        def res_aux(x_): r_ = surrogate_fn(x_); return r_, r_
        J, r = jacfwd(res_aux, has_aux=True)(x.detach().clone().requires_grad_(True))
        J = J.detach(); r = r.detach()
    return J, r


def _linearize_dual(surrogate_fn, l, z, impl):
    local_dim = l.numel()
    if impl == "legacy":
        J_l = jacfwd(lambda x: surrogate_fn(x, z))(l.detach().clone().requires_grad_(True)).detach()
        J_z = jacfwd(lambda x: surrogate_fn(l, x))(z.detach().clone().requires_grad_(True)).detach()
        r = surrogate_fn(l, z).detach()
    else:
        u = torch.cat([l, z])
        def res_aux(u_): l_ = u_[:local_dim]; z_ = u_[local_dim:]; r_ = surrogate_fn(l_, z_); return r_, r_
        J_joint, r = jacfwd(res_aux, has_aux=True)(u.detach().clone().requires_grad_(True))
        J_l = J_joint[:, :local_dim].detach(); J_z = J_joint[:, local_dim:].detach(); r = r.detach()
    return J_l, J_z, r


# ============================================================
# Block gradient validation
# ============================================================
def validate_block_gradient(true_obj_global, x_blocks, compute_gn_block, label, args):
    x_global = torch.cat([xb.detach().clone().requires_grad_(True) for xb in x_blocks])
    g_true = torch.autograd.grad(true_obj_global(x_global), x_global, create_graph=False)[0].detach().to(torch.float64)
    g_block = compute_gn_block().to(torch.float64)
    diff = g_true - g_block
    abs_max = float(diff.abs().max()); rel_l2 = float(diff.norm() / max(g_true.norm().item(), 1.0))
    ok = rel_l2 <= args.irls_gradient_rel_tol
    print(f"[BLOCK_IRLS {label}] abs_max={abs_max:.2e} rel_l2={rel_l2:.2e} {'PASS' if ok else 'FAIL'}")
    return {"abs_max": abs_max, "rel_l2": rel_l2, "pass": ok}


# ============================================================
# Block LM — Single block (S1 / S2A)
# ============================================================
def block_lm_single(x_blocks, frame_dims, linearize_fn, true_obj_global, raw_metrics_global,
                    max_acc, max_trials, args, label="LM", validate_fn=None):
    T = len(x_blocks); total_dim = sum(frame_dims)
    sync = (lambda: torch.cuda.synchronize()) if (args.profile_timing and torch.cuda.is_available()) else (lambda: None)

    sync(); t0 = time.perf_counter()
    lin_result = linearize_fn(x_blocks); Hs, gs = lin_result["Hs"], lin_result["gs"]
    jacfwd_calls = lin_result.get("jacfwd_calls", T); lin_cnt = 1

    lam_min, lam_max = 1e-10, 1e12
    lam = max(float(torch.diag(H_t).max()) for H_t in Hs) * 1e-3
    acc, rej, slv_cnt, chol_ok, chol_fb = 0, 0, 0, 0, 0
    status, irls_ck = "RUNNING", []; prog_n = args.progress_every_accepted

    for outer in range(max_acc):
        sync(); obj_before = float(true_obj_global(torch.cat(x_blocks)).detach().item())
        accepted_this = False
        for trial in range(max_trials):
            deltas, ok = [], True; ls_count, c_ok, c_fb = 0, 0, 0
            for t in range(T):
                A = Hs[t] + lam * torch.eye(frame_dims[t], dtype=torch.float64); ls_count += 1
                try:
                    L_ch, info = torch.linalg.cholesky_ex(A)
                    if info.item(): raise RuntimeError
                    deltas.append(torch.cholesky_solve(-gs[t].unsqueeze(1), L_ch).squeeze(1)); c_ok += 1
                except Exception:
                    try: deltas.append(torch.linalg.solve(A, -gs[t].unsqueeze(1)).squeeze(1)); c_fb += 1
                    except Exception: ok = False; break
            slv_cnt += ls_count; chol_ok += c_ok; chol_fb += c_fb
            if not ok: lam = min(lam * 10, lam_max); rej += 1; continue

            x_trial = [(x_blocks[t] + deltas[t].to(torch.float32)).detach() for t in range(T)]
            sync(); obj_trial = float(true_obj_global(torch.cat(x_trial)).detach().item())
            pred = sum(float(-gs[t].dot(deltas[t]) - 0.5 * (deltas[t] @ Hs[t] @ deltas[t])) for t in range(T))
            actual = obj_before - obj_trial; rho = actual / max(pred, 1e-12) if pred > 0 else 0.0
            if pred > 0 and actual > 0 and rho > 0:
                x_blocks = x_trial; acc += 1; accepted_this = True
                lam = max(lam / 3, lam_min) if rho > 0.75 else (min(lam * 3, lam_max) if rho < 0.25 else lam)
                break
            else: lam = min(lam * 10, lam_max); rej += 1

        if not accepted_this:
            gn2 = sum(float(gs[t].norm() ** 2) for t in range(T))
            sg = math.sqrt(gn2) / max(1.0, max(float(torch.norm(Hs[t])) for t in range(T)))
            status = "CONVERGED_SCALED_GRADIENT" if (acc > 4 and sg < 1e-4) else "LM_NO_ACCEPTED_TRIAL"; break

        lin_result = linearize_fn(x_blocks); Hs, gs = lin_result["Hs"], lin_result["gs"]
        jacfwd_calls += lin_result.get("jacfwd_calls", T); lin_cnt += 1
        if validate_fn: validate_fn(x_blocks, lin_result, acc, irls_ck)

        sync()
        rel = (obj_before - float(true_obj_global(torch.cat(x_blocks)).detach().item())) / max(abs(obj_before), 1e-16)
        if prog_n > 0 and acc % prog_n == 0:
            print(f"[{label}] prog acc={acc} rej={rej} obj={float(true_obj_global(torch.cat(x_blocks)).detach().item()):.6e} "
                  f"lam={lam:.2e} wall={(time.perf_counter()-t0)*1000:.0f}ms")
        if abs(rel) < args.rel_obj_tol: status = "CONVERGED_REL_OBJECTIVE"; break
        sc_grad_f = math.sqrt(sum(float(gs[t].norm()**2) for t in range(T))) / max(1.0, max(float(torch.norm(Hs[t])) for t in range(T)))
        if sc_grad_f < args.scaled_grad_tol: status = "CONVERGED_SCALED_GRADIENT"; break
        if float(torch.norm(torch.cat(deltas))) < args.step_tol: status = "CONVERGED_STEP"; break

    if status == "RUNNING": status = "MAX_ACCEPTED_STEPS"
    sync(); xf_global = torch.cat(x_blocks).detach()
    obj_final = float(true_obj_global(xf_global).detach().item()); rms_full = raw_metrics_global(xf_global)

    first_order_checks = {}
    if irls_ck:
        first_order_checks = {"initial": irls_ck[0], "accepted_endpoints": irls_ck[1:] if len(irls_ck) > 1 else [],
                              "worst_abs_max": max(c["abs_max"] for c in irls_ck),
                              "worst_rel_l2": max(c["rel_l2"] for c in irls_ck), "all_pass": all(c["pass"] for c in irls_ck)}

    print(f"[{label}] {status} acc={acc} rej={rej} lins={lin_cnt} slv={chol_ok+chol_fb} "
          f"chol_ok={chol_ok} fb={chol_fb} obj={obj_final:.6e} rms={rms_full['rms']:.1f}")
    return {"label": label, "x": xf_global, "obj": obj_final, "status": status, "accepted_steps": acc,
            "rejected_trials": rej, "solver_jacfwd_call_count": jacfwd_calls, "linearization_count": lin_cnt,
            "solve_count": chol_ok + chol_fb, "cholesky_success_count": chol_ok, "solve_fallback_count": chol_fb,
            "final_lambda": lam, "wall_time_ms": elapsed_ms(t0, xf_global), "rms": rms_full,
            "first_order_checks": first_order_checks}


# ============================================================
# Block LM — Arrowhead (Schur complement) for S2B
# ============================================================
def block_lm_arrowhead(x_blocks, z, frame_dim, beta_dim_val, linearize_fn, true_obj_global,
                        raw_metrics_global, shape_residual_fn,
                        max_acc, max_trials, args, label="LM", validate_fn=None):
    T = len(x_blocks); dim_l = frame_dim
    sync = (lambda: torch.cuda.synchronize()) if (args.profile_timing and torch.cuda.is_available()) else (lambda: None)

    sync(); t0 = time.perf_counter()
    lin_result = linearize_fn(x_blocks, z)
    A_lst, B_lst, g_lst = lin_result["A_lst"], lin_result["B_lst"], lin_result["g_lst"]
    Cc, gzc = lin_result["Cc"], lin_result["gzc"]
    C_shape, gz_shape = lin_result["C_shape"], lin_result["gz_shape"]
    jacfwd_calls = lin_result.get("jacfwd_calls", 2 * T + 1); lin_cnt = 1

    lam = max(float(torch.diag(A_t).max()) for A_t in A_lst) * 1e-3
    acc, rej = 0, 0; local_chol_ok, local_chol_fb, schur_chol_ok, schur_chol_fb = 0, 0, 0, 0
    status, irls_ck = "RUNNING", []; prog_n = args.progress_every_accepted

    for outer in range(max_acc):
        sync(); obj_before = float(true_obj_global(torch.cat([*x_blocks, z])).detach().item())
        accepted_this = False
        for trial in range(max_trials):
            invA_Bs, invA_gs, ok = [], [], True
            for t in range(T):
                A_hat = A_lst[t] + lam * torch.eye(dim_l, dtype=torch.float64)
                try:
                    L_ch, info = torch.linalg.cholesky_ex(A_hat)
                    if info.item(): raise RuntimeError
                    invA_Bs.append(torch.cholesky_solve(B_lst[t], L_ch))
                    invA_gs.append(torch.cholesky_solve(g_lst[t].unsqueeze(1), L_ch).squeeze(1)); local_chol_ok += 1
                except Exception:
                    try:
                        invA_Bs.append(torch.linalg.solve(A_hat, B_lst[t]))
                        invA_gs.append(torch.linalg.solve(A_hat, g_lst[t].unsqueeze(1)).squeeze(1)); local_chol_fb += 1
                    except Exception: ok = False; break
            if not ok: lam = min(lam * 10, 1e12); rej += 1; continue

            S = sum(Cc) + C_shape + lam * torch.eye(beta_dim_val, dtype=torch.float64)
            q = sum(gzc) + gz_shape
            for t in range(T): S = S - B_lst[t].T @ invA_Bs[t]; q = q - B_lst[t].T @ invA_gs[t]
            try:
                L_s, info = torch.linalg.cholesky_ex(S)
                if info.item(): raise RuntimeError
                delta_z = torch.cholesky_solve((-q).unsqueeze(1), L_s).squeeze(1); schur_chol_ok += 1
            except Exception:
                try: delta_z = torch.linalg.solve(S, (-q).unsqueeze(1)).squeeze(1); schur_chol_fb += 1
                except Exception: lam = min(lam * 10, 1e12); rej += 1; continue

            deltas = [-(invA_gs[t] + invA_Bs[t] @ delta_z) for t in range(T)]
            x_trial = [(x_blocks[t] + deltas[t].to(torch.float32)).detach() for t in range(T)]
            z_trial = (z + delta_z.to(torch.float32)).detach()
            sync(); obj_trial = float(true_obj_global(torch.cat([*x_trial, z_trial])).detach().item())
            pred = sum(float(-g_lst[t].dot(deltas[t]) - 0.5 * (deltas[t] @ A_lst[t] @ deltas[t]) - (deltas[t] @ B_lst[t] @ delta_z)) for t in range(T))
            pred += float(-(sum(gzc) + gz_shape).dot(delta_z) - 0.5 * (delta_z @ (sum(Cc) + C_shape) @ delta_z))
            actual = obj_before - obj_trial; rho = actual / max(pred, 1e-12) if pred > 0 else 0.0
            if pred > 0 and actual > 0 and rho > 0:
                x_blocks = x_trial; z = z_trial; acc += 1; accepted_this = True
                lam = max(lam / 3, 1e-10) if rho > 0.75 else (min(lam * 3, 1e12) if rho < 0.25 else lam)
                break
            else: lam = min(lam * 10, 1e12); rej += 1

        if not accepted_this:
            gn2 = sum(float(g_lst[t].norm() ** 2) for t in range(T)) + float((sum(gzc) + gz_shape).norm() ** 2)
            sg = math.sqrt(gn2) / max(1.0, max(max(float(torch.norm(A_lst[t])) for t in range(T)), float(torch.norm(sum(Cc) + C_shape))))
            status = "CONVERGED_SCALED_GRADIENT" if (acc > 4 and sg < 1e-4) else "LM_NO_ACCEPTED_TRIAL"; break

        lin_result = linearize_fn(x_blocks, z)
        A_lst, B_lst, g_lst = lin_result["A_lst"], lin_result["B_lst"], lin_result["g_lst"]
        Cc, gzc = lin_result["Cc"], lin_result["gzc"]
        C_shape, gz_shape = lin_result["C_shape"], lin_result["gz_shape"]
        jacfwd_calls += lin_result.get("jacfwd_calls", 2 * T + 1); lin_cnt += 1
        if validate_fn: validate_fn(x_blocks, z, lin_result, acc, irls_ck)

        sync()
        rel = (obj_before - float(true_obj_global(torch.cat([*x_blocks, z])).detach().item())) / max(abs(obj_before), 1e-16)
        if prog_n > 0 and acc % prog_n == 0:
            print(f"[{label}] prog acc={acc} rej={rej} obj={float(true_obj_global(torch.cat([*x_blocks,z])).detach().item()):.6e} "
                  f"lam={lam:.2e} wall={(time.perf_counter()-t0)*1000:.0f}ms")
        if abs(rel) < args.rel_obj_tol: status = "CONVERGED_REL_OBJECTIVE"; break
        if max(float(d.abs().max()) for d in deltas) < args.step_tol: status = "CONVERGED_STEP"; break

    if status == "RUNNING": status = "MAX_ACCEPTED_STEPS"
    sync(); xf_global = torch.cat([*x_blocks, z]).detach()
    obj_final = float(true_obj_global(xf_global).detach().item()); rms_full = raw_metrics_global(xf_global)

    first_order_checks = {}
    if irls_ck:
        first_order_checks = {"initial": irls_ck[0], "accepted_endpoints": irls_ck[1:] if len(irls_ck) > 1 else [],
                              "worst_abs_max": max(c["abs_max"] for c in irls_ck),
                              "worst_rel_l2": max(c["rel_l2"] for c in irls_ck), "all_pass": all(c["pass"] for c in irls_ck)}

    print(f"[{label}] {status} acc={acc} rej={rej} lins={lin_cnt} "
          f"local_chol={local_chol_ok} local_fb={local_chol_fb} "
          f"schur_chol={schur_chol_ok} schur_fb={schur_chol_fb} obj={obj_final:.6e} rms={rms_full['rms']:.1f}")
    return {"label": label, "x": xf_global, "obj": obj_final, "status": status, "accepted_steps": acc,
            "rejected_trials": rej, "solver_jacfwd_call_count": jacfwd_calls, "linearization_count": lin_cnt,
            "wall_time_ms": elapsed_ms(t0, xf_global), "rms": rms_full, "first_order_checks": first_order_checks,
            "schur": {"frame_count": T, "local_dim": dim_l, "beta_dim": beta_dim_val,
                      "local_cholesky_success_count": local_chol_ok, "local_solve_fallback_count": local_chol_fb,
                      "schur_cholesky_success_count": schur_chol_ok, "schur_solve_fallback_count": schur_chol_fb}}


# ============================================================
# Per-frame validation helper (shared by S2A and S2B)
# ============================================================
def _make_perframe_validator(true_obj, surrogates_fn, compute_irls_grad_fn, label, args, impl):
    """surrogates_fn(result) -> [(surrogate, p_t) ...]; compute_irls_grad_fn(p_t, surrogate, args, impl) -> g_t."""

    def validate(x_blocks, lin_result, acc_val, irls_ck_list):
        # Use single-block gradient check
        g_block_parts = []
        for t, (surr, p_t) in enumerate(surrogates_fn(lin_result, x_blocks)):
            g_block_parts.append(compute_irls_grad_fn(p_t.detach(), surr))
        ck = validate_block_gradient(true_obj, x_blocks, lambda: torch.cat(g_block_parts),
                                      f"{label}_acc{acc_val}", args)
        irls_ck_list.append(ck)
        if not ck["pass"]: raise RuntimeError(f"BLOCK_IRLS_FIRST_ORDER_MISMATCH {label} outer={acc_val}")
    return validate


def _make_perframe_arrowhead_validator(true_obj, surrogates_fn, shape_res_fn, label, args, impl):
    """surrogates_fn(result, x_blocks, z) -> list of (surrogate, p_t, t)"""

    def validate(x_blocks, z_val, lin_result, acc_val, irls_ck_list):
        g_l_parts, gz_from_frames = [], torch.zeros(z_val.numel(), dtype=torch.float64)
        for t, (surr, l_t) in enumerate(surrogates_fn(lin_result, x_blocks)):
            J_l, J_z, r_t = _linearize_dual(surr, l_t.detach(), z_val.detach(), impl)
            Jl64, Jz64, r64 = J_l.to(torch.float64), J_z.to(torch.float64), r_t.to(torch.float64)
            g_l_parts.append(Jl64.T @ r64); gz_from_frames += Jz64.T @ r64
        J_s = jacfwd(shape_res_fn)(z_val.detach().clone().requires_grad_(True)).detach().to(torch.float64)
        r_s = shape_res_fn(z_val).detach().to(torch.float64)
        g_block = torch.cat([*g_l_parts, gz_from_frames + J_s.T @ r_s])
        x_all = torch.cat([*[xb.detach().clone().requires_grad_(True) for xb in x_blocks],
                            z_val.detach().clone().requires_grad_(True)])
        g_true = torch.autograd.grad(true_obj(x_all), x_all, create_graph=False)[0].detach().to(torch.float64)
        diff = g_true - g_block.to(torch.float64)
        abs_max = float(diff.abs().max()); rel_l2 = float(diff.norm() / max(g_true.norm().item(), 1.0))
        ok = rel_l2 <= args.irls_gradient_rel_tol
        print(f"[BLOCK_IRLS {label}_acc{acc_val}] abs_max={abs_max:.2e} rel_l2={rel_l2:.2e} {'PASS' if ok else 'FAIL'}")
        irls_ck_list.append({"abs_max": abs_max, "rel_l2": rel_l2, "pass": ok})
        if not ok: raise RuntimeError(f"BLOCK_IRLS_FIRST_ORDER_MISMATCH {label} outer={acc_val}")
    return validate


def _run_perframe_init_validation(true_obj, x_blocks, make_lin_fn, compute_irls_grad_fn, label, args, impl):
    """Run init first-order validation for per-frame mode. Returns (ck_init, irc_list)."""
    init_surrogates = [make_lin_fn(t)(x_blocks[t])["surrogate_fn"] for t in range(len(x_blocks))]
    def compute_gn():
        parts = []
        for t, surr in enumerate(init_surrogates):
            parts.append(compute_irls_grad_fn(x_blocks[t].detach(), surr))
        return torch.cat(parts)
    ck = validate_block_gradient(true_obj, x_blocks, compute_gn, f"{label}_init", args)
    if not ck["pass"]: raise RuntimeError(f"BLOCK_IRLS_FIRST_ORDER_MISMATCH {label}")
    foc = {"initial": ck, "accepted_endpoints": []}
    foc.update(worst_abs_max=ck["abs_max"], worst_rel_l2=ck["rel_l2"], all_pass=ck["pass"])
    return ck, [ck]


# ============================================================
# Handoff helpers
# ============================================================
def handoff_s1_to_s2a(x_s1, bp_anchors, T):
    parts = []; [parts.extend([x_s1[t * 12:(t + 1) * 12], bp_anchors[t].clone()]) for t in range(T)]
    return torch.cat(parts).clone()


def check_handoff_s1_to_s2a(x_s1, x_s2a, bp_anchors, T):
    pf = S1_PF + POSE_PF; ot_diff, bp_diff = 0.0, 0.0
    for t in range(T):
        ot_diff = max(ot_diff, float((x_s2a[t * pf:t * pf + 12] - x_s1[t * 12:(t + 1) * 12]).abs().max()))
        bp_diff = max(bp_diff, float((x_s2a[t * pf + 12:(t + 1) * pf] - bp_anchors[t]).abs().max()))
    return {"ot_max": ot_diff, "bp_max": bp_diff, "ok": ot_diff < 1e-7 and bp_diff < 1e-7}


def check_handoff_s2a_to_s2b(x_s2a, l_blocks, z, beta_anchor, beta_abs_limit, sa_scale,
                               raw_fn_s2a_fn, raw_fn_s2b_fn, true_obj_s2a_fn, true_obj_s2b_fn):
    T = len(l_blocks); pf_dim = S1_PF + POSE_PF
    l_global = torch.cat(l_blocks); state_max = float((l_global - x_s2a).abs().max())
    beta_init = beta_abs_limit * torch.tanh(z); bp_max = float((beta_init - beta_anchor).abs().max())
    x_s2b_global = torch.cat([l_global, z])
    raw_diff = float((flatten_groups(raw_fn_s2b_fn(x_s2b_global)) - flatten_groups(raw_fn_s2a_fn(x_s2a))).abs().max())
    sa_obj = float(0.5 * torch.sum((beta_init - beta_anchor) ** 2 * sa_scale ** 2))
    obj_s2a = float(true_obj_s2a_fn(x_s2a).detach().item())
    obj_s2b = float(true_obj_s2b_fn(x_s2b_global).detach().item())
    rel_obj = abs(obj_s2b - obj_s2a) / max(abs(obj_s2a), 1e-16)
    ok = (state_max <= 1e-7 and bp_max <= 1e-6 and raw_diff <= 1e-3 and sa_obj <= 1e-10 and rel_obj <= 1e-6)
    return {"state_max_diff": state_max, "bp_init_max_diff": bp_max, "raw_reproj_max_diff": raw_diff,
            "shape_anchor_obj_init": sa_obj, "rel_obj_s2b_vs_s2a": rel_obj, "ok": ok}


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    frame_list = [int(s.strip()) for s in args.frames.split(",")]
    T = len(frame_list); W, H = args.image_width, args.image_height
    out_d = Path(args.output_dir); eps = args.charbonnier_eps

    rrc = Path(config_script.PROJECT_ROOT) / "configs" / "rotate_upright_rules.json"
    rotate_rules = resolve_dataset_rotate_rules(rrc, Path(args.bbox_dir))
    hi = set(); obs_list = []; dh2, dh4 = {}, {}
    for fidx in frame_list:
        obs_list.append(load_observations(fidx, Path(args.sam3d_dense_dir), Path(args.bbox_dir), W, H, rotate_rules))
        dh2[fidx] = try_load_smplx_init(fidx, Path(args.hmr_dir), hi, "cam2")
        dh4[fidx] = try_load_smplx_init(fidx, Path(args.hmr_dir), hi, "cam4")
    sp2, sp4 = [], []
    for fidx in frame_list:
        sp2.append(build_hmr_init_for_frame(fidx, frame_list, dh2)[0])
        sp4.append(build_hmr_init_for_frame(fidx, frame_list, dh4)[0])
    si2, si4 = build_segment_smplx_init(sp2, sp4)
    fitter = DualCamERPFitter(primary_observation_source="sam3d_dense", gender="neutral", R_cam4=R_CAM4)
    sc2, sc4, sbeta, sbpose = fitter._init_params(si2, si4, [o[0] for o in obs_list], [o[1] for o in obs_list], (W, H))
    body_pose_dim = int(sbpose.shape[-1]); beta_dim = int(sbeta.numel())
    bp_anchors = [sbpose.detach().clone()[t].reshape(-1) for t in range(T)]
    beta_anchor = sbeta.detach().clone().reshape(-1)
    model = fitter._get_model(); frozen_list = [build_frozen_params(sc2, sc4, t) for t in range(T)]
    sample_ids = fitter._sam3d_dense_sample_ids
    info_list = [precompute_valid_points(obs_list[t][0], obs_list[t][1], sample_ids) for t in range(T)]
    vid_c2_l, tgt_c2_l, vid_c4_l, tgt_c4_l = [], [], [], []
    for t in range(T):
        i2, i4 = info_list[t]
        for lst, idx in [(vid_c2_l, 0), (tgt_c2_l, 0), (vid_c4_l, 1), (tgt_c4_l, 1)]:
            lst.append(torch.as_tensor(info_list[t][idx]["valid_ids" if idx == 0 else "target_2d"].astype(np.int64 if idx == 0 else np.float32)))

    beta_frozen = beta_anchor; pf_dim = S1_PF + body_pose_dim
    pa_scale = math.sqrt(2.0 * args.pose_anchor_weight / (T * body_pose_dim))
    sa_scale = math.sqrt(2.0 * args.shape_anchor_weight / beta_dim)
    if float(beta_anchor.abs().max()) >= args.beta_abs_limit * (1.0 - BETA_ATANH_EPS):
        raise RuntimeError("INFEASIBLE_BETA_ANCHOR")
    temporal_jvp = (args.temporal_jacobian_impl == "batched_jvp")
    N_fixed = len(sample_ids); sid_t = torch.as_tensor(sample_ids.astype(np.int64))
    view_fwd = args.view_forward_impl; impl = args.linearization_impl
    do_iv = args.validate_irls_first_order; do_v = args.validate_irls_each_accepted

    # --- Factory: stage objective/metrics ---
    def _make_stage_objs(stride, beta_src="frozen", bp_frozen_list=None):
        def raw_groups(x_global):
            groups = []
            for t in range(T):
                x_t = x_global[t * stride:(t + 1) * stride]
                bp = bp_frozen_list[t] if bp_frozen_list is not None else None
                b = args.beta_abs_limit * torch.tanh(x_global[stride * T:]) if beta_src == "from_z" else beta_frozen
                groups.extend(make_raw_reproj_for_frame(x_t, frame_list[t], vid_c2_l[t], tgt_c2_l[t],
                    vid_c4_l[t], tgt_c4_l[t], frozen_list[t], model, fitter, W, H, stride, b,
                    bp_frozen=bp, view_forward_impl=view_fwd))
            return groups

        def true_obj(x_global):
            obj = compute_primary_charbonnier_tensor(raw_groups(x_global), eps, static_spec)
            pa_parts = [(torch.zeros(body_pose_dim, device=x_global.device, dtype=x_global.dtype) if bp_frozen_list is not None
                         else (x_global[t * stride + 12:t * stride + stride] - bp_anchors[t]) * pa_scale)
                        for t in range(T)]
            obj = obj + 0.5 * torch.sum(torch.cat(pa_parts) ** 2)
            if beta_src == "from_z":
                beta_s = args.beta_abs_limit * torch.tanh(x_global[stride * T:])
                obj = obj + 0.5 * torch.sum((beta_s - beta_anchor) ** 2 * sa_scale ** 2)
            return obj

        def raw_metrics(x_global):
            return raw_reproj_metrics(flatten_groups(raw_groups(x_global)).detach())
        return raw_groups, true_obj, raw_metrics

    # --- Prepare temporal JVP data ---
    bp_anchors_tensor = torch.stack(bp_anchors)
    beta_anchor_tensor = beta_anchor.unsqueeze(0)

    if temporal_jvp:
        targets_c2_pad, masks_c2, targets_c4_pad, masks_c4 = [], [], [], []
        for t in range(T):
            for lst, i_idx in [(targets_c2_pad, 0), (masks_c2, 0), (targets_c4_pad, 1), (masks_c4, 1)]:
                p, m = _prepare_padded_targets(torch.as_tensor(info_list[t][i_idx]["valid_ids"]),
                                                torch.as_tensor(info_list[t][i_idx]["target_2d"]).float(), N_fixed, sid_t)
                lst.append(p if lst in (targets_c2_pad, targets_c4_pad) else m)
        targets_c2_pad = torch.stack(targets_c2_pad); masks_c2 = torch.stack(masks_c2)
        targets_c4_pad = torch.stack(targets_c4_pad); masks_c4 = torch.stack(masks_c4)

        lh_c2 = torch.cat([fl["lhand_c2"] for fl in frozen_list], dim=0)
        lh_c4 = torch.cat([fl["lhand_c4"] for fl in frozen_list], dim=0)
        rh_c2 = torch.cat([fl["rhand_c2"] for fl in frozen_list], dim=0)
        rh_c4 = torch.cat([fl["rhand_c4"] for fl in frozen_list], dim=0)
        sc_c2 = torch.cat([fl["scale_c2"].expand(1) for fl in frozen_list], dim=0)
        sc_c4 = torch.cat([fl["scale_c4"].expand(1) for fl in frozen_list], dim=0)
        lhand_2T = torch.cat([lh_c2, lh_c4], dim=0); rhand_2T = torch.cat([rh_c2, rh_c4], dim=0)
        scale_2T = torch.cat([sc_c2, sc_c4], dim=0)
        print(f"[TEMPORAL] N_fixed={N_fixed} batch_size={2*T} temporal_jvp=batched_jvp")

        # Helper: build temporal single-stage linearizer
        def _temporal_single_linearizer(stage, stride, dim, raw_fn, N, masks_c2, masks_c4):
            sw2 = torch.zeros(T, N, device=torch.tensor(0.0).device if T == 0 else x0_s1_blocks[0].device); sw4 = sw2.clone()
            def refresh(L):
                refresh_temporal_irls_weights(L, None, stage, raw_fn, static_spec, eps, N, masks_c2, masks_c4, sw2, sw4, T)
            batched_fn = _build_batched_residual(T, W, H, sw2.device, bp_anchors_tensor, beta_anchor_tensor,
                lhand_2T, rhand_2T, scale_2T, targets_c2_pad, masks_c2, sw2, targets_c4_pad, masks_c4, sw4,
                model, fitter, sample_ids, N, stride, pa_scale, stage=stage)
            def linearize(x_blocks):
                L = torch.stack(x_blocks); refresh(L)
                r, J_blocks, jc = _temporal_jvp_local(batched_fn, L.detach(), dim, T)
                return {"Hs": [Jb.to(torch.float64).T @ Jb.to(torch.float64) for Jb in J_blocks],
                        "gs": [Jb.to(torch.float64).T @ r[t].to(torch.float64) for t, Jb in enumerate(J_blocks)],
                        "jacfwd_calls": jc}
            return linearize, sw2.device if T == 0 else x0_s1_blocks[0].device

    # --- Initial params ---
    x0_s1_blocks = [torch.cat([sc2.orient.detach()[t].reshape(-1), sc2.transl.detach()[t].reshape(-1),
                                sc4.orient.detach()[t].reshape(-1), sc4.transl.detach()[t].reshape(-1)]).clone()
                    for t in range(T)]
    s1_dim = T * S1_PF

    # Build static aggregation spec (frozen at init params for S1)
    raw_groups_s1, _, _ = _make_stage_objs(S1_PF, bp_frozen_list=bp_anchors)
    x0_s1_global = torch.cat(x0_s1_blocks)
    static_spec = build_primary_aggregation_spec(raw_groups_s1(x0_s1_global))
    get_raw_s1, true_obj_s1, raw_metrics_s1 = _make_stage_objs(S1_PF, bp_frozen_list=bp_anchors)

    # ===== STAGE 1 =====
    results = {"config": vars(args), "handoffs": {}}

    if temporal_jvp:
        s1_lin, _ = _temporal_single_linearizer("S1", S1_PF, S1_PF, get_raw_s1, N_fixed, masks_c2, masks_c4)
        s1_lm = block_lm_single(x0_s1_blocks, [S1_PF] * T, s1_lin, true_obj_s1, raw_metrics_s1,
                                 args.stage1_lm_max_acc, args.lm_max_trials, args, label="S1_TEMPORAL")
        results["stage1"] = {"lm": {k: v for k, v in s1_lm.items() if k != "x"},
            "input_summary": {"T": T, "param_dim": s1_dim, "temporal_batch_size": 2 * T,
                              "fixed_residual_dim": 4 * N_fixed}, "verdict": "TEMPORAL_S1_COMPLETE"}
    else:
        raise NotImplementedError("Per-frame S1 without temporal JVP not in unified path")

    # ===== STAGE 2A =====
    s2a_dim = T * pf_dim
    x0_s2a_global = handoff_s1_to_s2a(s1_lm["x"], bp_anchors, T)
    x0_s2a_blocks = [x0_s2a_global[t * pf_dim:(t + 1) * pf_dim].clone() for t in range(T)]
    hoff_s1 = check_handoff_s1_to_s2a(s1_lm["x"], x0_s2a_global, bp_anchors, T)
    print(f"[HANDOFF_S1_S2A] ot={hoff_s1['ot_max']:.2e} bp={hoff_s1['bp_max']:.2e} ok={hoff_s1['ok']}")
    results["handoffs"]["lm_s1_to_s2a"] = hoff_s1
    if not hoff_s1["ok"]: raise RuntimeError("Block handoff S1->S2A failed")

    get_raw_s2a, true_obj_s2a, raw_metrics_s2a = _make_stage_objs(pf_dim)

    def _make_s2a_lin_fn(t):
        def raw_fn_t(x_t):
            return make_raw_reproj_for_frame(x_t, frame_list[t], vid_c2_l[t], tgt_c2_l[t],
                     vid_c4_l[t], tgt_c4_l[t], frozen_list[t], model, fitter, W, H, pf_dim, beta_frozen,
                     view_forward_impl=view_fwd)
        return lambda x: make_irls_linearization(raw_fn_t, lambda y, ft=t: (y[12:pf_dim] - bp_anchors[ft]) * pa_scale, x, eps, static_spec)
    s2a_make_lin_fns = [_make_s2a_lin_fn(t) for t in range(T)]
    s2a_dims = [pf_dim] * T

    if temporal_jvp:
        s2a_lin, _ = _temporal_single_linearizer("S2A", pf_dim, pf_dim, get_raw_s2a, N_fixed, masks_c2, masks_c4)
        s2a_lm = block_lm_single(x0_s2a_blocks, s2a_dims, s2a_lin, true_obj_s2a, raw_metrics_s2a,
                                  args.stage2a_lm_max_acc, args.lm_max_trials, args, label="S2A_TEMPORAL")
        results["stage2a"] = {"lm": {k: v for k, v in s2a_lm.items() if k != "x"},
            "input_summary": {"T": T, "param_dim": s2a_dim, "temporal_batch_size": 2 * T,
                              "fixed_residual_dim": 4 * N_fixed + POSE_PF}, "verdict": "TEMPORAL_S2A_COMPLETE"}
    else:
        # Per-frame: inline linearizer
        frame_lins = [None] * T
        def pf_linearize(x_blocks):
            Hs, gs = [], []
            for t in range(T):
                lin_t = s2a_make_lin_fns[t](x_blocks[t]); frame_lins[t] = lin_t
                J_t, r_t = _linearize_single(lin_t["surrogate_fn"], x_blocks[t].detach(), impl)
                Hs.append(J_t.to(torch.float64).T @ J_t.to(torch.float64))
                gs.append(J_t.to(torch.float64).T @ r_t.to(torch.float64))
            return {"Hs": Hs, "gs": gs, "jacfwd_calls": T, "frame_lins": frame_lins}

        if do_v:
            s2a_validate = _make_perframe_validator(true_obj_s2a,
                lambda r, xb: [(r["frame_lins"][i]["surrogate_fn"], xb[i]) for i in range(T)],
                lambda pt, surr: _linearize_single(surr, pt, impl)[0].to(torch.float64).T @ _linearize_single(surr, pt, impl)[1].to(torch.float64),
                "S2A", args, impl)
        else:
            s2a_validate = None

        s2a_lm = block_lm_single(x0_s2a_blocks, s2a_dims, pf_linearize, true_obj_s2a, raw_metrics_s2a,
                                  args.stage2a_lm_max_acc, args.lm_max_trials, args, label="S2A_BLOCK",
                                  validate_fn=s2a_validate)

        if do_iv:
            ck_init, irc = _run_perframe_init_validation(true_obj_s2a, x0_s2a_blocks, _make_s2a_lin_fn,
                lambda pt, surr: _linearize_single(surr, pt, impl)[0].to(torch.float64).T @ _linearize_single(surr, pt, impl)[1].to(torch.float64),
                "S2A", args, impl)
            foc = s2a_lm.get("first_order_checks") or {}
            combined = list(irc) + (foc.get("accepted_endpoints") or [])
            s2a_lm["first_order_checks"] = {"initial": ck_init, "accepted_endpoints": combined[1:] if len(combined) > 1 else [],
                "worst_abs_max": max(c["abs_max"] for c in combined),
                "worst_rel_l2": max(c["rel_l2"] for c in combined), "all_pass": all(c["pass"] for c in combined)}
        results["stage2a"] = {"lm": {k: v for k, v in s2a_lm.items() if k != "x"},
            "input_summary": {"T": T, "param_dim": s2a_dim,
                "smplx_forward_call_count": T * s2a_lm["linearization_count"]}, "verdict": "BLOCK_S2A_COMPLETE"}

    if args.stop_after_stage == "stage2a":
        _save(out_d, frame_list, results, x0_s1=x0_s1_global, x_s1_lm=s1_lm["x"],
              x0_s2a=x0_s2a_global, x_s2a_lm=s2a_lm["x"], beta_anchor=beta_anchor)
        raise SystemExit(0)

    # ===== STAGE 2B =====
    s2b_dim = s2a_dim + beta_dim
    beta_latent = torch.atanh(beta_anchor / args.beta_abs_limit)
    x0_l_blocks = [s2a_lm["x"][t * pf_dim:(t + 1) * pf_dim].clone() for t in range(T)]
    z0 = beta_latent.clone()

    get_raw_s2b, true_obj_s2b, raw_metrics_s2b = _make_stage_objs(pf_dim, beta_src="from_z")
    s2b_shape_res = lambda z_: (args.beta_abs_limit * torch.tanh(z_) - beta_anchor) * sa_scale

    hoff_s2 = check_handoff_s2a_to_s2b(s2a_lm["x"], x0_l_blocks, z0, beta_anchor, args.beta_abs_limit, sa_scale,
        get_raw_s2a, get_raw_s2b, true_obj_s2a, true_obj_s2b)
    print(f"[HANDOFF_S2A_S2B] state={hoff_s2['state_max_diff']:.2e} bp={hoff_s2['bp_init_max_diff']:.2e} "
          f"raw={hoff_s2['raw_reproj_max_diff']:.2e} sa={hoff_s2['shape_anchor_obj_init']:.2e} "
          f"rel={hoff_s2['rel_obj_s2b_vs_s2a']:.2e} ok={hoff_s2['ok']}")
    results["handoffs"]["lm_s2a_to_s2b"] = hoff_s2
    if not hoff_s2["ok"]: raise RuntimeError("Block handoff S2A->S2B failed")

    def _make_s2b_lin_fn(t):
        def raw_fn_t(lt, zv):
            return make_raw_reproj_for_frame(lt, frame_list[t], vid_c2_l[t], tgt_c2_l[t],
                     vid_c4_l[t], tgt_c4_l[t], frozen_list[t], model, fitter, W, H, pf_dim,
                     args.beta_abs_limit * torch.tanh(zv))
        return lambda l, z: make_irls_linearization(raw_fn_t, lambda y, ft=t: (y[12:pf_dim] - bp_anchors[ft]) * pa_scale,
                                                     l, eps, static_spec, z=z)
    s2b_make_lin_fns = [_make_s2b_lin_fn(t) for t in range(T)]
    print(f"\n{'='*60}\n[S2B_BLOCK] T={T} param_dim={s2b_dim} local_dim={pf_dim} beta_dim={beta_dim}\n{'='*60}")

    if temporal_jvp:
        sw2 = torch.zeros(T, N_fixed, device=x0_l_blocks[0].device); sw4 = sw2.clone()
        def s2b_refresh(L, z_val):
            refresh_temporal_irls_weights(L, z_val, "S2B", get_raw_s2b, static_spec, eps,
                N_fixed, masks_c2, masks_c4, sw2, sw4, T)
        batched_fn_s2b = _build_batched_residual(T, W, H, sw2.device, bp_anchors_tensor, beta_anchor_tensor,
            lhand_2T, rhand_2T, scale_2T, targets_c2_pad, masks_c2, sw2, targets_c4_pad, masks_c4, sw4,
            model, fitter, sample_ids, N_fixed, pf_dim, pa_scale, stage="S2B", beta_abs_limit=args.beta_abs_limit)
        def s2b_linearize(x_blocks, z_val):
            L = torch.stack(x_blocks).detach(); s2b_refresh(L, z_val)
            fn_L = lambda L_: batched_fn_s2b(L_, z_val)
            r, Jl_blocks, jc_l = _temporal_jvp_local(fn_L, L.detach(), pf_dim, T)
            Jb_blocks, jc_b = _temporal_jvp_beta(batched_fn_s2b, L.detach(), z_val.detach(), beta_dim, T)
            A_lst, B_lst, g_lst, Cc, gzc = [], [], [], [], []
            for t in range(T):
                Jl64 = Jl_blocks[t].to(torch.float64); Jb64 = Jb_blocks[t].to(torch.float64); r64 = r[t].to(torch.float64)
                A_lst.append(Jl64.T @ Jl64); B_lst.append(Jl64.T @ Jb64); g_lst.append(Jl64.T @ r64)
                Cc.append(Jb64.T @ Jb64); gzc.append(Jb64.T @ r64)
            J_s = jacfwd(s2b_shape_res)(z_val.detach().clone().requires_grad_(True)).detach().to(torch.float64)
            r_s = s2b_shape_res(z_val).detach().to(torch.float64)
            return {"A_lst": A_lst, "B_lst": B_lst, "g_lst": g_lst, "Cc": Cc, "gzc": gzc,
                    "C_shape": J_s.T @ J_s, "gz_shape": J_s.T @ r_s, "jacfwd_calls": jc_l + jc_b}
        s2b_lm = block_lm_arrowhead(x0_l_blocks, z0, pf_dim, beta_dim, s2b_linearize,
                                     true_obj_s2b, raw_metrics_s2b, s2b_shape_res,
                                     args.stage2b_lm_max_acc, args.lm_max_trials, args, label="S2B_TEMPORAL")
    else:
        # Per-frame: inline arrowhead linearizer
        pf_alins = [None] * T
        def pf_arrow_linearize(x_blocks, z_val):
            A_lst, B_lst, g_lst, Cc, gzc = [], [], [], [], []
            for t in range(T):
                lin_t = s2b_make_lin_fns[t](x_blocks[t], z_val); pf_alins[t] = lin_t
                J_l, J_z, r_t = _linearize_dual(lin_t["surrogate_fn"], x_blocks[t].detach(), z_val.detach(), impl)
                Jl64, Jz64, r64 = J_l.to(torch.float64), J_z.to(torch.float64), r_t.to(torch.float64)
                A_lst.append(Jl64.T @ Jl64); B_lst.append(Jl64.T @ Jz64); g_lst.append(Jl64.T @ r64)
                Cc.append(Jz64.T @ Jz64); gzc.append(Jz64.T @ r64)
            J_s = jacfwd(s2b_shape_res)(z_val.detach().clone().requires_grad_(True)).detach().to(torch.float64)
            r_s = s2b_shape_res(z_val).detach().to(torch.float64)
            return {"A_lst": A_lst, "B_lst": B_lst, "g_lst": g_lst, "Cc": Cc, "gzc": gzc,
                    "C_shape": J_s.T @ J_s, "gz_shape": J_s.T @ r_s,
                    "jacfwd_calls": T * (1 if impl == "fused" else 2), "frame_lins": pf_alins}

        if do_v:
            s2b_validate = _make_perframe_arrowhead_validator(true_obj_s2b,
                lambda r, xb: [(r["frame_lins"][i]["surrogate_fn"], xb[i]) for i in range(T)],
                s2b_shape_res, "S2B", args, impl)
        else:
            s2b_validate = None

        s2b_lm = block_lm_arrowhead(x0_l_blocks, z0, pf_dim, beta_dim, pf_arrow_linearize,
                                     true_obj_s2b, raw_metrics_s2b, s2b_shape_res,
                                     args.stage2b_lm_max_acc, args.lm_max_trials, args, label="S2B_BLOCK",
                                     validate_fn=s2b_validate)

        if do_iv:
            # Init validation (must duplicate structure since _make_perframe_arrowhead_validator is complex)
            s2b_lin_vals = [fn(x0_l_blocks[t], z0) for t, fn in enumerate(s2b_make_lin_fns)]
            g_l_parts, gz_frames = [], torch.zeros(z0.numel(), dtype=torch.float64)
            for t in range(T):
                J_l, J_z, r_t = _linearize_dual(s2b_lin_vals[t]["surrogate_fn"], x0_l_blocks[t].detach(), z0.detach(), impl)
                Jl64, Jz64, r64 = J_l.to(torch.float64), J_z.to(torch.float64), r_t.to(torch.float64)
                g_l_parts.append(Jl64.T @ r64); gz_frames += Jz64.T @ r64
            J_s = jacfwd(s2b_shape_res)(z0.detach().clone().requires_grad_(True)).detach().to(torch.float64)
            r_s = s2b_shape_res(z0).detach().to(torch.float64)
            g_block_init = torch.cat([*g_l_parts, gz_frames + J_s.T @ r_s])
            x_global_init = torch.cat([*[xb.detach().clone().requires_grad_(True) for xb in x0_l_blocks],
                                        z0.detach().clone().requires_grad_(True)])
            g_true_init = torch.autograd.grad(true_obj_s2b(x_global_init), x_global_init, create_graph=False)[0].detach().to(torch.float64)
            diff_init = g_true_init - g_block_init.to(torch.float64)
            abs_m = float(diff_init.abs().max()); rel_l2 = float(diff_init.norm() / max(g_true_init.norm().item(), 1.0))
            ck_init = {"abs_max": abs_m, "rel_l2": rel_l2, "pass": rel_l2 <= args.irls_gradient_rel_tol}
            print(f"[BLOCK_IRLS S2B_init] abs_max={abs_m:.2e} rel_l2={rel_l2:.2e} {'PASS' if ck_init['pass'] else 'FAIL'}")
            if not ck_init["pass"]: raise RuntimeError("BLOCK_IRLS_FIRST_ORDER_MISMATCH S2B")
            foc = s2b_lm.get("first_order_checks") or {}
            irc = [ck_init] + (foc.get("accepted_endpoints") or [])
            s2b_lm["first_order_checks"] = {"initial": ck_init, "accepted_endpoints": irc[1:] if len(irc) > 1 else [],
                "worst_abs_max": max(c["abs_max"] for c in irc),
                "worst_rel_l2": max(c["rel_l2"] for c in irc), "all_pass": all(c["pass"] for c in irc)}

    # Beta metrics
    beta_final = args.beta_abs_limit * torch.tanh(s2b_lm["x"][s2a_dim:])
    b_metrics = {"anchor_max": float(beta_anchor.abs().max()), "final_max": float(beta_final.abs().max()),
                 "saturation": float(beta_final.abs().max()) / args.beta_abs_limit,
                 "delta_norm": float(torch.norm(beta_final - beta_anchor))}
    results["stage2b"] = {"lm": {k: v for k, v in s2b_lm.items() if k != "x"},
        "input_summary": {"T": T, "param_dim": s2b_dim, "smplx_forward_call_count": T * s2b_lm.get("linearization_count", 18)},
        "beta": b_metrics, "verdict": "BLOCK_S2B_COMPLETE"}

    # ===== Reference parity =====
    if args.dense_reference_npz:
        ref_npz = np.load(args.dense_reference_npz)
        ref_s1 = torch.from_numpy(ref_npz["x_s1_lm"]); ref_s2a = torch.from_numpy(ref_npz["x_s2a_lm"])
        def _parity(label, blk_x, blk_obj, ref_x, true_fn, raw_fn):
            ref_obj = float(true_fn(ref_x).detach().item()); ref_r = raw_fn(ref_x); blk_r = raw_fn(blk_x)
            rel = abs(blk_obj - ref_obj) / max(abs(ref_obj), 1e-16)
            smd = float((blk_x - ref_x).abs().max()); rms_d = abs(blk_r["rms"] - ref_r["rms"])
            med_d = abs(blk_r["median"] - ref_r["median"])
            ok = rel <= 1e-4 and rms_d <= 1e-3 and med_d <= 1e-3
            print(f"[REF_PARITY_{label}] rel_obj={rel:.2e} rms_diff={rms_d:.2e} med_diff={med_d:.2e} state_max={smd:.2e} ok={ok}")
            return {"rel_obj": rel, "rms_diff_px": rms_d, "median_diff_px": med_d, "state_max_diff": smd, "ok": ok}
        rp = {"s1": _parity("S1", s1_lm["x"], s1_lm["obj"], ref_s1, true_obj_s1, raw_metrics_s1),
              "s2a": _parity("S2A", s2a_lm["x"], s2a_lm["obj"], ref_s2a, true_obj_s2a, raw_metrics_s2a)}
        if "x_s2b_lm" in ref_npz:
            ref_s2b = torch.from_numpy(ref_npz["x_s2b_lm"])
            rp["s2b"] = _parity("S2B", s2b_lm["x"], s2b_lm["obj"], ref_s2b, true_obj_s2b, raw_metrics_s2b)
        results["reference_parity"] = rp

    # ===== Endpoint reference =====
    if args.endpoint_reference_npz:
        ref_npz = np.load(args.endpoint_reference_npz)
        ref_s1 = torch.from_numpy(ref_npz["x_s1_lm"]); ref_s2a = torch.from_numpy(ref_npz["x_s2a_lm"])
        ref_s2b = torch.from_numpy(ref_npz["x_s2b_lm"])
        ref_r1, ref_ra, ref_rb = raw_metrics_s1(ref_s1), raw_metrics_s2a(ref_s2a), raw_metrics_s2b(ref_s2b)
        for tag, blk_x, blk_obj, ref_x, true_fn, ref_r, blk_r in [
            ("S1", s1_lm["x"], s1_lm["obj"], ref_s1, true_obj_s1, ref_r1, s1_lm["rms"]),
            ("S2A", s2a_lm["x"], s2a_lm["obj"], ref_s2a, true_obj_s2a, ref_ra, s2a_lm["rms"]),
            ("S2B", s2b_lm["x"], s2b_lm["obj"], ref_s2b, true_obj_s2b, ref_rb, s2b_lm["rms"])]:
            sm = float((blk_x - ref_x).abs().max())
            rel = abs(blk_obj - float(true_fn(ref_x).detach().item())) / max(abs(float(true_fn(ref_x).detach().item())), 1e-16)
            print(f"[ENDPOINT_REF] {tag} state={sm:.2e} obj_rel={rel:.2e} rms={abs(blk_r['rms']-ref_r['rms']):.2e} median={abs(blk_r['median']-ref_r['median']):.2e}")
        ref_beta = args.beta_abs_limit * torch.tanh(ref_s2b[s2a_dim:])
        print(f"[ENDPOINT_REF] beta_diff={float((beta_final-ref_beta).abs().max()):.2e} "
              f"beta_sat={abs(b_metrics['saturation']-float(ref_beta.abs().max())/args.beta_abs_limit):.2e}")

    _save(out_d, frame_list, results,
          x0_s1=x0_s1_global, x_s1_lm=s1_lm["x"], x0_s2a=x0_s2a_global, x_s2a_lm=s2a_lm["x"],
          x0_s2b=torch.cat([*x0_l_blocks, z0]), x_s2b_lm=s2b_lm["x"],
          beta_anchor=beta_anchor, beta_lm=beta_final,
          raw_reproj_s1_lm=flatten_groups(get_raw_s1(s1_lm["x"])).detach().cpu().numpy(),
          raw_reproj_s2a_lm=flatten_groups(get_raw_s2a(s2a_lm["x"])).detach().cpu().numpy(),
          raw_reproj_s2b_lm=flatten_groups(get_raw_s2b(s2b_lm["x"])).detach().cpu().numpy())


def _save(out_d, frame_list, results, **npz_kw):
    seg = f"{frame_list[0]}_{frame_list[-1]}"; out_frame = out_d / seg
    out_frame.mkdir(parents=True, exist_ok=True)
    stage_name = "block_lm_phase2" if "stage2b" in results else "block_lm_phase1"
    with (out_frame / f"{stage_name}.json").open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    npz_save = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in npz_kw.items()}
    np.savez_compressed(str(out_frame / f"{stage_name}.npz"), **npz_save)
    print(f"[SAVE] {out_frame}")


if __name__ == "__main__":
    main()
