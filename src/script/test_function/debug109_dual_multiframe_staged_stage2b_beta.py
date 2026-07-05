"""Staged LM with IRLS Charbonnier production parity. Gate A + Gate B + Gate C."""

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

R_CAM4 = torch.tensor([-1,0,0,0,1,0,0,0,-1], dtype=torch.float32).reshape(3,3)
MIN_VALID_POINTS = 16; S1_PF = 12; POSE_PF = 63; BETA_ATANH_EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Staged LM with IRLS Charbonnier")
    p.add_argument("--frames", type=str, required=True)
    p.add_argument("--hmr_dir", type=str, required=True)
    p.add_argument("--sam3d_dense_dir", type=str, required=True)
    p.add_argument("--bbox_dir", type=str, required=True)
    p.add_argument("--image_width", type=int, default=5760); p.add_argument("--image_height", type=int, default=2880)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--reproj_objective", choices=["raw_l2","charbonnier_irls"], default="charbonnier_irls")
    p.add_argument("--reproj_sigma_px", type=float, default=10.0)
    p.add_argument("--charbonnier_eps", type=float, default=1.0)
    p.add_argument("--pose_anchor_weight", type=float, default=1.0)
    p.add_argument("--beta_abs_limit", type=float, default=2.5)
    p.add_argument("--shape_anchor_weight", type=float, default=10.0)
    p.add_argument("--stop_after_stage", choices=["stage1","stage2a","stage2b"], default="stage2b")
    p.add_argument("--run_lbfgs_oracle", action="store_true", default=False)
    p.add_argument("--validate_irls_first_order", action="store_true", dest="validate_irls_first_order", default=True)
    p.add_argument("--no_validate_irls_first_order", action="store_false", dest="validate_irls_first_order")
    p.add_argument("--validate_irls_each_accepted", action="store_true", default=False)
    p.add_argument("--irls_gradient_rel_tol", type=float, default=5e-4)
    p.add_argument("--stage1_lm_max_acc", type=int, default=30)
    p.add_argument("--stage2a_lm_max_acc", type=int, default=60)
    p.add_argument("--stage2b_lm_max_acc", type=int, default=60)
    p.add_argument("--stage1_bf_outer", type=int, default=15)
    p.add_argument("--stage2a_bf_outer", type=int, default=20)
    p.add_argument("--stage2b_bf_outer", type=int, default=20)
    p.add_argument("--lm_max_trials", type=int, default=8)
    p.add_argument("--rel_obj_tol", type=float, default=1e-7)
    p.add_argument("--scaled_grad_tol", type=float, default=1e-7)
    p.add_argument("--step_tol", type=float, default=1e-7)
    p.add_argument("--lbfgs_max_iter", type=int, default=10)
    p.add_argument("--lbfgs_lr", type=float, default=0.3)
    p.add_argument("--profile_timing", action="store_true", default=False)
    p.add_argument("--progress_every_accepted", type=int, default=0)
    return p.parse_args()


def elapsed_ms(t0, ref=None):
    if ref is not None and ref.is_cuda: torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def load_observations(fidx, sam3d_d, bbox_d, W, H, rotate_rules):
    cb = load_bbox_json(bbox_d / "cam2_bboxes.json"); c4b = load_bbox_json(bbox_d / "cam4_bboxes.json")
    rr = resolve_rotate_rule(fidx, rotate_rules)
    o2 = load_sam3d_dense_result(fidx, sam3d_d, "cam2", W, H, crop_bbox=cb.get(str(fidx)), rotate_rule=rr)
    o4 = load_sam3d_dense_result(fidx, sam3d_d, "cam4", W, H, crop_bbox=c4b.get(str(fidx)), rotate_rule=rr)
    if o2 is None or o4 is None: raise RuntimeError(f"Missing SAM3D dense for frame {fidx}")
    return o2, o4


def build_frozen_params(sc2, sc4, t):
    s2 = sc2.scale.detach().clone().reshape(-1); s4 = sc4.scale.detach().clone().reshape(-1)
    return {"lhand_c2": sc2.left_hand_pose.detach()[t:t+1].clone(),
            "rhand_c2": sc2.right_hand_pose.detach()[t:t+1].clone(),
            "lhand_c4": sc4.left_hand_pose.detach()[t:t+1].clone(),
            "rhand_c4": sc4.right_hand_pose.detach()[t:t+1].clone(),
            "scale_c2": s2[t].reshape(1) if t < s2.shape[0] else s2[0].reshape(1),
            "scale_c4": s4[t].reshape(1) if t < s4.shape[0] else s4[0].reshape(1)}


def precompute_valid_points(obs_c2, obs_c4, sample_ids):
    def _p(obs, cam):
        vid, tgt = select_valid_sam3d_dense_points(obs, sample_ids)
        if int(vid.shape[0]) < MIN_VALID_POINTS: raise RuntimeError(f"{cam}: {vid.shape[0]} < {MIN_VALID_POINTS}")
        return {"valid_ids": vid, "target_2d": tgt, "num_valid": int(vid.shape[0])}
    return _p(obs_c2, "cam2"), _p(obs_c4, "cam4")


def raw_reproj_metrics(r_raw):
    r2d = r_raw.reshape(-1,2); n = torch.norm(r2d, dim=1).cpu(); sq = n**2
    return {"rms": float(torch.sqrt(sq.mean())), "median": float(n.median()),
            "p95": float(n.kthvalue(max(1,int(n.numel()*0.95))).values)}


# ============================================================
# Raw reprojection groups
# ============================================================

def make_raw_reproj_groups(x, frame_idxs, vid_c2_l, tgt_c2_l, vid_c4_l, tgt_c4_l,
                           frozen_list, model, fitter, W, H, T, stride, beta_fn, bp_anchors=None):
    device, dtype = x.device, x.dtype; beta = beta_fn(x)
    groups = []
    for t in range(T):
        fr = frozen_list[t]; s = t * stride
        o2 = x[s:s+3].view(1,3); t2 = x[s+3:s+6].view(1,3); o4 = x[s+6:s+9].view(1,3); t4 = x[s+9:s+12].view(1,3)
        bp = bp_anchors[t].to(device,dtype).view(1,-1) if bp_anchors else x[s+12:s+stride].view(1,-1)
        v2, _, _ = forward_smplx_tensor_batch(model, o2, bp, fr["lhand_c2"].to(device,dtype),
            fr["rhand_c2"].to(device,dtype), beta.unsqueeze(0).to(device,dtype), t2, fr["scale_c2"].expand(1).to(device,dtype))
        v4, _, _ = forward_smplx_tensor_batch(model, o4, bp, fr["lhand_c4"].to(device,dtype),
            fr["rhand_c4"].to(device,dtype), beta.unsqueeze(0).to(device,dtype), t4, fr["scale_c4"].expand(1).to(device,dtype))
        p2 = fitter._project_cam2(v2, W, H).squeeze(0); p4 = fitter._project_cam2(v4, W, H).squeeze(0)
        r2d_c2 = (p2.index_select(0, vid_c2_l[t].to(device)) - tgt_c2_l[t].to(device,dtype))
        r2d_c4 = (p4.index_select(0, vid_c4_l[t].to(device)) - tgt_c4_l[t].to(device,dtype))
        groups.append({"frame_idx": int(frame_idxs[t]), "camera": "cam2", "r2d": r2d_c2, "n_valid": int(r2d_c2.shape[0])})
        groups.append({"frame_idx": int(frame_idxs[t]), "camera": "cam4", "r2d": r2d_c4, "n_valid": int(r2d_c4.shape[0])})
    return groups


def flatten_groups(groups):
    return torch.cat([g["r2d"].reshape(-1) for g in groups])


# ============================================================
# Production aggregation spec — keyed by (frame_idx, camera)
# ============================================================

def build_primary_aggregation_spec(groups):
    frames_by_cam = {"cam2": set(), "cam4": set()}
    for g in groups:
        frames_by_cam[g["camera"]].add(g["frame_idx"])
    terms = {}
    for g in groups:
        c = g["camera"]
        n_frames = len(frames_by_cam[c])
        alpha = 1.0 / (n_frames * g["n_valid"]) if g["n_valid"] > 0 and n_frames > 0 else 0.0
        key = f"{g['frame_idx']}|{g['camera']}"
        terms[key] = {
            "n_valid": g["n_valid"],
            "valid_frame_count": n_frames,
            "camera_weight": 1.0,
            "alpha_per_point": alpha,
        }
    return {"terms": terms, "source": "sam3d_dense.py:272,135 + erp_dual.py:393"}


def compute_primary_charbonnier_tensor(groups, eps, spec):
    total = torch.tensor(0.0, device=groups[0]["r2d"].device, dtype=groups[0]["r2d"].dtype)
    for g in groups:
        if g["n_valid"] == 0: continue
        key = f"{g['frame_idx']}|{g['camera']}"
        term = spec["terms"][key]
        alpha = term["alpha_per_point"]
        if alpha <= 0: continue
        norms = torch.norm(g["r2d"], dim=1)
        rho = torch.sqrt(norms ** 2 + eps ** 2) - eps
        total = total + alpha * torch.sum(rho)
    return total


def compute_irls_point_weights(groups, eps, spec):
    weights = []
    for g in groups:
        if g["n_valid"] == 0: weights.append(torch.tensor([])); continue
        key = f"{g['frame_idx']}|{g['camera']}"
        term = spec["terms"][key]
        alpha = term["alpha_per_point"]
        if alpha <= 0: weights.append(torch.tensor([])); continue
        norms = torch.norm(g["r2d"], dim=1)
        weights.append((alpha / torch.sqrt(norms ** 2 + eps ** 2)).detach())
    return weights


def make_irls_linearization(raw_groups_fn, anchor_residual_fn, x_lin, eps, spec):
    groups_lin = raw_groups_fn(x_lin)
    weights = compute_irls_point_weights(groups_lin, eps, spec)
    flat_w = torch.cat([w for w in weights if w.numel() > 0])
    w_stats = {"min": float(flat_w.min()) if flat_w.numel() else 0,
               "mean": float(flat_w.mean()) if flat_w.numel() else 0,
               "median": float(flat_w.median()) if flat_w.numel() else 0,
               "max": float(flat_w.max()) if flat_w.numel() else 0, "n_points": int(flat_w.numel())}

    def surrogate_fn(x):
        groups = raw_groups_fn(x)
        parts = []
        for i, g in enumerate(groups):
            if g["n_valid"] == 0 or i >= len(weights) or weights[i].numel() == 0:
                parts.append(g["r2d"].reshape(-1))
            else:
                sw = torch.sqrt(weights[i].to(g["r2d"].device))
                parts.append((g["r2d"] * sw.unsqueeze(1)).reshape(-1))
        if anchor_residual_fn is not None:
            parts.append(anchor_residual_fn(x))
        return torch.cat(parts)

    return {"surrogate_fn": surrogate_fn, "weight_stats": w_stats, "spec": spec}


def validate_first_order(surrogate_fn, true_obj_fn, x, label, args):
    xg = x.detach().clone().requires_grad_(True)
    obj_t = true_obj_fn(xg)
    if not isinstance(obj_t, torch.Tensor): raise RuntimeError(f"Type: {type(obj_t)}")
    g_true = torch.autograd.grad(obj_t, xg, create_graph=False)[0].detach().to(torch.float64)
    J = jacfwd(surrogate_fn)(x.detach().clone().requires_grad_(True)).detach().to(torch.float64)
    g_surr = J.T @ surrogate_fn(x).detach().to(torch.float64)
    diff = g_true - g_surr
    abs_max, rel_l2 = float(diff.abs().max()), float(diff.norm() / max(g_true.norm().item(), 1.0))
    ok = rel_l2 <= args.irls_gradient_rel_tol
    print(f"[IRLS {label}] abs_max={abs_max:.2e} rel_l2={rel_l2:.2e} {'PASS' if ok else 'FAIL'}")
    return {"abs_max": abs_max, "rel_l2": rel_l2, "pass": ok}


# ============================================================
# Identity LM — explicit true_obj_fn, correct counters
# ============================================================

def identity_lm(label, make_lin, raw_metrics_fn, x0, param_dim, max_acc, max_trials, args, validate_irls=False, true_obj_fn=None, init_irls_check=None):
    if true_obj_fn is None:
        raise RuntimeError("identity_lm requires explicit true_obj_fn")
    profile = args.profile_timing; prog_n = args.progress_every_accepted
    do_sync = profile and torch.cuda.is_available()
    def sync():
        if do_sync: torch.cuda.synchronize()

    timing = {"initial_linearization_ms": 0.0, "solver_jacobian_ms": 0.0, "validation_ms": 0.0,
              "objective_eval_ms": 0.0, "linear_solve_ms": 0.0, "final_metrics_ms": 0.0}

    sync(); t_init = time.perf_counter()
    lam_min, lam_max = 1e-10, 1e12; t0 = time.perf_counter()
    x = x0.detach().clone()
    lin = make_lin(x)
    t_j = time.perf_counter(); J = jacfwd(lin["surrogate_fn"])(x.detach().clone().requires_grad_(True)).detach()
    if profile: timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
    if profile: timing["initial_linearization_ms"] = (time.perf_counter() - t_init) * 1000.0
    lam = float(torch.diag((J.to(torch.float64).T @ J.to(torch.float64))).max()) * 1e-3
    acc, rej = 0, 0
    solver_jac = 1; validation_jac = 0; lin_cnt = 1; slv_cnt = 0; chol_ok = 0; chol_fb = 0
    status = "RUNNING"
    w_history = [{"outer": -1, **lin.get("weight_stats", {})}]
    irls_ck = []
    if init_irls_check is not None:
        irls_ck.append(init_irls_check)
    spec = lin.get("spec", {})

    for outer in range(max_acc):
        r64 = lin["surrogate_fn"](x).detach().to(torch.float64)
        J64 = J.to(torch.float64); H64 = J64.T @ J64; g64 = J64.T @ r64

        sync(); t_o = time.perf_counter()
        obj_before = float(true_obj_fn(x).detach().item())
        if profile: timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

        accepted_this = False
        for trial in range(max_trials):
            sync(); t_ls = time.perf_counter()
            A = H64 + lam * torch.eye(param_dim, dtype=torch.float64); slv_cnt += 1
            try:
                L, info = torch.linalg.cholesky_ex(A)
                if info.item(): raise RuntimeError
                delta = torch.cholesky_solve(-g64.unsqueeze(1), L).squeeze(1); chol_ok += 1
            except Exception:
                try: delta = torch.linalg.solve(A, -g64.unsqueeze(1)).squeeze(1); chol_fb += 1
                except Exception: lam = min(lam*10, lam_max); rej += 1; continue
            finally:
                if profile: timing["linear_solve_ms"] += (time.perf_counter() - t_ls) * 1000.0

            xt = (x + delta.to(torch.float32)).detach()
            sync(); t_o = time.perf_counter()
            obj_trial = float(true_obj_fn(xt).detach().item())
            if profile: timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

            d64 = delta.to(torch.float64)
            pred = float(-g64.dot(d64) - 0.5*(d64 @ H64 @ d64))
            actual = obj_before - obj_trial
            rho = actual / max(pred, 1e-12) if pred > 0 else 0.0
            if pred > 0 and actual > 0 and rho > 0:
                x = xt; acc += 1; accepted_this = True
                if rho > 0.75: lam = max(lam/3, lam_min)
                elif rho < 0.25: lam = min(lam*3, lam_max)
                break
            else:
                lam = min(lam*10, lam_max); rej += 1

        if not accepted_this:
            r_now = lin["surrogate_fn"](x).detach()
            t_j = time.perf_counter(); J_now = jacfwd(lin["surrogate_fn"])(x.detach().clone().requires_grad_(True)).detach()
            if profile: timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
            validation_jac += 1
            Jn = J_now.to(torch.float64)
            sg = float((Jn.T @ r_now.to(torch.float64)).norm()/max(1,(Jn.T@Jn).norm()))
            status = "CONVERGED_SCALED_GRADIENT" if (acc > 4 and sg < 1e-4) else "LM_NO_ACCEPTED_TRIAL"
            break

        next_lin = make_lin(x)
        t_j = time.perf_counter(); next_J = jacfwd(next_lin["surrogate_fn"])(x.detach().clone().requires_grad_(True)).detach()
        if profile: timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
        solver_jac += 1; lin_cnt += 1
        w_history.append({"outer": outer, **next_lin.get("weight_stats", {})})
        if validate_irls:
            sync(); t_v = time.perf_counter()
            ck = validate_first_order(next_lin["surrogate_fn"], true_obj_fn, x, f"{label}_acc{acc}", args)
            if profile: timing["validation_ms"] += (time.perf_counter() - t_v) * 1000.0
            irls_ck.append({"label": f"acc_{acc}", **ck})
            if not ck["pass"]: raise RuntimeError(f"IRLS_FIRST_ORDER_MISMATCH at {label} outer={outer}")

        r_f = next_lin["surrogate_fn"](x).detach(); Jf = next_J.to(torch.float64)
        sc_grad_f = float((Jf.T @ r_f.to(torch.float64)).norm() / max(1, (Jf.T @ Jf).norm()))
        sync(); t_o = time.perf_counter()
        rel = (obj_before - float(true_obj_fn(x).detach().item())) / max(abs(obj_before), 1e-16)
        if profile: timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

        if prog_n > 0 and acc % prog_n == 0:
            print(f"[{label}] prog acc={acc} rej={rej} obj={float(true_obj_fn(x).detach().item()):.6e} "
                  f"lam={lam:.2e} wall={(time.perf_counter()-t0)*1000:.0f}ms")

        if sc_grad_f < args.scaled_grad_tol: status = "CONVERGED_SCALED_GRADIENT"; break
        if float(torch.norm(delta)) < args.step_tol: status = "CONVERGED_STEP"; break
        if abs(rel) < args.rel_obj_tol: status = "CONVERGED_REL_OBJECTIVE"; break
        lin, J = next_lin, next_J

    if status == "RUNNING": status = "MAX_ACCEPTED_STEPS"
    sync(); t_fm = time.perf_counter()
    xf = x.detach().clone(); rms = raw_metrics_fn(xf)
    obj_final = float(true_obj_fn(xf).detach().item())
    if profile: timing["final_metrics_ms"] = (time.perf_counter() - t_fm) * 1000.0

    ms = elapsed_ms(t0, x)

    first_order_checks = {}
    if irls_ck:
        first_order_checks = {
            "initial": irls_ck[0],
            "accepted_endpoints": irls_ck[1:] if len(irls_ck) > 1 else [],
            "worst_abs_max": max(c["abs_max"] for c in irls_ck),
            "worst_rel_l2": max(c["rel_l2"] for c in irls_ck),
            "all_pass": all(c["pass"] for c in irls_ck),
        }

    # Include init validation (done in setup_stage) in validation_ms
    if profile and init_irls_check is not None:
        # init check was timed outside; add placeholder — actual time goes via validate_first_order
        pass

    solve_count = chol_ok + chol_fb
    print(f"[{label}] {status} acc={acc} rej={rej} solver_jac={solver_jac} val_jac={validation_jac} "
          f"total_jac={solver_jac+validation_jac} lins={lin_cnt} slv={slv_cnt} "
          f"chol_ok={chol_ok} fb={chol_fb} obj={obj_final:.6e} rms={rms['rms']:.1f}")
    ret = {"label": label, "x": xf, "obj": obj_final, "status": status,
            "solver_jacobian_count": solver_jac, "validation_jacobian_count": validation_jac,
            "jacobian_count_total": solver_jac + validation_jac,
            "linearization_count": lin_cnt, "solve_count": solve_count,
            "cholesky_success_count": chol_ok, "solve_fallback_count": chol_fb,
            "accepted_steps": acc, "rejected_trials": rej,
            "final_lambda": lam, "wall_time_ms": ms, "rms": rms,
            "w_history": w_history, "first_order_checks": first_order_checks, "spec": spec}
    if profile: ret["profile_timing"] = timing
    return ret


def run_lbfgs(label, true_obj_fn, raw_metrics_fn, x0, outer_steps, args):
    x = x0.detach().clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], line_search_fn="strong_wolfe", lr=args.lbfgs_lr, max_iter=args.lbfgs_max_iter)
    cc = [0]; t0 = time.perf_counter()
    def c():
        cc[0] += 1; opt.zero_grad(); obj = true_obj_fn(x)
        if not isinstance(obj, torch.Tensor): raise TypeError(f"Type: {type(obj)}")
        if obj.ndim != 0: raise RuntimeError(f"Ndims: {obj.shape}")
        obj.backward(); return obj
    for _ in range(outer_steps): opt.step(c)
    xf = x.detach(); ms = elapsed_ms(t0, xf)
    print(f"[{label}] cls={cc[0]} obj={float(true_obj_fn(xf).detach().item()):.6e} rms={raw_metrics_fn(xf)['rms']:.1f}")
    return {"label": label, "x": xf, "obj": float(true_obj_fn(xf).detach().item()), "clos": cc[0], "wall_time_ms": ms}


# ============================================================
# Handoff helpers
# ============================================================

def stage_verdict(lm, bf, raw_metrics_fn):
    if not bf: return None
    ok = (lm["obj"] <= bf["obj"]*1.02 and lm["rms"]["rms"] <= raw_metrics_fn(bf["x"])["rms"]*1.05
          and lm["rms"]["median"] <= raw_metrics_fn(bf["x"])["median"]*1.05
          and lm["status"] not in ("LM_NO_ACCEPTED_TRIAL","NUMERICAL_FAILURE"))
    return "LM_MATCHES_LBFGS" if ok else "LM_LBFGS_GAP"


def check_s1_to_s2a(x_s1, x_s2a, bp_anchors, T):
    pf = S1_PF + POSE_PF; ot_diff, bp_diff = 0.0, 0.0
    for t in range(T):
        ot = float((x_s2a[t*pf:t*pf+12] - x_s1[t*12:(t+1)*12]).abs().max()); ot_diff = max(ot_diff, ot)
        bp = float((x_s2a[t*pf+12:(t+1)*pf] - bp_anchors[t]).abs().max()); bp_diff = max(bp_diff, bp)
    return {"ot_max": ot_diff, "bp_max": bp_diff, "ok": ot_diff < 1e-6 and bp_diff < 1e-6}


def handoff_s1_to_s2a(x_s1, bp_anchors, T):
    parts = []; [parts.extend([x_s1[t*12:(t+1)*12], bp_anchors[t].clone()]) for t in range(T)]
    return torch.cat(parts).clone()


def check_s2a_to_s2b(x_s2a, x_s2b, beta_anchor, beta_abs_limit, sa_scale,
                      raw_groups_fn_s2a, raw_groups_fn_s2b, true_obj_s2a, true_obj_s2b):
    s2a_dim = x_s2a.shape[0]; beta_dim = x_s2b.shape[0] - s2a_dim
    state_max_diff = float((x_s2b[:s2a_dim] - x_s2a).abs().max())
    beta_init = beta_abs_limit * torch.tanh(x_s2b[-beta_dim:])
    bp_max_diff = float((beta_init - beta_anchor).abs().max())
    raw_diff = float((flatten_groups(raw_groups_fn_s2b(x_s2b)).detach()
                      - flatten_groups(raw_groups_fn_s2a(x_s2a)).detach()).abs().max())
    sa_obj_init = float(0.5 * torch.sum((beta_init - beta_anchor) ** 2 * sa_scale ** 2))
    obj_s2a = float(true_obj_s2a(x_s2a).detach().item())
    obj_s2b_init = float(true_obj_s2b(x_s2b).detach().item())
    rel_obj = abs(obj_s2b_init - obj_s2a) / max(abs(obj_s2a), 1e-16)
    ok = (state_max_diff <= 1e-7 and bp_max_diff <= 1e-6 and raw_diff <= 1e-3
          and sa_obj_init <= 1e-10 and rel_obj <= 1e-6)
    return {"state_max_diff": state_max_diff,
            "physical_beta_init_max_diff_to_anchor": bp_max_diff,
            "raw_reprojection_max_diff": raw_diff,
            "shape_anchor_objective_at_init": sa_obj_init,
            "relative_difference_s2b_init_to_s2a": rel_obj,
            "ok": ok}


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    frame_list = [int(s.strip()) for s in args.frames.split(",")]
    T = len(frame_list); W, H = args.image_width, args.image_height
    out_d = Path(args.output_dir); is_irls = args.reproj_objective == "charbonnier_irls"; eps = args.charbonnier_eps
    mode = "charbonnier_irls" if is_irls else "raw_l2"
    print(f"[MODE] {mode}")

    rrc = Path(config_script.PROJECT_ROOT) / "configs" / "rotate_upright_rules.json"
    rotate_rules = resolve_dataset_rotate_rules(rrc, Path(args.bbox_dir))
    hi = set(); obs_list = []; dh2, dh4 = {}, {}
    for fidx in frame_list:
        obs_list.append(load_observations(fidx, Path(args.sam3d_dense_dir), Path(args.bbox_dir), W, H, rotate_rules))
        dh2[fidx] = try_load_smplx_init(fidx, Path(args.hmr_dir), hi, "cam2")
        dh4[fidx] = try_load_smplx_init(fidx, Path(args.hmr_dir), hi, "cam4")
    sp2, sp4 = [], []
    for fidx in frame_list:
        sp2.append(build_hmr_init_for_frame(fidx, frame_list, dh2)[0]); sp4.append(build_hmr_init_for_frame(fidx, frame_list, dh4)[0])
    si2, si4 = build_segment_smplx_init(sp2, sp4)
    fitter = DualCamERPFitter(primary_observation_source="sam3d_dense", gender="neutral", R_cam4=R_CAM4)
    sc2, sc4, sbeta, sbpose = fitter._init_params(si2, si4, [o[0] for o in obs_list], [o[1] for o in obs_list], (W, H))
    body_pose_dim = int(sbpose.shape[-1]); beta_dim = int(sbeta.numel())
    bp_anchors = [sbpose.detach().clone()[t].reshape(-1) for t in range(T)]
    beta_anchor = sbeta.detach().clone().reshape(-1)
    model = fitter._get_model(); frozen_list = [build_frozen_params(sc2, sc4, t) for t in range(T)]
    sample_ids = fitter._sam3d_dense_sample_ids; info_list = []
    for t in range(T): info_list.append(precompute_valid_points(obs_list[t][0], obs_list[t][1], sample_ids))
    vid_c2_l, tgt_c2_l, vid_c4_l, tgt_c4_l = [], [], [], []
    for t in range(T):
        i2, i4 = info_list[t]; vid_c2_l.append(torch.as_tensor(i2["valid_ids"].astype(np.int64)))
        tgt_c2_l.append(torch.as_tensor(i2["target_2d"].astype(np.float32)))
        vid_c4_l.append(torch.as_tensor(i4["valid_ids"].astype(np.int64)))
        tgt_c4_l.append(torch.as_tensor(i4["target_2d"].astype(np.float32)))

    safe_bound = args.beta_abs_limit * (1.0 - BETA_ATANH_EPS)
    if float(beta_anchor.abs().max()) >= safe_bound: raise RuntimeError("INFEASIBLE_BETA_ANCHOR")
    beta_latent0 = torch.atanh(beta_anchor / args.beta_abs_limit)

    pf_dim = S1_PF + body_pose_dim
    rep_scale = 1.0 / (args.reproj_sigma_px * math.sqrt(max(sum(i2["num_valid"] + i4["num_valid"] for i2, i4 in info_list), 1)))
    pa_scale = math.sqrt(2.0 * args.pose_anchor_weight / (T * body_pose_dim))
    sa_scale = math.sqrt(2.0 * args.shape_anchor_weight / beta_dim)
    do_v = args.validate_irls_each_accepted and is_irls
    do_iv = args.validate_irls_first_order and is_irls

    def beta_frozen(x): return beta_anchor
    def beta_active(x): return args.beta_abs_limit * torch.tanh(x[-beta_dim:])

    x0_s1 = torch.cat([torch.cat([sc2.orient.detach()[t].reshape(-1), sc2.transl.detach()[t].reshape(-1),
                                   sc4.orient.detach()[t].reshape(-1), sc4.transl.detach()[t].reshape(-1)])
                        for t in range(T)]).clone(); s1_dim = T * S1_PF
    results = {"config": vars(args), "handoffs": {}}

    # ===== Helper to create lin + true_obj per stage =====
    def setup_stage(name, raw_groups_fn, anchor_fn, x0_lm, x0_lbfgs, dim, max_acc, raw_met_fn, bf_outer):
        if is_irls:
            static_spec = build_primary_aggregation_spec(raw_groups_fn(x0_lm))
            def make_lin(x): return make_irls_linearization(raw_groups_fn, anchor_fn, x, eps, static_spec)
            def true_obj(x):
                g = raw_groups_fn(x)
                obj = compute_primary_charbonnier_tensor(g, eps, static_spec)
                if anchor_fn is not None: obj = obj + 0.5 * torch.sum(anchor_fn(x) ** 2)
                return obj
            init_lin = make_lin(x0_lm); init_ck = None
            if do_iv:
                if args.profile_timing and torch.cuda.is_available(): torch.cuda.synchronize()
                t_iv = time.perf_counter() if args.profile_timing else 0
                init_ck = validate_first_order(init_lin["surrogate_fn"], true_obj, x0_lm, f"{name}_init", args)
                if args.profile_timing:
                    iv_ms = (time.perf_counter() - t_iv) * 1000.0
                if not init_ck["pass"]: raise RuntimeError(f"IRLS_FIRST_ORDER_MISMATCH at {name}")
        else:
            def tot_fn(x):
                r = flatten_groups(raw_groups_fn(x)) * rep_scale
                return r if anchor_fn is None else torch.cat([r, anchor_fn(x)])
            def make_lin(x): return {"surrogate_fn": tot_fn, "weight_stats": {}}
            def true_obj(x): return 0.5 * torch.sum(tot_fn(x) ** 2)
            init_ck = None

        sample_groups = raw_groups_fn(x0_lm); n_groups = len(sample_groups)
        total_valid = sum(g["n_valid"] for g in sample_groups)
        flat_sample = flatten_groups(sample_groups); surr_dim = int(flat_sample.numel())
        if anchor_fn is not None: surr_dim += int(anchor_fn(x0_lm).numel())
        print(f"\n{'='*60}\n[{name}] T={T} param_dim={dim} groups={n_groups} "
              f"valid_pts={total_valid} surr_dim={surr_dim}\n{'='*60}")
        lm = identity_lm(f"{name}_LM", make_lin, raw_met_fn, x0_lm, dim, max_acc, args.lm_max_trials,
                         args, true_obj_fn=true_obj, validate_irls=do_v, init_irls_check=init_ck)
        if args.profile_timing and do_iv and "profile_timing" in lm:
            lm["profile_timing"]["validation_ms"] += iv_ms
        bf = run_lbfgs(f"{name}_LBFGS", true_obj, raw_met_fn, x0_lbfgs, bf_outer, args) if (args.run_lbfgs_oracle and x0_lbfgs is not None) else None
        v = stage_verdict(lm, bf, raw_met_fn) if bf else None
        if v: print(f"[V_{name}] {v}")
        return lm, bf, v, true_obj

    # ===== STAGE 1 =====
    def rg_s1(x): return make_raw_reproj_groups(x, frame_list, vid_c2_l, tgt_c2_l, vid_c4_l, tgt_c4_l,
                                                 frozen_list, model, fitter, W, H, T, S1_PF, beta_frozen, bp_anchors)
    def rm_s1(x): return raw_reproj_metrics(flatten_groups(rg_s1(x)).detach())
    s1_lm, s1_lbfgs, s1_v, _ = setup_stage("S1", rg_s1, None,
        x0_lm=x0_s1.clone(), x0_lbfgs=x0_s1.clone(),
        dim=s1_dim, max_acc=args.stage1_lm_max_acc, raw_met_fn=rm_s1, bf_outer=args.stage1_bf_outer)
    results["stage1"] = {"lm": {k: v for k, v in s1_lm.items() if k != "x"},
                          "lbfgs": {k: v for k, v in s1_lbfgs.items() if k != "x"} if s1_lbfgs else None,
                          "input_summary": {"T": T, "param_dim": s1_dim, "n_groups": 2*T,
                              "total_valid_points": sum(g["n_valid"] for g in rg_s1(x0_s1))},
                          "verdict": s1_v}
    if args.stop_after_stage == "stage1":
        del results["handoffs"]
        npz_s1 = {"x0_s1": x0_s1.cpu().numpy(), "x_s1_lm": s1_lm["x"].cpu().numpy(),
                  "beta_anchor": beta_anchor.cpu().numpy(), "beta_latent0": beta_latent0.cpu().numpy(),
                  "raw_reproj_s1_lm": flatten_groups(rg_s1(s1_lm["x"])).detach().cpu().numpy()}
        if s1_lbfgs: npz_s1["x_s1_lbfgs"] = s1_lbfgs["x"].cpu().numpy()
        seg = f"{frame_list[0]}_{frame_list[-1]}"; out_frame = out_d / seg; out_frame.mkdir(parents=True, exist_ok=True)
        json.dump(results, (out_frame / "staged_stage2b_beta.json").open("w"), ensure_ascii=False, indent=2, default=str)
        np.savez_compressed(str(out_frame / "staged_stage2b_beta.npz"), **npz_s1); print(f"[SAVE] {out_frame}")
        raise SystemExit(0)

    # ===== STAGE 2A =====
    s2a_dim = T * pf_dim
    x0_s2a_lm = handoff_s1_to_s2a(s1_lm["x"], bp_anchors, T)
    x0_s2a_bf = handoff_s1_to_s2a(s1_lbfgs["x"], bp_anchors, T) if s1_lbfgs else None
    hoff_s1_s2a_lm = check_s1_to_s2a(s1_lm["x"], x0_s2a_lm, bp_anchors, T)
    print(f"[HANDOFF_LM_S1_S2A] ot={hoff_s1_s2a_lm['ot_max']:.2e} bp={hoff_s1_s2a_lm['bp_max']:.2e} ok={hoff_s1_s2a_lm['ok']}")
    if not hoff_s1_s2a_lm["ok"]: raise RuntimeError("LM handoff S1->S2A failed")
    results["handoffs"]["lm_s1_to_s2a"] = hoff_s1_s2a_lm
    if s1_lbfgs:
        hoff_s1_s2a_bf = check_s1_to_s2a(s1_lbfgs["x"], x0_s2a_bf, bp_anchors, T)
        print(f"[HANDOFF_BF_S1_S2A] ot={hoff_s1_s2a_bf['ot_max']:.2e} bp={hoff_s1_s2a_bf['bp_max']:.2e} ok={hoff_s1_s2a_bf['ok']}")
        if not hoff_s1_s2a_bf["ok"]: raise RuntimeError("BF handoff S1->S2A failed")
        results["handoffs"]["bf_s1_to_s2a"] = hoff_s1_s2a_bf

    def rg_s2a(x): return make_raw_reproj_groups(x, frame_list, vid_c2_l, tgt_c2_l, vid_c4_l, tgt_c4_l,
                                                  frozen_list, model, fitter, W, H, T, pf_dim, beta_frozen)
    def a_s2a(x):
        parts = []; [parts.append((x[t*pf_dim+12:(t+1)*pf_dim] - bp_anchors[t]) * pa_scale) for t in range(T)]
        return torch.cat(parts)
    def rm_s2a(x): return raw_reproj_metrics(flatten_groups(rg_s2a(x)).detach())
    s2a_lm, s2a_lbfgs, s2a_v, s2a_true_obj = setup_stage("S2A", rg_s2a, a_s2a,
        x0_lm=x0_s2a_lm, x0_lbfgs=x0_s2a_bf,
        dim=s2a_dim, max_acc=args.stage2a_lm_max_acc, raw_met_fn=rm_s2a, bf_outer=args.stage2a_bf_outer)
    results["stage2a"] = {"lm": {k: v for k, v in s2a_lm.items() if k != "x"},
                           "lbfgs": {k: v for k, v in s2a_lbfgs.items() if k != "x"} if s2a_lbfgs else None,
                           "input_summary": {"T": T, "param_dim": s2a_dim, "n_groups": 2*T,
                               "total_valid_points": sum(g["n_valid"] for g in rg_s2a(x0_s2a_lm))},
                           "verdict": s2a_v}
    if args.stop_after_stage == "stage2a":
        raw_s1 = flatten_groups(rg_s1(s1_lm["x"])).detach().cpu().numpy()
        raw_s2a = flatten_groups(rg_s2a(s2a_lm["x"])).detach().cpu().numpy()
        npz_s2a = {"x0_s1": x0_s1.cpu().numpy(), "x0_s2a_lm": x0_s2a_lm.cpu().numpy(),
                   "x_s1_lm": s1_lm["x"].cpu().numpy(), "x_s2a_lm": s2a_lm["x"].cpu().numpy(),
                   "beta_anchor": beta_anchor.cpu().numpy(), "beta_latent0": beta_latent0.cpu().numpy(),
                   "raw_reproj_s1_lm": raw_s1, "raw_reproj_s2a_lm": raw_s2a}
        if s1_lbfgs: npz_s2a["x_s1_lbfgs"] = s1_lbfgs["x"].cpu().numpy()
        if s2a_lbfgs: npz_s2a["x_s2a_lbfgs"] = s2a_lbfgs["x"].cpu().numpy()
        if s2a_lbfgs: npz_s2a["x0_s2a_lbfgs"] = x0_s2a_bf.cpu().numpy()
        seg = f"{frame_list[0]}_{frame_list[-1]}"; out_frame = out_d / seg; out_frame.mkdir(parents=True, exist_ok=True)
        json.dump(results, (out_frame / "staged_stage2b_beta.json").open("w"), ensure_ascii=False, indent=2, default=str)
        np.savez_compressed(str(out_frame / "staged_stage2b_beta.npz"), **npz_s2a); print(f"[SAVE] {out_frame}")
        raise SystemExit(0)

    # ===== STAGE 2B =====
    s2b_dim = s2a_dim + beta_dim
    beta_latent = torch.atanh(beta_anchor / args.beta_abs_limit)
    x0_s2b_lm = torch.cat([s2a_lm["x"].clone(), beta_latent.clone()]).clone()
    x0_s2b_bf = torch.cat([s2a_lbfgs["x"].clone(), beta_latent.clone()]).clone() if s2a_lbfgs else None

    def rg_s2b(x): return make_raw_reproj_groups(x, frame_list, vid_c2_l, tgt_c2_l, vid_c4_l, tgt_c4_l,
                                                  frozen_list, model, fitter, W, H, T, pf_dim, beta_active)
    def a_pa_s2b(x): return a_s2a(x)
    def a_sa_s2b(x): return (args.beta_abs_limit * torch.tanh(x[-beta_dim:]) - beta_anchor) * sa_scale
    def a_all_s2b(x): return torch.cat([a_pa_s2b(x), a_sa_s2b(x)])
    def rm_s2b(x): return raw_reproj_metrics(flatten_groups(rg_s2b(x)).detach())

    s2b_lm, s2b_lbfgs, s2b_v, s2b_true_obj = setup_stage("S2B", rg_s2b, a_all_s2b,
        x0_lm=x0_s2b_lm, x0_lbfgs=x0_s2b_bf,
        dim=s2b_dim, max_acc=args.stage2b_lm_max_acc, raw_met_fn=rm_s2b, bf_outer=args.stage2b_bf_outer)

    # S2A->S2B handoff checks
    hoff_s2a_s2b_lm = check_s2a_to_s2b(s2a_lm["x"], x0_s2b_lm, beta_anchor, args.beta_abs_limit, sa_scale,
                                         rg_s2a, rg_s2b, s2a_true_obj, s2b_true_obj)
    print(f"[HANDOFF_LM_S2A_S2B] state={hoff_s2a_s2b_lm['state_max_diff']:.2e} "
          f"bp={hoff_s2a_s2b_lm['physical_beta_init_max_diff_to_anchor']:.2e} "
          f"raw={hoff_s2a_s2b_lm['raw_reprojection_max_diff']:.2e} "
          f"sa={hoff_s2a_s2b_lm['shape_anchor_objective_at_init']:.2e} "
          f"rel={hoff_s2a_s2b_lm['relative_difference_s2b_init_to_s2a']:.2e} "
          f"ok={hoff_s2a_s2b_lm['ok']}")
    if not hoff_s2a_s2b_lm["ok"]: raise RuntimeError("LM handoff S2A->S2B failed")
    results["handoffs"]["lm_s2a_to_s2b"] = hoff_s2a_s2b_lm
    if s2b_lbfgs:
        hoff_s2a_s2b_bf = check_s2a_to_s2b(s2a_lbfgs["x"], x0_s2b_bf, beta_anchor, args.beta_abs_limit, sa_scale,
                                            rg_s2a, rg_s2b, s2a_true_obj, s2b_true_obj)
        print(f"[HANDOFF_BF_S2A_S2B] state={hoff_s2a_s2b_bf['state_max_diff']:.2e} "
              f"bp={hoff_s2a_s2b_bf['physical_beta_init_max_diff_to_anchor']:.2e} "
              f"raw={hoff_s2a_s2b_bf['raw_reprojection_max_diff']:.2e} "
              f"sa={hoff_s2a_s2b_bf['shape_anchor_objective_at_init']:.2e} "
              f"rel={hoff_s2a_s2b_bf['relative_difference_s2b_init_to_s2a']:.2e} "
              f"ok={hoff_s2a_s2b_bf['ok']}")
        if not hoff_s2a_s2b_bf["ok"]: raise RuntimeError("BF handoff S2A->S2B failed")
        results["handoffs"]["bf_s2a_to_s2b"] = hoff_s2a_s2b_bf

    # Beta metrics
    beta_lm = args.beta_abs_limit * torch.tanh(s2b_lm["x"][-beta_dim:])
    b_s2b = {"lm": {"max": float(beta_lm.abs().max()),
                     "sat": float(beta_lm.abs().max()) / args.beta_abs_limit,
                     "delta": float(torch.norm(beta_lm - beta_anchor))}}
    if s2b_lbfgs:
        beta_bf = args.beta_abs_limit * torch.tanh(s2b_lbfgs["x"][-beta_dim:])
        b_s2b["lbfgs"] = {"max": float(beta_bf.abs().max()),
                           "sat": float(beta_bf.abs().max()) / args.beta_abs_limit,
                           "delta": float(torch.norm(beta_bf - beta_anchor))}

    # S2B objective breakdown
    def obj_breakdown(raw_groups_fn, a_pa_fn, a_sa_fn, true_obj, x):
        g = raw_groups_fn(x)
        primary = float(compute_primary_charbonnier_tensor(g, eps, build_primary_aggregation_spec(g)).detach().item())
        pose_anchor = float(0.5 * torch.sum(a_pa_fn(x) ** 2).detach().item()) if a_pa_fn is not None else 0.0
        shape_anchor = float(0.5 * torch.sum(a_sa_fn(x) ** 2).detach().item()) if a_sa_fn is not None else 0.0
        return {"primary": primary, "pose_anchor": pose_anchor, "shape_anchor": shape_anchor}
    break_lm = obj_breakdown(rg_s2b, a_pa_s2b, a_sa_s2b, s2b_true_obj, s2b_lm["x"])
    b_s2b["lm"]["breakdown"] = break_lm
    if s2b_lbfgs:
        break_bf = obj_breakdown(rg_s2b, a_pa_s2b, a_sa_s2b, s2b_true_obj, s2b_lbfgs["x"])
        b_s2b["lbfgs"]["breakdown"] = break_bf

    results["stage2b"] = {"lm": {k: v for k, v in s2b_lm.items() if k != "x"},
                           "lbfgs": {k: v for k, v in s2b_lbfgs.items() if k != "x"} if s2b_lbfgs else None,
                           "input_summary": {"T": T, "param_dim": s2b_dim, "n_groups": 2*T,
                               "total_valid_points": sum(g["n_valid"] for g in rg_s2b(x0_s2b_lm))},
                           "beta": b_s2b, "verdict": s2b_v}

    # Gate C verdict
    s1_ok = s1_v == "LM_MATCHES_LBFGS"
    s2a_ok = s2a_v == "LM_MATCHES_LBFGS"
    s2b_ok = s2b_v == "LM_MATCHES_LBFGS"
    h1_lm = results["handoffs"].get("lm_s1_to_s2a", {}).get("ok", True)
    h1_bf = results["handoffs"].get("bf_s1_to_s2a", {}).get("ok", True)
    h2_lm = results["handoffs"].get("lm_s2a_to_s2b", {}).get("ok", True)
    h2_bf = results["handoffs"].get("bf_s2a_to_s2b", {}).get("ok", True)
    beta_sat_ok = b_s2b["lm"]["sat"] < 0.90
    all_ok = s1_ok and s2a_ok and s2b_ok and h1_lm and h1_bf and h2_lm and h2_bf and beta_sat_ok
    gc = "GATE_C_PASS_READY_FOR_MULTIFRAME_GATE_D" if all_ok else "GATE_C_BLOCKED"
    results["gate_c_verdict"] = {"verdict": gc, "checks": {
        "s1_match": s1_ok, "s2a_match": s2a_ok, "s2b_match": s2b_ok,
        "handoff_s1_s2a_lm": h1_lm, "handoff_s1_s2a_bf": h1_bf,
        "handoff_s2a_s2b_lm": h2_lm, "handoff_s2a_s2b_bf": h2_bf,
        "beta_sat_lt_0.9": beta_sat_ok}}
    print(f"[GATE_C] {gc}")

    # Collect raw reprojection arrays for NPZ
    raw_s1_lm = flatten_groups(rg_s1(s1_lm["x"])).detach().cpu().numpy()
    raw_s1_bf = flatten_groups(rg_s1(s1_lbfgs["x"])).detach().cpu().numpy() if s1_lbfgs else None
    raw_s2a_lm = flatten_groups(rg_s2a(s2a_lm["x"])).detach().cpu().numpy()
    raw_s2a_bf = flatten_groups(rg_s2a(s2a_lbfgs["x"])).detach().cpu().numpy() if s2a_lbfgs else None
    raw_s2b_lm = flatten_groups(rg_s2b(s2b_lm["x"])).detach().cpu().numpy()
    raw_s2b_bf = flatten_groups(rg_s2b(s2b_lbfgs["x"])).detach().cpu().numpy() if s2b_lbfgs else None

    npz_dict = {
        "x0_s1": x0_s1.cpu().numpy(),
        "x0_s2a_lm": x0_s2a_lm.cpu().numpy(),
        "x0_s2b_lm": x0_s2b_lm.cpu().numpy(),
        "x_s1_lm": s1_lm["x"].cpu().numpy(),
        "x_s2a_lm": s2a_lm["x"].cpu().numpy(),
        "x_s2b_lm": s2b_lm["x"].cpu().numpy(),
        "beta_anchor": beta_anchor.cpu().numpy(),
        "beta_latent0": beta_latent0.cpu().numpy(),
        "beta_lm": beta_lm.cpu().numpy(),
        "raw_reproj_s1_lm": raw_s1_lm,
        "raw_reproj_s2a_lm": raw_s2a_lm,
        "raw_reproj_s2b_lm": raw_s2b_lm,
    }
    if s1_lbfgs:
        npz_dict["x_s1_lbfgs"] = s1_lbfgs["x"].cpu().numpy()
        npz_dict["raw_reproj_s1_lbfgs"] = raw_s1_bf
    if s2a_lbfgs:
        npz_dict["x0_s2a_lbfgs"] = x0_s2a_bf.cpu().numpy()
        npz_dict["x_s2a_lbfgs"] = s2a_lbfgs["x"].cpu().numpy()
        npz_dict["raw_reproj_s2a_lbfgs"] = raw_s2a_bf
    if s2b_lbfgs:
        npz_dict["x0_s2b_lbfgs"] = x0_s2b_bf.cpu().numpy()
        npz_dict["x_s2b_lbfgs"] = s2b_lbfgs["x"].cpu().numpy()
        npz_dict["beta_lbfgs"] = beta_bf.cpu().numpy()
        npz_dict["raw_reproj_s2b_lbfgs"] = raw_s2b_bf

    seg = f"{frame_list[0]}_{frame_list[-1]}"; out_frame = out_d / seg
    out_frame.mkdir(parents=True, exist_ok=True)
    with (out_frame / "staged_stage2b_beta.json").open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    np.savez_compressed(str(out_frame / "staged_stage2b_beta.npz"), **npz_dict)
    print(f"[SAVE] {out_frame}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
