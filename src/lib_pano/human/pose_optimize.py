"""VPoser 姿态优化（Step 3）+ 可微 Silhouette SDF Loss"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R_scipy
from scipy.ndimage import distance_transform_edt

from lib_pano.camera.omni_camera  import omni_project_torch
from lib_pano.human.detect        import SMPL_ALIGN
from lib_pano.human.vposer        import smpl_from_vposer, fk_init_z
from lib_pano.vis.visual_human          import save_joint_overlay

logger = logging.getLogger(__name__)


# ── 可微 Silhouette SDF Loss ───────────────────────────────────

def build_seg_sdf(seg_mask: np.ndarray) -> np.ndarray:
    """bool mask → SDF (正=内, 负=外, 单位像素)"""
    return (distance_transform_edt(seg_mask) -
            distance_transform_edt(~seg_mask)).astype(np.float32)


def sil_sdf_loss(verts_cam, sdf_tensor, cam, H, W):
    """单向 silhouette SDF loss：顶点投影超出 seg_mask 时惩罚"""
    uv = omni_project_torch(verts_cam.unsqueeze(0), **cam).squeeze(0)
    uv_n = torch.stack([(uv[:,0]/(W-1.))*2.-1.,
                        (uv[:,1]/(H-1.))*2.-1.], -1).view(1,1,-1,2)
    sampled = F.grid_sample(sdf_tensor, uv_n, mode='bilinear',
                            align_corners=True, padding_mode='border').view(-1)
    z_w = torch.sigmoid((verts_cam[:,2]-0.01)*200.)
    return (F.relu(-sampled)*z_w).mean()


# ── Step 3: VPoser 姿态优化 ─────────────────────────────────────

def step3_vposer_fit(s1, s2, smpl, enc, dec, cam,
                     device, obj_path, output_dir, seg_mask=None):
    """Exp5 VPoser 姿态优化：Exp2 热启动 → 三阶段优化 → 返回最终结果"""
    logger.info("Step 3: Exp5 VPoser 姿态优化 (Exp2 热启动)")
    H_img, W_img = s1['H'], s1['W']

    # 构建分割 SDF（预计算，不参与反传自身，但采样结果可微）
    sdf_tensor = None
    if seg_mask is not None:
        sdf_np     = build_seg_sdf(seg_mask)                           # (H,W) float32
        sdf_tensor = torch.tensor(sdf_np, device=device,
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        logger.info(f"  分割 SDF 构建完成，形状: {tuple(sdf_tensor.shape)}，"
                    f"  范围=[{sdf_np.min():.0f}, {sdf_np.max():.0f}]px")
    else:
        logger.warning("  ⚠ 无分割 mask，silhouette loss 将全程为 0")

    target_t = torch.tensor(s1['target_uv'], dtype=torch.float32, device=device)
    vmask_t  = torch.tensor(s1['valid_mask'], dtype=torch.float32, device=device)
    smpl.to(device); enc.to(device); dec.to(device)

    # ─ 初始化参数 ─
    # 1. global_orient ← Exp2 R_combo 轴角
    R_combo  = s2['R_combo']
    go_aa_np = R_scipy.from_matrix(R_combo).as_rotvec().astype(np.float32)
    global_orient = nn.Parameter(
        torch.tensor(go_aa_np, dtype=torch.float32, device=device).unsqueeze(0))

    # 2. z ← FK 逆解
    enc.eval()
    z_init = fk_init_z(smpl, enc, device, obj_path)
    enc.train()
    z = nn.Parameter(z_init.to(device).clone())

    # 3. transl, log_scale ← Exp2
    transl    = nn.Parameter(torch.tensor(s2['trans'], dtype=torch.float32,
                                          device=device).unsqueeze(0))
    log_scale = nn.Parameter(torch.tensor([s2['log_scale']], dtype=torch.float32,
                                          device=device))
    betas     = nn.Parameter(torch.zeros(1, 10, device=device))
    init_depth_t = transl[0, 2].detach()

    logger.info(f"  初始化: global_orient ||aa||={np.linalg.norm(go_aa_np):.3f}rad"
               f"  depth={transl[0,2].item():.3f}m  scale={log_scale.exp().item():.3f}")

    best_loss, best_state = float('inf'), {}
    SMPL_HEAD_J, SMPL_FEET_J = 15, [7, 8, 10, 11]  # head-up prior: head_y<0, feet_y>0 in OpenCV

    def run_stage(params, n_iter, lr, dr_w, vp_w, upright_w, name,
                  log_every=100, sdf_tensor=None, sil_w=0.0):
        nonlocal best_loss, best_state
        opt = torch.optim.Adam(params, lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter, lr*0.05)
        sil_tag = f", sil={sil_w:.0e}" if (sdf_tensor is not None and sil_w > 0) else ""
        logger.info(f"  ── {name} ({n_iter}it, lr={lr}, vp={vp_w}, up={upright_w:.0e}{sil_tag}) ──")
        for it in range(n_iter):
            opt.zero_grad()
            j3d, v_all = smpl_from_vposer(dec, z, smpl, betas, global_orient)
            sc      = torch.exp(log_scale)
            pts     = sc * j3d + transl.unsqueeze(1)          # (1,24,3) camera space
            pts_aln = pts[:, SMPL_ALIGN, :]                   # (1,7,3)
            uv_pred = omni_project_torch(pts_aln, **cam)      # (1,7,2)
            dist_sq = ((uv_pred.squeeze(0) - target_t)**2).sum(-1)  # (7,) 
            loss_j  = (dist_sq * vmask_t).sum() / vmask_t.sum().clamp(1) # joint reprojection loss
            loss_vp = z.pow(2).mean()   # VPoser prior loss (z 越大越偏离 VPoser 潜在空间)
            loss_zp = F.relu(0.05 - pts_aln[...,2]).pow(2).mean() * 1e4  # depth prior: joints should be in front of camera (z>0.05m) to avoid bad local minima; 
            loss_dp = F.relu(0.1  - transl[0,2]).pow(2) * 1e6  # depth prior: translation along z-axis should be > 10cm
            loss_dr = (transl[0,2] - init_depth_t).pow(2) * dr_w # depth regularization: don't deviate too much from initial depth (Exp2 result)
            loss_b  = betas.pow(2).mean() * 0.1  # shape prior: keep shape close to mean
            head_y  = j3d[:, SMPL_HEAD_J, 1]
            feet_y  = j3d[:, SMPL_FEET_J, 1].mean(dim=-1)
            # loss_up = (F.relu(head_y).pow(2) + F.relu(-feet_y).pow(2)).mean() * upright_w
            loss_up = 0
            if sdf_tensor is not None and sil_w > 0:
                loss_sil = sil_sdf_loss((sc*v_all + transl.unsqueeze(1)).squeeze(0),
                                        sdf_tensor, cam, H_img, W_img) * sil_w
            else:
                loss_sil = torch.zeros(1, device=device)
            loss = (loss_j + vp_w*loss_vp + loss_zp + loss_dp
                    + loss_dr + loss_b + loss_up + loss_sil)
            loss.backward(); opt.step(); sch.step()
            if loss_j.item() < best_loss:   # 用关节误差判断最优（不含 upright）
                best_loss  = loss_j.item()
                best_state = {k: v.detach().clone() for k, v in
                              [('z',z),('go',global_orient),('transl',transl),
                               ('betas',betas),('ls',log_scale)]}
            if it % log_every == 0 or it == n_iter-1:
                rmse    = dist_sq.mul(vmask_t).sum().div(vmask_t.sum().clamp(1)).sqrt().item()
                head_ok = 'HD↑' if head_y.item() < 0 else 'HD↓!'
                sil_str = (f"  sil_raw={loss_sil.item()/sil_w:.1f}px→weighted={loss_sil.item():.0f}"
                           if (sdf_tensor is not None and sil_w > 0) else "")
                logger.info(f"  [{it:4d}] jt={loss_j.item():.0f}  RMSE={rmse:.1f}px"
                             f"  depth={transl[0,2].item():.3f}m  {head_ok}{sil_str}")

    # 三阶段优化
    # Stage0: z 固定，先让刚体参数（global_orient/transl/scale）对准；
    #         z 释放太早 + 高lr 会导致 z 跑偏（32维），破坏 FK_init
    run_stage([global_orient, transl, log_scale, betas],
              600, 1e-2, 5e2, 0.0, 1e3, "Stage0: 刚体对齐 z固定 (先对朝向)", 150,
              sdf_tensor=None, sil_w=0.0)
    # Stage1: 刚体已对准，释放 z 做姿态拟合；关节优先，不加轮廓
    run_stage([z, global_orient, transl, log_scale, betas],
              600, 5e-3, 2e3, 0.5, 5e2, "Stage1: 全参数粗对齐 (关节优先)", 150,
              sdf_tensor=None, sil_w=0.0)
    # Stage2: 关节基本收敛后，小权重轮廓辅助
    run_stage([z, global_orient, transl, log_scale, betas],
              1000, 5e-4, 1e3, 1.5, 2e2, "Stage2: 精细对齐+轮廓辅助 (VPoser增强)", 250,
              sdf_tensor=sdf_tensor, sil_w=1e1)

    # 恢复最优
    z.data[:]             = best_state['z']
    global_orient.data[:] = best_state['go']
    transl.data[:]        = best_state['transl']
    betas.data[:]         = best_state['betas']
    log_scale.data[:]     = best_state['ls']

    # 最终结果
    with torch.no_grad():
        j3d_f, verts_f = smpl_from_vposer(dec, z, smpl, betas, global_orient)
        sc_f      = torch.exp(log_scale)
        pts_f     = sc_f * j3d_f + transl.unsqueeze(1)     # (1,24,3) camera space
        verts_cam = (sc_f * verts_f + transl.unsqueeze(1)).squeeze(0).cpu().numpy()  # (6890,3)
        uv_all    = omni_project_torch(pts_f, **cam).squeeze(0).cpu().numpy()        # (24,2)
        uv_aln    = uv_all[SMPL_ALIGN]   # (7,2)
        head_cam  = pts_f[0, 15, :].cpu().numpy()   # SMPL joint 15 = Head, camera space (3,)
        dist_sq_f = np.sum((uv_aln - s1['target_uv'])**2, axis=-1)
        rmse_f    = float(np.sqrt(np.mean(dist_sq_f[s1['valid_mask']])))

    logger.info(f"  最终 RMSE={rmse_f:.2f}px  depth={transl[0,2].item():.3f}m  scale={sc_f.item():.3f}")

    # 保存参数
    npz_path = os.path.join(output_dir, 'exp5_params.npz')
    np.savez(npz_path,
             z=z.detach().cpu().numpy(),
             global_orient=global_orient.detach().cpu().numpy(),
             transl=transl.detach().cpu().numpy(),
             betas=betas.detach().cpu().numpy(),
             log_scale=log_scale.detach().cpu().numpy(),
             uv_pred=uv_aln, verts_cam=verts_cam)
    logger.info(f"  参数已保存: exp5_params.npz")

    # 可视化
    save_joint_overlay(s1['img'], s1['target_uv'], s1['valid_mask'],
                       f"Step3: VPoser Fit  RMSE={rmse_f:.1f}px  Green=GT  Orange=SMPL",
                       'step3_vposer_fit.png', output_dir, pred_uv=uv_aln)
    return {'verts_cam': verts_cam, 'uv_all': uv_all,
            'verts_cam_head': head_cam, 'rmse': rmse_f,
            'transl': transl.detach().cpu().numpy()}
