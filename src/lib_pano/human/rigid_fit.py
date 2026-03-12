"""刚体变换拟合（Step 2）— scale + rotation + translation"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R_scipy

from lib_pano.camera.omni_camera  import omni_project_torch
from lib_pano.human.smpl_model    import axis_angle_to_matrix_torch
from lib_pano.human.detect        import SMPL_ALIGN
from lib_pano.vis.visual_human          import save_joint_overlay

logger = logging.getLogger(__name__)


# ── Step 2: 刚体拟合 ────────────────────────────────────────────

class RigidFitter(nn.Module):
    """Exp2 刚体变换优化器（只有 scale, rot_aa, trans，无 SMPL pose）"""
    R_GL2CV = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])

    def __init__(self, joints_hmr: np.ndarray, init_depth: float = 0.5):
        super().__init__()
        jc       = (self.R_GL2CV.numpy() @ joints_hmr.T).T
        centroid = jc.mean(0)
        self.register_buffer('jc_centered',
                             torch.tensor(jc - centroid, dtype=torch.float32))
        self.register_buffer('centroid',
                             torch.tensor(centroid, dtype=torch.float32))
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.trans     = nn.Parameter(torch.tensor([0., 0., init_depth]))
        self.rot_aa    = nn.Parameter(torch.zeros(3))

    def transform(self) -> torch.Tensor:
        s = torch.exp(self.log_scale)
        R = axis_angle_to_matrix_torch(self.rot_aa)
        return s * (R @ self.jc_centered.T).T + self.trans

    def project(self, **cam) -> torch.Tensor:
        return omni_project_torch(self.transform().unsqueeze(0), **cam).squeeze(0)


def step2_rigid_fit(s1: dict, J_reg: np.ndarray, cam: dict,
                    obj_path: str, output_dir: str) -> dict:
    """Exp2 刚体拟合：从 OBJ 初始化 → 优化 scale/R/T → 匹配 2D 关节"""
    logger.info("Step 2: Exp2 刚体拟合 (scale + R + T)")
    # 加载 OBJ 顶点
    verts_hmr = []
    with open(obj_path) as f:
        for l in f:
            if l.startswith('v '):
                p = l.split(); verts_hmr.append([float(p[1]),float(p[2]),float(p[3])])
    verts_hmr  = np.array(verts_hmr, dtype=np.float32)
    joints_hmr = (J_reg @ verts_hmr)[SMPL_ALIGN]   # (7,3) OpenGL

    # 深度估算
    fx, xi = cam['fx'], cam['xi']
    f_eff  = fx / (1 + xi)
    kps17, sc17 = s1['kps17'], s1['sc17']
    if sc17[5] > 0.3 and sc17[6] > 0.3:
        pix_d = float(np.linalg.norm(kps17[5] - kps17[6]))
        # Experience value shoulder width 0.3m
        init_depth = f_eff * 0.30 / pix_d if pix_d > 1 else 0.5
    else:
        init_depth = 0.5

    target_t = torch.tensor(s1['target_uv'], dtype=torch.float32)
    vmask_t  = torch.tensor(s1['valid_mask'], dtype=torch.float32)
    fitter   = RigidFitter(joints_hmr, init_depth)
    opt      = torch.optim.Adam(fitter.parameters(), lr=5e-3)
    sch      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-4)

    for it in range(3000):
        opt.zero_grad()
        uv_pred = fitter.project(**cam)
        dist_sq = ((uv_pred - target_t)**2).sum(-1)
        loss_j  = (dist_sq * vmask_t).sum() / vmask_t.sum().clamp(1)
        z_pts   = fitter.transform()[:, 2]
        loss_z  = F.relu(0.05 - z_pts).pow(2).sum() * 5e4
        loss    = loss_j + loss_z
        loss.backward(); opt.step(); sch.step()
        if it % 600 == 0 or it == 2999:
            rmse = dist_sq.mul(vmask_t).sum().div(vmask_t.sum().clamp(1)).sqrt().item()
            logger.info(f"  [{it:4d}] RMSE={rmse:.1f}px  depth={fitter.trans[2].item():.3f}m"
                        f"  scale={fitter.log_scale.exp().item():.3f}")

    with torch.no_grad():
        smpl_uv    = fitter.project(**cam).numpy()
        joints_cam = fitter.transform().numpy()

    R_rot   = R_scipy.from_rotvec(fitter.rot_aa.detach().numpy()).as_matrix()
    R_GL2CV = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
    R_combo = R_rot @ R_GL2CV   # global_orient one step from SMPL to OpenCV camera space

    # 保存参数
    npz_path = os.path.join(output_dir, 'exp2_params.npz')
    np.savez(npz_path,
             log_scale=fitter.log_scale.detach().numpy(),
             trans=fitter.trans.detach().numpy(),
             rot_aa=fitter.rot_aa.detach().numpy(),
             centroid=fitter.centroid.numpy(),
             R_GL2CV=R_GL2CV, joints_cam=joints_cam,
             smpl_uv=smpl_uv,
             valid_mask=s1['valid_mask'].astype(np.float32))
    logger.info(f"  参数已保存: exp2_params.npz")

    # 可视化
    img    = s1['img']
    rmse_f = float(np.sqrt(np.mean([np.linalg.norm(smpl_uv[i]-s1['target_uv'][i])**2
                                     for i in range(7) if s1['valid_mask'][i]])))
    save_joint_overlay(img, s1['target_uv'], s1['valid_mask'],
                       f"Step2: Rigid Fit  RMSE={rmse_f:.1f}px  Green=GT  Orange=SMPL",
                       'step2_rigid_fit.png', output_dir, pred_uv=smpl_uv)
    logger.info(f"  RMSE={rmse_f:.1f}px")
    return {'npz_path': npz_path, 'joints_hmr': joints_hmr,
            'joints_cam': joints_cam, 'smpl_uv': smpl_uv,
            'R_combo': R_combo, 'log_scale': fitter.log_scale.item(),
            'trans': fitter.trans.detach().numpy(),
            'rot_aa': fitter.rot_aa.detach().numpy(),
            'centroid': fitter.centroid.numpy()}
