"""SMPL 前向 LBS 模型 + 旋转工具 + OBJ 导出 + 上半身裁剪"""

import os, pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── 旋转工具 ────────────────────────────────────────────────────

def axis_angle_to_matrix_torch(aa: torch.Tensor) -> torch.Tensor:
    """(...,3) → (...,3,3) Rodrigues，可微"""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis  = aa / angle
    c, s  = angle.cos(), angle.sin()
    t     = 1 - c
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    R = torch.cat([
        t*x*x+c,   t*x*y-s*z, t*x*z+s*y,
        t*x*y+s*z, t*y*y+c,   t*y*z-s*x,
        t*x*z-s*y, t*y*z+s*x, t*z*z+c,
    ], dim=-1)
    return R.reshape(aa.shape[:-1] + (3, 3))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """6D → 旋转矩阵 (...,3,3) — Zhou et al. CVPR2019"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


# ── SMPL 前向模型 ───────────────────────────────────────────────

class SmplFull(nn.Module):
    """完整 SMPL LBS，输出 root-centered joints/verts（可微）"""

    def __init__(self, pkl_path: str):
        super().__init__()
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')

        def _t(x): return torch.tensor(np.array(x, dtype=np.float32))

        J_reg = raw['J_regressor']
        if hasattr(J_reg, 'todense'): J_reg = np.array(J_reg.todense())
        self.register_buffer('J_regressor', _t(J_reg))           # (24, 6890)

        sd = np.array(raw['shapedirs'], dtype=np.float32)
        if sd.ndim == 2: sd = sd.reshape(6890, 3, -1)
        self.register_buffer('shapedirs', _t(sd[:, :, :10]))     # (6890, 3, 10)
        self.register_buffer('v_template', _t(raw['v_template']))# (6890, 3)

        # posedirs shape: (6890,3,207) → reshape to (6890*3, 207) for einsum
        pd = np.array(raw['posedirs'], dtype=np.float32)
        if pd.ndim == 3 and pd.shape[0] == 6890:
            pd = pd.reshape(6890 * 3, pd.shape[2])               # (6890*3, 207)
        self.register_buffer('posedirs', _t(pd))                  # (6890*3, 207)
        self.register_buffer('lbs_weights', _t(raw['weights']))   # (6890, 24)
        self.faces = np.array(raw['f'], dtype=np.int32)           # (13776, 3) CPU only

        kintree = raw['kintree_table']
        self.parents = kintree[0].tolist()
        self.parents[0] = -1

    def _fk(self, J, R):  # J:(B,24,3) R:(B,24,3,3) → list[24] of (B,4,4)
        B, dev = J.shape[0], J.device
        G_list = []
        for i in range(24):
            p = self.parents[i]
            if p < 0:
                Gi = torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1)
                Gi[:, :3, :3] = R[:, i]
                Gi[:, :3,  3] = J[:, i]
            else:
                Gp = G_list[p]
                loc = torch.eye(4, device=dev).unsqueeze(0).repeat(B, 1, 1)
                loc[:, :3, :3] = R[:, i]
                loc[:, :3,  3] = J[:, i] - J[:, p]
                Gi = Gp @ loc
            G_list.append(Gi)
        return G_list

    def forward(self, betas, global_orient_R, body_pose_R):
        """→ joints_c (B,24,3), verts_c (B,6890,3)，骨盆为原点"""
        B, dev = betas.shape[0], betas.device

        # 1. Shape + rest joints
        v_shaped = self.v_template + torch.einsum('bj,vkj->bvk', betas, self.shapedirs)
        J = torch.einsum('jv,bvk->bjk', self.J_regressor, v_shaped)
        # 2. Pose blend shapes
        R_full    = torch.cat([global_orient_R.unsqueeze(1), body_pose_R], dim=1)
        pose_feat = (body_pose_R - torch.eye(3, device=dev).view(1,1,3,3)).reshape(B, -1)
        v_posed   = v_shaped + torch.einsum('vp,bp->bv', self.posedirs, pose_feat).reshape(B,6890,3)
        # 3. FK + LBS rest-pose correction: G_j^* = G_j - G_j[:,:3,:3] @ J_j
        G_list  = self._fk(J, R_full)
        G_mats  = torch.stack(G_list, dim=1)
        G_corr  = G_mats.clone()
        G_corr[:,:,:3,3] = (G_mats[:,:,:3,:3] @ J[:,:,:,None]).squeeze(-1).neg() + G_mats[:,:,:3,3]
        T       = torch.einsum('vj,bjpq->bvpq', self.lbs_weights, G_corr)
        v_homo  = torch.cat([v_posed, torch.ones(B,6890,1,device=dev)], dim=-1)
        verts   = torch.einsum('bvpq,bvq->bvp', T, v_homo)[...,:3]
        # 4. Root-center
        J_root   = J[:,0:1,:]
        joints_c = torch.stack([G_list[i][:,:3,3] for i in range(24)], 1) - J_root
        verts_c  = verts - J_root
        return joints_c, verts_c


# ── OBJ 导出 ────────────────────────────────────────────────────

def export_obj(verts, faces, path, comment=''):
    """写出 OBJ（faces 0-indexed，自动转为 1-indexed）"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        if comment: f.write(f"# {comment}\n")
        f.write(f"# vertices: {len(verts)}  faces: {len(faces)}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    logger.info(f"  OBJ 已保存: {os.path.basename(path)}")


# ── 上半身裁剪 ──────────────────────────────────────────────────

# Spine1(3)=乳头, Spine2(6)=腹, Spine3(9)=背, 12~23=颈肩臂手
_UPPER_JOINTS = frozenset([3,6,9,12,13,14,15,16,17,18,19,20,21,22,23])


def crop_upper_body(verts, faces, lbs_weights):
    """按 LBS 主导关节裁剪上半身，返回 (verts_sub, faces_sub)"""
    # lbs_weights: (6890,24)  each vertex's skinning weights for 24 joints
    dom_joint  = lbs_weights.argmax(axis=1)                     # (6890,) 主导关节
    vert_mask  = np.array([j in _UPPER_JOINTS for j in dom_joint], dtype=bool)
    face_mask  = (vert_mask[faces[:, 0]] &
                  vert_mask[faces[:, 1]] &
                  vert_mask[faces[:, 2]])
    faces_sub  = faces[face_mask]
    old2new    = np.full(len(verts), -1, dtype=np.int32)
    old2new[vert_mask] = np.arange(vert_mask.sum())
    faces_sub  = old2new[faces_sub.reshape(-1)].reshape(-1, 3)  # again reindex faces
    verts_sub  = verts[vert_mask]
    logger.info(f"  上半身: {vert_mask.sum()}顶点/{face_mask.sum()}面 (原{len(verts)}/{len(faces)})")
    return verts_sub, faces_sub
