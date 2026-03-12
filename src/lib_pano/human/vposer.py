"""VPoser 编解码器 + FK 逆解 + smpl_from_vposer 辅助"""

import os, glob
import logging
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib_pano.human.smpl_model import rotation_6d_to_matrix, axis_angle_to_matrix_torch

logger = logging.getLogger(__name__)


# ── VPoser 网络 ─────────────────────────────────────────────────

class NormalDistDecoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.mu     = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x):
        return torch.distributions.Normal(self.mu(x), F.softplus(self.logvar(x)))


class VPoserEncoder(nn.Module):
    def __init__(self, num_neurons=512, latent_dim=32):
        super().__init__()
        self.encoder_net = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(21*3),
            nn.Linear(21*3, num_neurons), nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),  nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, latent_dim),
        )

    def forward(self, body_pose):
        return self.encoder_net(body_pose)


class VPoserDecoder(nn.Module):
    def __init__(self, num_neurons=512, latent_dim=32):
        super().__init__()
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, num_neurons), nn.LeakyReLU(), nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),
            nn.Linear(num_neurons, 21*6),
        )

    def forward(self, z):
        return self.decoder_net(z).view(-1, 21, 6)


def load_vposer(vposer_dir: str):
    """加载 VPoser 权重，返回 (encoder, decoder)"""
    ckpts = sorted(glob.glob(os.path.join(vposer_dir, 'snapshots', '*.ckpt')))
    best  = [c for c in ckpts if 'epoch=13' in c]
    src   = best[0] if best else ckpts[-1]
    logger.info(f"  VPoser ckpt: {os.path.basename(src)}")
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    sd   = {k[len('vp_model.'):]: v for k, v in ckpt['state_dict'].items()
            if k.startswith('vp_model.')}
    enc, dec = VPoserEncoder(), VPoserDecoder()
    enc.encoder_net.load_state_dict({k[len('encoder_net.'):]: v
                                     for k, v in sd.items() if k.startswith('encoder_net.')})
    dec.decoder_net.load_state_dict({k[len('decoder_net.'):]: v
                                     for k, v in sd.items() if k.startswith('decoder_net.')})
    enc.eval(); dec.eval()
    return enc, dec


# ── 辅助函数 ────────────────────────────────────────────────────

def smpl_from_vposer(dec, z, smpl, betas, global_orient):
    """VPoser z → (joints_c, verts_c)，复用于 run_stage 和最终导出"""
    d6    = dec(z)
    R_b21 = rotation_6d_to_matrix(d6)
    eye2  = torch.eye(3, device=z.device).view(1,1,3,3).expand(1,2,-1,-1)
    R_body = torch.cat([R_b21, eye2], dim=1)
    return smpl(betas, axis_angle_to_matrix_torch(global_orient), R_body)


# ── FK 逆解（z 热启动）─────────────────────────────────────────

def _rot_between(a, b):
    a = a / (np.linalg.norm(a)+1e-8)
    b = b / (np.linalg.norm(b)+1e-8)
    cr = np.cross(a, b); cn = np.linalg.norm(cr)
    dot = float(np.clip(np.dot(a, b), -1, 1))
    if cn < 1e-7:
        return np.eye(3, dtype=np.float32) if dot > 0 else -np.eye(3, dtype=np.float32)
    ax  = cr / cn; ang = np.arccos(dot)
    K   = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]], np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)


def fk_init_z(smpl, enc, device, obj_path: str) -> torch.Tensor:
    """OBJ FK 逆解 → VPoser z_init"""
    verts_hmr = []
    with open(obj_path) as f:
        for l in f:
            if l.startswith('v '):
                p = l.split()
                verts_hmr.append([float(p[1]), float(p[2]), float(p[3])])
    verts_hmr = np.array(verts_hmr, np.float32)

    J_reg   = smpl.J_regressor.cpu().numpy()
    v_tmpl  = smpl.v_template.cpu().numpy()
    parents = smpl.parents
    J_rest  = J_reg @ v_tmpl
    J_posed = J_reg @ verts_hmr

    children = defaultdict(list)
    for i in range(1, 24):
        children[parents[i]].append(i)

    R_world = [np.eye(3, dtype=np.float32) for _ in range(24)]
    R_local = np.zeros((21, 3), np.float32)
    for i in range(22):
        ch = children[i]
        if not ch:
            if i > 0: R_world[i] = R_world[parents[i]].copy()
            continue
        c = ch[0]
        R_world[i] = _rot_between(J_rest[c]-J_rest[i], J_posed[c]-J_posed[i])
        if i > 0:
            Rl    = R_world[parents[i]].T @ R_world[i]
            cos_a = float(np.clip((np.trace(Rl)-1)/2, -1, 1))
            ang   = np.arccos(cos_a)
            if abs(np.sin(ang)) > 1e-7:
                ax = np.array([(Rl[2,1]-Rl[1,2]), (Rl[0,2]-Rl[2,0]), (Rl[1,0]-Rl[0,1])],
                              np.float32) / (2*np.sin(ang))
                R_local[i-1] = ax / np.linalg.norm(ax) * ang

    aa_t = torch.tensor(R_local, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        dist  = enc(aa_t)
        z_raw = dist.mean
    z_norm = z_raw.norm().item()
    z_init = z_raw * (3.0 / max(z_norm, 3.0))
    logger.info(f"  FK z_init: raw_norm={z_norm:.2f} → clipped_norm={z_init.norm().item():.2f}")
    return z_init.cpu()
