from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch


class DenseKPLoss:
    def __init__(self, data_dir: Path, weight: float = 0.01, id_mask: np.ndarray | None = None):
        self._data_dir = data_dir
        self.weight = weight
        self._regressor: torch.Tensor | None = None
        self.id_mask = id_mask  # [138] bool, or None (use all 138 points)

    def _get_regressor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._regressor is not None:
            return self._regressor.to(device=device, dtype=dtype)

        with open(self._data_dir / "downsample_mat.pkl", "rb") as f:
            downsample = pickle.load(f)
        if not torch.is_tensor(downsample):
            downsample = torch.as_tensor(downsample)

        with open(self._data_dir / "smplx2smpl.pkl", "rb") as f:
            smplx2smpl = pickle.load(f)

        R = downsample.float() @ torch.from_numpy(smplx2smpl["matrix"]).float()
        assert R.shape == (138, 10475), f"Unexpected DenseKP regressor shape: {R.shape}"
        self._regressor = R
        return R.to(device=device, dtype=dtype)

    def compute(
        self,
        verts_world: torch.Tensor,
        densekp_list: list | None,
        project_fn,
        W: int,
        H: int,
    ) -> torch.Tensor:
        if densekp_list is None:
            return verts_world.new_zeros(())

        device = verts_world.device
        dtype = verts_world.dtype
        R = self._get_regressor(device, dtype)

        densekp_pts_3d = torch.einsum("kv,tvc->tkc", R, verts_world)
        proj = project_fn(densekp_pts_3d, W, H)

        id_mask_t = None
        if self.id_mask is not None:
            id_mask_t = torch.from_numpy(self.id_mask).to(device=device)

        losses = []
        for t, densekp in enumerate(densekp_list):
            if densekp is None:
                continue
            valid = densekp.valid
            if id_mask_t is not None:
                valid = valid & id_mask_t.cpu().numpy()
            if not valid.any():
                continue

            kp_t = torch.from_numpy(densekp.keypoints_2d).to(device=device, dtype=dtype)
            valid_t = torch.from_numpy(valid).to(device=device)

            diff = proj[t][valid_t] - kp_t[valid_t]
            losses.append(torch.norm(diff, dim=1).mean())

        if not losses:
            return verts_world.new_zeros(())
        return torch.stack(losses).mean()
