import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from argparse import Namespace
import warnings

# Suppress warnings from torch/raft
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Setup Paths for RAFT
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../../'))
RAFT_CORE_PATH = os.path.join(PROJECT_ROOT, 'tools/RAFT/core')
RAFT_MODELS_PATH = os.path.join(PROJECT_ROOT, 'tools/RAFT/models')

if RAFT_CORE_PATH not in sys.path:
    sys.path.append(RAFT_CORE_PATH)

try:
    from raft import RAFT
    from utils.utils import InputPadder
except ImportError as e:
    raise ImportError(f"Could not import RAFT modules from {RAFT_CORE_PATH}. Error: {e}")


def _flow_warp(image_source: torch.Tensor, flow: torch.Tensor, mode: str = 'bilinear', padding_mode: str = 'zeros') -> torch.Tensor:
    """
    Warp an image using the optical flow field.
        torch.Tensor: Warped image
    """
    B, C, H, W = image_source.size()

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)

    # grid[:, 0, :, :] 存 x, grid[:, 1, :, :] 存 y 
    if image_source.is_cuda:
        grid = torch.cat((xx, yy), 1).float().cuda()
    else:
        grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flow

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)  # B, H, W, 2

    warped = F.grid_sample(image_source, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return warped


class SafeInputPadder:
    """ Pads images such that dimensions are divisible by 8 and at least min_size """
    def __init__(self, dims, mode='sintel', min_size=64):
        self.ht, self.wd = dims[-2:]

        target_ht = ((self.ht + 7) // 8) * 8
        if target_ht < min_size:
            target_ht = ((min_size - 1) // 8 + 1) * 8

        pad_ht = target_ht - self.ht

        target_wd = ((self.wd + 7) // 8) * 8
        if target_wd < min_size:
            target_wd = ((min_size - 1) // 8 + 1) * 8

        pad_wd = target_wd - self.wd

        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class RAFTFlow:
    def __init__(self, model_path=None, device=None, iters=20, consistency_thresh=20.0):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_path is None:
            model_path = os.path.join(RAFT_MODELS_PATH, 'raft-things.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RAFT model not found at: {model_path}\nPlease download it using tools/RAFT/download_models.sh")

        self.args = Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False,
            dropout=0
        )

        if 'small' in model_path and not self.args.small:
            print("Warning: Model path contains 'small' but args.small is False. Enabling small mode.")
            self.args.small = True

        self.model = torch.nn.DataParallel(RAFT(self.args))

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

        self.iters = iters
        self.consistency_thresh = consistency_thresh

    def _preprocess(self, img_np):
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def _run_raft(self, t1, t2):
        _, flow_up = self.model(t1, t2, iters=self.iters, test_mode=True)
        return flow_up

    def _consistency_check(self, flow_tgt2src, flow_src2tgt):
        # 将 flow_src2tgt 沿 flow_tgt2src 方向 warp 到 target 坐标系
        # 一致性条件：flow_tgt2src + warped(flow_src2tgt) ≈ 0
        flow_src2tgt_warped = _flow_warp(flow_src2tgt, flow_tgt2src)
        diff = flow_tgt2src + flow_src2tgt_warped
        diff_norm = torch.norm(diff, dim=1, keepdim=True)
        return (diff_norm < self.consistency_thresh).float()

    def _postprocess_flow(self, flow_tensor, padder, scale, orig_size):
        flow_np = padder.unpad(flow_tensor)[0].permute(1, 2, 0).cpu().numpy()
        flow_np = np.nan_to_num(flow_np, nan=0.0, posinf=0.0, neginf=0.0)

        if scale < 1.0:
            orig_h, orig_w = orig_size
            flow_np = cv2.resize(flow_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            flow_np = flow_np / scale

        return flow_np

    def _postprocess_mask(self, mask_tensor, padder, scale, orig_size):
        if mask_tensor is None:
            return None

        mask_np = padder.unpad(mask_tensor)[0].permute(1, 2, 0).cpu().numpy()
        mask_np = np.nan_to_num(mask_np, nan=0.0, posinf=0.0, neginf=0.0)
        mask_np = mask_np[:, :, 0]

        if scale < 1.0:
            orig_h, orig_w = orig_size
            mask_np = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        return mask_np

    def compute(self, img_source, img_target, bidirectional=True):
        if img_target.shape[:2] != img_source.shape[:2]:
            h, w = img_target.shape[:2]
            print(f"[RAFTFlow] Warning: Input size mismatch. Resizing source {img_source.shape[:2]} to target {img_target.shape[:2]}")
            img_source = cv2.resize(img_source, (w, h), interpolation=cv2.INTER_LINEAR)

        orig_h, orig_w = img_target.shape[:2]
        orig_size = (orig_h, orig_w)

        raft_max_width = 1920
        scale = min(1.0, raft_max_width / orig_w)
        if scale < 1.0:
            raft_h = int(orig_h * scale)
            raft_w = int(orig_w * scale)
            img_target = cv2.resize(img_target, (raft_w, raft_h), interpolation=cv2.INTER_LINEAR)
            img_source = cv2.resize(img_source, (raft_w, raft_h), interpolation=cv2.INTER_LINEAR)
            print(f"[RAFTFlow] Downscaled input from {orig_w}x{orig_h} to {raft_w}x{raft_h} for RAFT")

        t_tgt = self._preprocess(img_target)
        t_src = self._preprocess(img_source)

        padder = SafeInputPadder(t_tgt.shape)
        t_tgt, t_src = padder.pad(t_tgt, t_src)

        with torch.no_grad():
            # model(target, source) → 每个 target 像素对应的 source 位移
            flow_tgt2src = self._run_raft(t_tgt, t_src)
            valid_mask = None

            if bidirectional:
                # model(source, target) → 反向流，用于一致性检验
                flow_src2tgt = self._run_raft(t_src, t_tgt)
                valid_mask = self._consistency_check(flow_tgt2src, flow_src2tgt)

        flow_np = self._postprocess_flow(flow_tgt2src, padder, scale, orig_size)
        valid_mask_np = self._postprocess_mask(valid_mask, padder, scale, orig_size)
        return flow_np, valid_mask_np
