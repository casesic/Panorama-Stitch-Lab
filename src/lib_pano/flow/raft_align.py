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
# Assuming this file is at src/lib_pano/stitch/raft_align.py
# RAFT core is at tools/RAFT/core
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
RAFT_CORE_PATH = os.path.join(PROJECT_ROOT, "tools/RAFT/core")
RAFT_MODELS_PATH = os.path.join(PROJECT_ROOT, "tools/RAFT/models")

if RAFT_CORE_PATH not in sys.path:
    sys.path.append(RAFT_CORE_PATH)

try:
    from raft import RAFT
    from utils.utils import InputPadder
except ImportError as e:
    raise ImportError(f"Could not import RAFT modules from {RAFT_CORE_PATH}. Error: {e}")

# =============================================================================
# Utility Functions
# =============================================================================

def flow_warp(image_source, flow, mode='bilinear', padding_mode='zeros'):
    """
    Warp an image using the optical flow field.
    Args:
        image_source (torch.Tensor): Source image [B, C, H, W]
        flow (torch.Tensor): Optical flow [B, 2, H, W] (from target to source)
        mode (str): Interpolation mode ('bilinear', 'nearest')
        padding_mode (str): Padding mode ('zeros', 'border', 'reflection')
    Returns:
        torch.Tensor: Warped image
    """
    B, C, H, W = image_source.size()
    
    # Create mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    
    if image_source.is_cuda:
        grid = torch.cat((xx, yy), 1).float().cuda()
    else:
        grid = torch.cat((xx, yy), 1).float()

    # Apply flow to grid
    vgrid = grid + flow

    # Normalize to [-1, 1] for grid_sample
    # 2 * x / (W-1) - 1
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    # Permute to [B, H, W, 2]
    vgrid = vgrid.permute(0, 2, 3, 1)

    # Sample
    warped = F.grid_sample(image_source, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return warped

# =============================================================================
# RAFT Aligner Class
# =============================================================================

class RAFTAligner:
    def __init__(self, model_path=None, device=None, args=None):
        """
        Initialize RAFT Aligner.
        Args:
            model_path (str): Path to the .pth model file. Defaults to 'raft-things.pth' in tools/RAFT/models.
            device (str): 'cuda' or 'cpu'. Defaults to cuda if available.
            args (Namespace): Optional override for RAFT args.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_path is None:
            # Default to raft-things.pth
            model_path = os.path.join(RAFT_MODELS_PATH, 'raft-things.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RAFT model not found at: {model_path}\nPlease download it using tools/RAFT/download_models.sh")

        # Setup RAFT args
        if args is None:
            self.args = Namespace(
                small=False,
                mixed_precision=False,
                alternate_corr=False,
                dropout=0
            )
        else:
            self.args = args

        # Check if model is 'small' based on filename if not specified
        if 'small' in model_path and not self.args.small:
            print("Warning: Model path contains 'small' but args.small is False. Enabling small mode.")
            self.args.small = True

        self.model = torch.nn.DataParallel(RAFT(self.args))
        
        # Load state dict
        # Note: Originals were saved with DataParallel, keys might have 'module.' prefix
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle state dict keys if needed (RAFT code wraps in DataParallel usually)
        # If loading into DataParallel, keys should represent that structure or be adaptable
        self.model.load_state_dict(state_dict)
        
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, img_np):
        """
        Convert numpy image (H, W, 3) to torch tensor (1, 3, H, W).
        """
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def align(self, img_source, img_target, iters=20, return_flow=False):
        """
        Align source image to target image using Optical Flow.
        
        Args:
            img_target (np.ndarray): Target image (Base), [H, W, 3], BGR or RGB (consistent).
            img_source (np.ndarray): Source image (To be warped), [H, W, 3].
            iters (int): Number of flow iterations.
            return_flow (bool): If True, return (warped_img, flow). Else return warped_img.
        
        Returns:
            np.ndarray: Warped source image [H, W, 3].
        """
        # Ensure dimensions match (RAFT requirement)
        if img_target.shape[:2] != img_source.shape[:2]:
            h, w = img_target.shape[:2]
            print(f"[RAFTAligner] Warning: Input size mismatch. Resizing source {img_source.shape[:2]} to target {img_target.shape[:2]}")
            img_source = cv2.resize(img_source, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor
        tensor_target = self.preprocess_image(img_target)
        tensor_source = self.preprocess_image(img_source)

        # Pad images to be divisible by 8
        padder = InputPadder(tensor_target.shape)
        tensor_target, tensor_source = padder.pad(tensor_target, tensor_source)

        # Compute flow
        with torch.no_grad():
            # flow_low, flow_up = model(image1, image2, ...)
            # image1=Target, image2=Source -> Flow output is vectors to move pixels in Target to match Source
            # Effectively, flow(x) = pos_in_source - pos_in_target
            # So Target(x) corresponds to Source(x + flow(x))
            _, flow_up = self.model(tensor_target, tensor_source, iters=iters, test_mode=True)

        # Warp source image to match target
        warped_source_tensor = flow_warp(tensor_source, flow_up)

        # Unpad
        warped_source_tensor = padder.unpad(warped_source_tensor)
        
        # Convert back to numpy
        # [1, C, H, W] -> [H, W, C]
        warped_img = warped_source_tensor[0].permute(1, 2, 0).cpu().numpy()
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)

        if return_flow:
            flow_np = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
            return warped_img, flow_np
        else:
            return warped_img
            
    def align_with_masking(self, img_source, img_target, iters=20, return_flow=False, 
                          mask_people=True, dilate_pixels=30, mask_threshold=0.2):
        """
        Align allowing automatic masking of people to prevent ghosting.
        Uses background features for alignment.
        """
        src_run = img_source
        dst_run = img_target
        
        if mask_people:
            # Lazy import to avoid circular dependency or requirement if not used
            from lib_pano.vis.masking import remove_person
            print(f"[RAFTAligner] Masking people (dilate={dilate_pixels}, thresh={mask_threshold})...")
            src_run, _ = remove_person(img_source, device=self.device, 
                                      dilate_pixels=dilate_pixels, threshold=mask_threshold)
            dst_run, _ = remove_person(img_target, device=self.device, 
                                      dilate_pixels=dilate_pixels, threshold=mask_threshold)
        # Run RAFT on (possibly masked) images
        # We always set return_flow=True here because we might need to warp the ORIGINAL image manually
        # if we masked the input.
        warped_masked, flow = self.align(src_run, dst_run, iters=iters, return_flow=True)
        
        if not mask_people:
            # If no masking was done, warped_masked IS the result we want
            if return_flow:
                return warped_masked, flow, src_run, dst_run
            else:
                return warped_masked, src_run, dst_run
        else:
            # If masking was done, 'warped_masked' has black holes where people were.
            # We usually want to apply the calculated flow to the ORIGINAL source image.
            
            # Re-run warp manually on original source
            # Convert original source to tensor
            tensor_source = self.preprocess_image(img_source)
            
            # Need to handle padding same as align()
            # But let's assume we can reuse standard flow warping logic?
            # flow is numpy (H, W, 2).
            # We need to use grid_sample.
            
            # To simplify, we can just call internal warping logic if we refactor, 
            # or just replicate the minimum needed:
            
            H, W = img_source.shape[:2]
            # flow upsampling is handled inside align() -> return_flow gives unpadded flow at original res.
            # So 'flow' is (H, W, 2).
            
            flow_tensor = torch.from_numpy(flow).permute(2,0,1).unsqueeze(0).to(self.device).float() # [1, 2, H, W]
            
            warped_original_tensor = flow_warp(tensor_source, flow_tensor)
            
            warped_original = warped_original_tensor[0].permute(1, 2, 0).cpu().numpy()
            warped_original = np.clip(warped_original, 0, 255).astype(np.uint8)
            
            if return_flow:
                return warped_original, flow, src_run, dst_run
            else:
                return warped_original, src_run, dst_run