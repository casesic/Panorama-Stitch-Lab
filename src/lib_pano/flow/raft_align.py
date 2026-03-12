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

# =============================================================================
# Utility Functions
# =============================================================================

def flow_warp(image_source: torch.Tensor, flow: torch.Tensor, mode: str = 'bilinear', padding_mode: str = 'zeros') -> torch.Tensor:
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
    
    # Create mesh grid  target coordinates
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    
    # grid[b, 0, y, x] = x  
    # grid[b, 1, y, x] = y  
    if image_source.is_cuda:
        grid = torch.cat((xx, yy), 1).float().cuda()
    else:
        grid = torch.cat((xx, yy), 1).float()

    # Apply flow to grid  source coordinates  
    vgrid = grid + flow

    # Normalize to [-1, 1] for grid_sample
    # 2 * x / (W-1) - 1
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    # Permute to [B, H, W, 2]
    vgrid = vgrid.permute(0, 2, 3, 1)
    
    warped = F.grid_sample(image_source, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return warped

class SafeInputPadder:
    """ Pads images such that dimensions are divisible by 8 and at least min_size """
    def __init__(self, dims, mode='sintel', min_size=64):
        self.ht, self.wd = dims[-2:]
        
        # Calculate target height
        # Ensure divisible by 8
        target_ht = ((self.ht // 8) + 1) * 8
        # Ensure at least min_size
        if target_ht < min_size:
            # We assume min_size is multiple of 8 or we round up to next multiple of 8
            target_ht = ((min_size - 1) // 8 + 1) * 8
        
        pad_ht = target_ht - self.ht
        
        # Calculate target width
        target_wd = ((self.wd // 8) + 1) * 8
        if target_wd < min_size:
            target_wd = ((min_size - 1) // 8 + 1) * 8
            
        pad_wd = target_wd - self.wd

        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

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
            # Default to raft-kitti.pth (outdoor/building scenes)
            model_path = os.path.join(RAFT_MODELS_PATH, 'raft-kitti.pth')
        
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

    def align(self, img_source, img_target, iters=20, return_flow=False, bidirectional=False):
        """
        Align source image to target image using Optical Flow.
        
        Args:
            img_target (np.ndarray): Target image (Base), [H, W, 3], BGR or RGB (consistent).
            img_source (np.ndarray): Source image (To be warped), [H, W, 3].
            iters (int): Number of flow iterations.
            return_flow (bool): If True, return (warped_img, flow). Else return warped_img.
            bidirectional (bool): If True, compute forward/backward flow and return consistency mask.
                                  ONLY affects return values if return_flow=True.
                                  Returns (warped_img, flow, valid_mask)
        
        Returns:
            np.ndarray: Warped source image [H, W, 3]. 
                        OR tuple if return_flow is True.
        """
        # Ensure dimensions match (RAFT requirement)
        if img_target.shape[:2] != img_source.shape[:2]:
            h, w = img_target.shape[:2]
            print(f"[RAFTAligner] Warning: Input size mismatch. Resizing source {img_source.shape[:2]} to target {img_target.shape[:2]}")
            img_source = cv2.resize(img_source, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor
        tensor_target = self.preprocess_image(img_target)
        tensor_source = self.preprocess_image(img_source)

        # Pad images to be divisible by 8 and satisfy minimum size
        padder = SafeInputPadder(tensor_target.shape)
        tensor_target, tensor_source = padder.pad(tensor_target, tensor_source)

        # Compute flow
        with torch.no_grad():
            # eg: source(50, 50) match target(100, 50)  flow (-50, 0)
            # model(target, source) → flow (from target to source)
            # flow_warp(source, flow) → new_image
            # new_image content: for each position p, sample from source at [p + flow(p)]
            # result: new_image ≈ target
            _, flow_up = self.model(tensor_target, tensor_source, iters=iters, test_mode=True)
            
            valid_mask_np = None
            if bidirectional and return_flow:
                # Calculate reverse flow: Source -> Target (vectors at Source grid pointing to Target)
                _, flow_rev = self.model(tensor_source, tensor_target, iters=iters, test_mode=True)
                
                # Check consistency
                '''
                P' = Q + R(Q)
                = [P + F(P)] + R(P + F(P))
                we want :P' ≈ P
                [P + F(P)] + R(P + F(P)) ≈ P
                smpilfy: F(P) + R(P + F(P)) ≈ 0
                '''
                flow_rev_warped = flow_warp(flow_rev, flow_up)
                diff = flow_up + flow_rev_warped
                
                # Calculate magnitude of difference
                diff_norm = torch.norm(diff, dim=1, keepdim=True) # [1, 1, H, W]
                
                # Create mask of valid pixels based on consistency threshold
                consistency_thresh = 3.0
                valid_mask = (diff_norm < consistency_thresh).float()
                
                valid_mask_np = padder.unpad(valid_mask)[0].permute(1, 2, 0).cpu().numpy() # [H, W, 1]
                # Squeeze to [H, W] or keep [H, W, 1] depending on preference. Keeping [H, W] usually easier.
                valid_mask_np = valid_mask_np[:, :, 0]

        # Warp source image to match target
        warped_source_tensor = flow_warp(tensor_source, flow_up)

        # Unpad
        warped_source_tensor = padder.unpad(warped_source_tensor)
        
        # Convert back to numpy
        # [1, C, H, W] -> [H, W, C]
        warped_img = warped_source_tensor[0].permute(1, 2, 0).cpu().numpy()
        warped_img = np.nan_to_num(warped_img, nan=0.0, posinf=255.0, neginf=0.0)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)

        if return_flow:
            flow_np = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
            # Ensure flow is valid
            flow_np = np.nan_to_num(flow_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            if bidirectional:
                return warped_img, flow_np, valid_mask_np
            else:
                return warped_img, flow_np
        else:
            return warped_img
            
    def align_with_masking(self, img_source, img_target, iters=20, return_flow=False, 
                          mask_people=True, dilate_pixels=30, mask_threshold=0.2, bidirectional=False):
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
        align_result = self.align(src_run, dst_run, iters=iters, return_flow=True, bidirectional=bidirectional)
        
        if bidirectional:
             warped_masked, flow, valid_mask = align_result
        else:
             warped_masked, flow = align_result
        
        if not mask_people:
            # If no masking was done, warped_masked IS the result we want
            if return_flow:
                if bidirectional:
                     return warped_masked, flow, valid_mask, src_run, dst_run
                else:
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
            warped_original = np.nan_to_num(warped_original, nan=0.0, posinf=255.0, neginf=0.0)
            warped_original = np.clip(warped_original, 0, 255).astype(np.uint8)
            
            if return_flow:
                if bidirectional:
                    return warped_original, flow, valid_mask, src_run, dst_run
                else:
                    return warped_original, flow, src_run, dst_run
            else:
                return warped_original, src_run, dst_run