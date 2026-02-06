import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from argparse import Namespace

# =============================================================================
# Setup Paths for PixelStitch
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
PIXELSTITCH_ROOT = os.path.join(PROJECT_ROOT, "tools/PixelStitch")

# Add PixelStitch to sys.path to allow importing its modules
if PIXELSTITCH_ROOT not in sys.path:
    sys.path.append(PIXELSTITCH_ROOT)

try:
    from modules.models.raft.raft import RAFTStitch
except ImportError as e:
    # Fallback or friendly error if environment is not set up
    print(f"[PixelStitchAligner] Error importing RAFTStitch: {e}")
    RAFTStitch = None

class PixelStitchAligner:
    def __init__(self, model_path=None, device=None, args=None):
        if RAFTStitch is None:
            raise ImportError("RAFTStitch could not be imported. Please check tools/PixelStitch installation.")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_path is None:
            model_path = os.path.join(PIXELSTITCH_ROOT, 'checkpoints', 'ckpt.pth')
        
        if not os.path.exists(model_path):
            print(f"[PixelStitchAligner] Warning: Model not found at {model_path}")
            print("Please ensure you have trained or downloaded the PixelStitch weights to tools/PixelStitch/checkpoints/ckpt.pth")

        # Default Args matching PixelStitch test.py
        if args is None:
            self.args = Namespace(
                small=False,
                mixed_precision=False,
                alternate_corr=False,
                dropout=0,
                corr_levels=4, 
                corr_radius=4, 
                ftdown=8,
                iters=4,
                corr_kernel_size=13 # This is critical to match PixelStitch defaults
            )
            # RAFTStitch logic usually calculates radius from corr_kernel_size if radius not explicitly set?
            # Actually RAFTStitch code says: 
            # if 'radius' in args: args.corr_radius = args.radius
            # else: args.corr_radius = 3 (default inside RAFTStitch init)
            
            # HOWEVER, test.py does: args.radius = args.corr_kernel_size // 2
            # default corr_kernel_size is 13 -> radius = 6
            
            # The error shows mismatch: 
            # checkpoint expects [256, 169, 1, 1] for convc1.weight -> 169 indicates kernel area or channels? 
            # 169 = 13*13. So kernel size is 13.
            # Current model has [256, 49, 1, 1] -> 49 = 7*7. So current kernel size/radius implied is smaller.
            
            # RAFT CorrBlock usually outputs channels = (2*r + 1)^2 * levels
            # But the error is in update_block.encoder ... wait.
            # update_block.flow_head.conv2.weight
            
            # Let's fix radius to match checkpoint (likely 13x13 window -> radius 6)
            self.args.radius = 6 # derived from corr_kernel_size=13 // 2
        else:
            self.args = args

        self.model = RAFTStitch(self.args)
        
        if os.path.exists(model_path):
            print(f"[PixelStitchAligner] Loading weights from {model_path} ...")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 兼容 DataParallel 或者是 原生权重
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            print("[PixelStitchAligner] Running with RANDOM weights! (Checkpoint not found)")
        
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, img_np):
        """
        Resize/Pad image to be divisible by 8 for PixelStitch
        """
        h, w = img_np.shape[:2]
        # Calculate padding needed
        dh = ((h - 1) // 8 + 1) * 8 - h
        dw = ((w - 1) // 8 + 1) * 8 - w
        
        img = torch.from_numpy(img_np).permute(2, 0, 1).float() # [C, H, W]
        # Pad: (left, right, top, bottom)
        img = F.pad(img, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return img[None].to(self.device), dh, dw

    def align(self, img_source, img_target, iters=5, return_flow=False, bidirectional=False):
        """
        Args:
            img_source: Source image (To be warped)
            img_target: Target image (Reference)
        Returns:
            warped_source, flow (optional), valid_mask (optional)
        """
        # Preprocess
        tensor_target, dh_t, dw_t = self.preprocess_image(img_target)
        tensor_source, dh_s, dw_s = self.preprocess_image(img_source)
        
        # Ensure input shapes match (usually crop_source and crop_target are quite similar)
        # If sizes differ drastically, naive concatenation will fail.
        # Assuming crops from extract_overlap_roi are same size or we handle it.
        if tensor_target.shape != tensor_source.shape:
             # Resize source tensor to match target tensor structure if needed
             # For simpler integration, we rely on caller giving same sized crops
             pass

        # PixelStitch: forward expects [2, C, H, W]
        imagePair = torch.cat([tensor_target, tensor_source], dim=0)

        with torch.no_grad():
            # forward() computes features for both and dual correlation
            # Batch 0: Target -> Source Flow
            # Batch 1: Source -> Target Flow
            flows, match_pred = self.model(imagePair, iters=iters, dual=True)
            
            final_flow = flows[-1] # [2, 2, H, W]
            
            # flow_t2s: Defined on Target grid, pointing to Source pixels.
            # Used to warp Source to match Target.
            flow_t2s = final_flow[0:1] 
            
            flow_s2t = final_flow[1:2]
            
            valid_mask_np = None
            if bidirectional:
                # Need to warp flow_s2t to Target frame using flow_t2s
                
                def flow_warp_torch(img, flow):
                    B, C, H, W = img.size()
                    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
                    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
                    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
                    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
                    grid = torch.cat((xx, yy), 1).float().to(img.device)
                    vgrid = grid + flow
                    
                    # Normalize
                    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
                    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
                    vgrid = vgrid.permute(0, 2, 3, 1)
                    
                    return F.grid_sample(img, vgrid, mode='bilinear', align_corners=True)

                flow_s2t_warped = flow_warp_torch(flow_s2t, flow_t2s)
                
                # Consistency check: Fwd + Bwd_warped approx 0
                diff = flow_t2s + flow_s2t_warped
                diff_norm = torch.norm(diff, dim=1, keepdim=True)
                
                # Threshold usually 1.0 ~ 3.0
                valid_mask = (diff_norm < 3.0).float()
                
                # Unpad mask
                h, w = diff_norm.shape[2:]
                valid_mask = valid_mask[..., dh_t//2 : h - (dh_t - dh_t//2), dw_t//2 : w - (dw_t - dw_t//2)]
                valid_mask_np = valid_mask[0, 0].cpu().numpy() # [H_orig, W_orig]

        # Warp Source to Target using flow_t2s
        B, _, H, W = flow_t2s.shape
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(self.device)
        vgrid = grid + flow_t2s
        
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        
        warped_source_tensor = F.grid_sample(tensor_source, vgrid, mode='bilinear', align_corners=True)

        # Unpad Image
        h, w = warped_source_tensor.shape[2:]
        warped_source_tensor = warped_source_tensor[..., dh_t//2 : h - (dh_t - dh_t//2), dw_t//2 : w - (dw_t - dw_t//2)]
        
        warped_img = warped_source_tensor[0].permute(1, 2, 0).cpu().numpy()
        warped_img = np.nan_to_num(warped_img, nan=0.0, posinf=255.0, neginf=0.0)
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)

        if return_flow:
            # Unpad Flow
            h, w = flow_t2s.shape[2:]
            flow_out = flow_t2s[..., dh_t//2 : h - (dh_t - dh_t//2), dw_t//2 : w - (dw_t - dw_t//2)]
            flow_np = flow_out[0].permute(1, 2, 0).cpu().numpy()
            
            if bidirectional:
                return warped_img, flow_np, valid_mask_np
            else:
                return warped_img, flow_np
        
        return warped_img

    def align_with_masking(self, img_source, img_target, iters=5, return_flow=False, 
                          mask_people=True, dilate_pixels=30, mask_threshold=0.2, bidirectional=False):
        """
        High-level wrapper to support 'align_with_masking' interface
        """
        src_run = img_source
        dst_run = img_target
        
        if mask_people:
             from lib_pano.vis.masking import remove_person
             src_run, _ = remove_person(img_source, device=self.device, dilate_pixels=dilate_pixels, threshold=mask_threshold)
             dst_run, _ = remove_person(img_target, device=self.device, dilate_pixels=dilate_pixels, threshold=mask_threshold)

        res = self.align(src_run, dst_run, iters=iters, return_flow=True, bidirectional=bidirectional)
        
        if bidirectional:
            warped, flow, mask = res
        else:
            warped, flow = res
            mask = None
            
        # If we need to re-warp the original image (because src_run was masked/blacked out)
        if mask_people:
            # Re-run warp logic manually on original source
            # Reuse flow calculated from masked images
            
            # Need to invoke the manual warp part again. 
            # Simplified: we just call align again on original images ONLY for specific flow? 
            # No that's wasteful.
            # Let's just do the warp quickly here.
            
            tensor_source, dh_s, dw_s = self.preprocess_image(img_source)
            tensor_target, dh_t, dw_t = self.preprocess_image(img_target) # Just for padding dims
            
            # Flow is already unpadded in output 'flow', so we need to RE-PAD it or use unpadded warp?
            # Easiest: Used Unpadded Flow to warp Unpadded Source.
            
            # 'flow' is numpy (H, W, 2). Convert to grid.
            H, W = flow.shape[:2]
            B = 1
            
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            img_tensor = torch.from_numpy(img_source).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
            
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float().to(self.device)
            vgrid = grid + flow_tensor
            
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
            vgrid = vgrid.permute(0, 2, 3, 1)
            
            warped_t = F.grid_sample(img_tensor, vgrid, mode='bilinear', align_corners=True)
            warped = warped_t[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        if return_flow:
             if bidirectional:
                 return warped, flow, mask, src_run, dst_run
             else:
                 return warped, flow, src_run, dst_run
        return warped, src_run, dst_run
