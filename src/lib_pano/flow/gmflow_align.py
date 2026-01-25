import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from argparse import Namespace
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Setup Paths for GMFlow
# =============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
GMFLOW_ROOT = os.path.join(PROJECT_ROOT, "tools/GMFlow")
GMFLOW_MODELS_PATH = os.path.join(GMFLOW_ROOT, "pretrained")

if GMFLOW_ROOT not in sys.path:
    sys.path.append(GMFLOW_ROOT)

# Try dynamic import to handle potential missing library during setup
try:
    from gmflow.gmflow import GMFlow
except ImportError:
    print(f"[GMFlowAligner] Warning: Could not import GMFlow from {GMFLOW_ROOT}.")
    print("Please ensure you have downloaded https://github.com/haofeixu/gmflow to tools/GMFlow")
    GMFlow = None

# =============================================================================
# Utility Functions (Reused or Adapted)
# =============================================================================

def flow_warp(image_source, flow, mode='bilinear', padding_mode='zeros'):
    """
    Warp an image using the optical flow field.
    Same standard warping logic as in RAFT.
    """
    B, C, H, W = image_source.size()
    
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    
    if image_source.is_cuda:
        grid = torch.cat((xx, yy), 1).float().cuda()
    else:
        grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flow

    # Normalize to [-1, 1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    warped = F.grid_sample(image_source, vgrid, mode=mode, padding_mode=padding_mode, align_corners=True)
    return warped

class GMFlowAligner:
    def __init__(self, model_path=None, device=None, args=None):
        if GMFlow is None:
            raise ImportError("GMFlow library not found. Check tools/GMFlow installation.")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if model_path is None:
            # Prefer sintel model for general flow
            model_path = os.path.join(GMFLOW_MODELS_PATH, 'gmflow_sintel.pth')
        
        # Setup GMFlow args
        # GMFlow requires specific feature channels etc. 
        # We construct a minimal args namespace that matches GMFlow's expectation
        if args is None:
            self.args = Namespace(
                num_scales=1,
                feature_channels=128,
                upsample_factor=8,
                num_head=1,
                attention_type='swin',
                ffn_dim_expansion=4,
                num_transformer_layers=6,
            )
        else:
            self.args = args

        print(f"[GMFlowAligner] Loading model from {model_path}...")
        
        self.model = GMFlow(
            feature_channels=self.args.feature_channels,
            num_scales=self.args.num_scales,
            upsample_factor=self.args.upsample_factor,
            num_head=self.args.num_head,
            attention_type=self.args.attention_type,
            ffn_dim_expansion=self.args.ffn_dim_expansion,
            num_transformer_layers=self.args.num_transformer_layers,
        ).to(self.device)

        if os.path.exists(model_path):
            print(f"[GMFlowAligner] Loading weights from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            # Standard GMFlow checkpoint handling
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
             # Critical Fix: Do not proceed without weights
             raise FileNotFoundError(f"[GMFlowAligner] Model weights not found at {model_path}.\nPlease download gmflow_sintel.pth (or similar) from https://github.com/haofeixu/gmflow and place it in tools/GMFlow/pretrained/")

        self.model.eval()

    def preprocess_image(self, img_np):
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def align(self, img_source, img_target, pred_bidir_flow=False, return_flow=False):
        """
        GMFlow inference.
        """
        # Padding to multiples of 8 or 16 usually required by Transformers
        factor = 32 # Swin transformer usually likes 32
        h, w = img_target.shape[:2]
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        
        if img_target.shape[:2] != img_source.shape[:2]:
             img_source = cv2.resize(img_source, (w, h))

        # Critical Fix: Convert BGR (OpenCV) to RGB (GMFlow Expectation)
        img_target_rgb = cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB)
        img_source_rgb = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)

        image1 = self.preprocess_image(img_target_rgb) # Target (Base)
        image2 = self.preprocess_image(img_source_rgb) # Source (Warp)

        # Padding
        if pad_h > 0 or pad_w > 0:
            image1 = F.pad(image1, (0, pad_w, 0, pad_h))
            image2 = F.pad(image2, (0, pad_w, 0, pad_h))

        with torch.no_grad():
            # GMFlow forward pass
            # Returns flow output dictionary
            results_dict = self.model(image1, image2, 
                                    attn_splits_list=[2], 
                                    corr_radius_list=[-1], 
                                    prop_radius_list=[-1], 
                                    pred_bidir_flow=pred_bidir_flow)
            
            flow_up = results_dict['flow_preds'][-1] # List of flows, take last one

        # Unpad flow
        if pad_h > 0 or pad_w > 0:
            flow_up = flow_up[:, :, :h, :w]

        # Use flow to warp source
        # But wait, we warped source (image2) padded? 
        # Actually it's better to warp the unpadded original tensor with unpadded flow
        tensor_source_orig = self.preprocess_image(img_source)
        warped_source_tensor = flow_warp(tensor_source_orig, flow_up)
        
        warped_img = warped_source_tensor[0].permute(1, 2, 0).cpu().numpy()
        warped_img = np.clip(warped_img, 0, 255).astype(np.uint8)

        if return_flow:
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
            return warped_img, flow_np
        else:
            return warped_img

    def align_with_masking(self, img_source, img_target, return_flow=False, 
                          mask_people=True, dilate_pixels=30, mask_threshold=0.2):
        """
        Wrapper to support the same interface as RAFTAligner for masking people.
        """
        src_run = img_source
        dst_run = img_target
        
        if mask_people:
            from lib_pano.vis.masking import remove_person
            print(f"[GMFlowAligner] Masking people...")
            src_run, _ = remove_person(img_source, device=self.device, 
                                      dilate_pixels=dilate_pixels, threshold=mask_threshold)
            dst_run, _ = remove_person(img_target, device=self.device, 
                                      dilate_pixels=dilate_pixels, threshold=mask_threshold)

        warped_masked, flow = self.align(src_run, dst_run, return_flow=True)
        
        if not mask_people:
            if return_flow:
                return warped_masked, flow, src_run, dst_run
            else:
                return warped_masked, src_run, dst_run
        else:
            # Warp original image using calculated flow
            tensor_source = self.preprocess_image(img_source)
            flow_tensor = torch.from_numpy(flow).permute(2,0,1).unsqueeze(0).to(self.device).float()
            warped_original_tensor = flow_warp(tensor_source, flow_tensor)
            warped_original = warped_original_tensor[0].permute(1, 2, 0).cpu().numpy()
            warped_original = np.clip(warped_original, 0, 255).astype(np.uint8)
            
            if return_flow:
                return warped_original, flow, src_run, dst_run
            else:
                return warped_original, src_run, dst_run
