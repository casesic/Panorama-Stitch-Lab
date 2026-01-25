import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import os

class PersonSegmenter:
    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"[PersonSegmenter] Initializing on {self.device}...")
        
        try:
            weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
        except AttributeError:
            print("[PersonSegmenter] Warning: New weights API not found, falling back to pretrained=True")
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        except Exception as e:
            print(f"[PersonSegmenter] Error loading model: {e}")
            raise e

        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_mask(self, img_bgr, threshold=0.5, dilate_pixels=0):
        """
        输入 BGR 图像，返回人像 Mask (numpy uint8, 0 or 255)
        dilate_pixels: 膨胀像素数，用于覆盖人手持物品（如手机）。
        """
        # 1. 预处理
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
        
        # 2. 推理
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # 3. 解析结果 (DeepLabV3 output shape: [21, H, W])
        # Class 15 is 'person'
        probs = output.softmax(dim=0)
        person_prob = probs[15] # [H, W]
        
        mask = (person_prob > threshold).byte().cpu().numpy() * 255
        
        # Resize mask back to original image size
        if mask.shape != img_bgr.shape[:2]:
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 4. 膨胀操作 (Expand mask to cover held objects like phones)
        if dilate_pixels > 0:
            kernel_size = int(dilate_pixels)
            if kernel_size % 2 == 0: kernel_size += 1
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        return mask

    def mask_out_person(self, img_bgr, fill_value=(0, 0, 0), dilate_pixels=0, threshold=0.5):
        """
        将图片中的人像区域涂抹为指定颜色
        """
        mask = self.get_mask(img_bgr, threshold=threshold, dilate_pixels=dilate_pixels)
        img_masked = img_bgr.copy()
        
        # Apply fill
        img_masked[mask == 255] = fill_value
             
        return img_masked, mask

# 全局单例
_segmenter = None

def get_person_mask(img_bgr, device=None, threshold=0.5, dilate_pixels=0):
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter(device)
    return _segmenter.get_mask(img_bgr, threshold=threshold, dilate_pixels=dilate_pixels)

def remove_person(img_bgr, device=None, fill_value=(0,0,0), dilate_pixels=0, threshold=0.5):
    """
    Helper function to remove person.
    Args:
        dilate_pixels: (int) Pixels to dilate (e.g. 30 for phones). 0 for none.
        threshold: (float) Confidence threshold. Lower to 0.2-0.3 for distorted limbs.
    """
    global _segmenter
    if _segmenter is None:
        _segmenter = PersonSegmenter(device)
    return _segmenter.mask_out_person(img_bgr, fill_value=fill_value, dilate_pixels=dilate_pixels, threshold=threshold)
