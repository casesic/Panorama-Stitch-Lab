from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

try:
    from safetensors.torch import load_file
except ImportError as exc:
    raise ImportError("safetensors is required. Please install it with: pip install safetensors") from exc

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import (
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_sam3_model,
    _create_sam3_transformer,
    _create_segmentation_head,
    _create_text_encoder,
    _create_vl_backbone,
    _create_vision_backbone,
    build_tracker,
)


CHECKPOINT_PATH = r"models\SAM3\sam3.safetensors"
BPE_PATH = r"models\SAM3\assets\bpe_simple_vocab_16e6.txt.gz"
INPUT_DIR = r"data_process\Output\20260311_riding\image\equirect_rotate_crop\cam4"
OUTPUT_DIR = r"data_process\Output\20260311_riding\seg\sam3"
TEXT_PROMPT = "person"


def build_sam3_image_model_local(
    bpe_path: str,
    device: str,
    eval_mode: bool = True,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
):
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    text_encoder = _create_text_encoder(bpe_path)
    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None

    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_sam3_image_model_local(
        bpe_path=BPE_PATH,
        device="cpu",
        eval_mode=True,
    )
    state_dict = load_file(CHECKPOINT_PATH)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    processor = Sam3Processor(model, device=device)

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])
    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        image = Image.open(image_path).convert("RGB")
        img_tensor = TF.to_tensor(image).to(device)

        with torch.inference_mode():
            state = processor.set_image(img_tensor)
            state = processor.set_text_prompt(TEXT_PROMPT, state)

        masks = state["masks"]
        if masks is None or masks.shape[0] == 0:
            print(f"  No mask found for {image_path.name}")
            continue

        combined = masks.any(dim=0).cpu().numpy().astype(np.uint8) * 255
        out_path = output_dir / (image_path.stem + "_mask.png")
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved mask to {out_path}")


if __name__ == "__main__":
    main()
