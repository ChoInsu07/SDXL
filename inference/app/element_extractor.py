import torch
import numpy as np
from PIL import Image, ImageDraw
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM_CHECKPOINT = os.path.join(BASE_DIR, "models", "sam", "sam_vit_h_4b8939.pth")
device = "mps" if torch.backends.mps.is_available() else "cpu"

_mask_generator = None

def _load_sam():
    global _mask_generator
    if _mask_generator is not None:
        return _mask_generator
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    _mask_generator = SamAutomaticMaskGenerator(sam)
    return _mask_generator


def extract_element(image: Image.Image, area_threshold_ratio: float = 0.15) -> Image.Image:
    """
    이미지에서 가장 중요한 요소(로고, 아이콘 등)를 추출.
    area_threshold_ratio: 전체 대비 요소 면적 비율 (0.15 = 15% 이하 요소)
    """
    mask_generator = _load_sam()
    img_np = np.array(image.convert("RGB"))

    masks = mask_generator.generate(img_np)

    if not masks:
        return image

    total_area = img_np.shape[0] * img_np.shape[1]
    masks = [m for m in masks if m["area"] / total_area < area_threshold_ratio]
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    if not masks:
        return image

    target = masks[0]
    mask = target["segmentation"]
    x, y, w, h = target["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)

    element = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    element_np = np.array(element)
    for c in range(3):
        element_np[:, :, c] = img_np[y : y + h, x : x + w, c] * mask[y : y + h, x : x + w]
    element_np[:, :, 3] = mask[y : y + h, x : x + w].astype(np.uint8) * 255

    return Image.fromarray(element_np)


def tile_pattern(element: Image.Image, target_size=(1024, 1536), spacing: int = 50) -> Image.Image:
    """
    추출된 요소를 반복 배열해서 패턴 시트 생성.
    spacing: 요소 간 간격 (px)
    """
    canvas = Image.new("RGBA", target_size, (255, 255, 255, 255))
    ew, eh = element.size
    step_x = ew + spacing
    step_y = eh + spacing

    for y in range(0, target_size[1], step_y):
        for x in range(0, target_size[0], step_x):
            canvas.paste(element, (x, y), element)

    return canvas.convert("RGB")
