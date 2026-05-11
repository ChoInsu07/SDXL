import os
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    DDIMScheduler,
)
import cv2


device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SDXL_BASE_PATH = os.path.join(MODELS_DIR, "sdxl", "sd_xl_base")
SDXL_REFINER_PATH = os.path.join(MODELS_DIR, "sdxl", "sd_xl_refiner")
CONTROLNET_PATH = os.path.join(MODELS_DIR, "controlnet", "canny")
IP_ADAPTER_PATH = os.path.join(MODELS_DIR, "ip_adapter")

# ── ControlNet ──
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_PATH, torch_dtype=dtype
).to(device)

# ── SDXL Base + ControlNet ──
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    SDXL_BASE_PATH,
    controlnet=controlnet,
    torch_dtype=dtype,
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# ── Refiner ──
refiner = StableDiffusionXLPipeline.from_pretrained(
    SDXL_REFINER_PATH, torch_dtype=dtype
).to(device)

# ── IP-Adapter (선택) ──
ip_adapter_weight = os.path.join(IP_ADAPTER_PATH, "sdxl_models", "ip-adapter_sdxl.safetensors")
if os.path.exists(ip_adapter_weight):
    pipe.load_ip_adapter(ip_adapter_weight, subfolder="")
    pipe.set_ip_adapter_scale(0.6)

# ── LoRA (선택) ──
lora_dir = os.path.join(MODELS_DIR, "lora")
if os.path.isdir(lora_dir):
    lora_files = [f for f in os.listdir(lora_dir) if f.endswith((".safetensors", ".bin"))]
    if lora_files:
        pipe.load_lora_weights(os.path.join(lora_dir, lora_files[0]))


def to_canny(image: Image.Image, low=100, high=200) -> Image.Image:
    img = np.array(image.convert("L"))
    edges = cv2.Canny(img, low, high)
    return Image.fromarray(edges).convert("RGB")


@torch.inference_mode()
def generate_kv_image(
    prompt: str,
    init_image: Image.Image,
    negative_prompt: str = "low quality, blurry, distorted",
    num_inference_steps: int = 30,
    controlnet_scale: float = 0.8,
):
    control_image = to_canny(init_image)
    gen = torch.Generator(device=device).manual_seed(42)

    latent = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        controlnet_conditioning_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        generator=gen,
        output_type="latent",
    ).images[0]

    final = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=latent[None, ...],
        num_inference_steps=max(10, num_inference_steps // 2),
        generator=gen,
    ).images[0]

    return final
