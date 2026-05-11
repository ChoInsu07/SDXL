from huggingface_hub import snapshot_download
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")

models = [
    ("stabilityai/stable-diffusion-xl-base-1.0", os.path.join(MODELS_DIR, "sdxl", "sd_xl_base")),
    ("stabilityai/stable-diffusion-xl-refiner-1.0", os.path.join(MODELS_DIR, "sdxl", "sd_xl_refiner")),
    ("lllyasviel/sd-controlnet-canny", os.path.join(MODELS_DIR, "controlnet", "canny")),
    ("lllyasviel/sd-controlnet-depth", os.path.join(MODELS_DIR, "controlnet", "depth")),
    ("h94/IP-Adapter", os.path.join(MODELS_DIR, "ip_adapter")),
]

for repo_id, local_dir in models:
    print(f"\nDownloading {repo_id} ...")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  -> saved to {local_dir}")

print("\nAll models downloaded successfully.")
