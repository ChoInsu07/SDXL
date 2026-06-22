from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from app.pipeline import generate_kv_image
from PIL import Image
import io

app = FastAPI(title="KV-AI Key Visual Generator")

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    product_image: UploadFile = File(...),
    composition_image: UploadFile = File(None),
    negative_prompt: str = Form("low quality, blurry, distorted"),
    controlnet_scale: float = Form(0.5),
    ip_adapter_scale: float = Form(0.6),
    num_inference_steps: int = Form(30),
    use_pattern: bool = Form(False),
    pattern_threshold_ratio: float = Form(0.15),
    pattern_spacing: int = Form(50),
):
    product_bytes = await product_image.read()
    product_img = Image.open(io.BytesIO(product_bytes)).convert("RGB")

    if composition_image:
        comp_bytes = await composition_image.read()
        comp_img = Image.open(io.BytesIO(comp_bytes)).convert("RGB")
    else:
        comp_img = product_img

    result = generate_kv_image(
        prompt=prompt,
        product_image=product_img,
        composition_image=comp_img,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        controlnet_scale=controlnet_scale,
        ip_adapter_scale=ip_adapter_scale,
        use_pattern=use_pattern,
        pattern_threshold_ratio=pattern_threshold_ratio,
        pattern_spacing=pattern_spacing,
    )

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
