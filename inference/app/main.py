from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from app.pipeline import generate_kv_image
from PIL import Image
import io

app = FastAPI(title="KV-AI Key Visual Generator")

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    negative_prompt: str = Form("low quality, blurry, distorted"),
    controlnet_scale: float = Form(0.8),
    num_inference_steps: int = Form(30),
):
    image_bytes = await image.read()
    init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = generate_kv_image(
        prompt=prompt,
        init_image=init_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        controlnet_scale=controlnet_scale,
    )

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")
