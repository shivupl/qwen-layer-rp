import os
import base64
from io import BytesIO
from pathlib import Path
import requests


import torch
import runpod
from PIL import Image
from diffusers import QwenImageLayeredPipeline

#new imports
import uuid
import boto3
from botocore.client import Config



# RunPod Cached Models live here (HF cache layout)
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

# Force offline so we only ever load from the cached snapshot
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


MODEL_ID = "Qwen/Qwen-Image-Layered"
LOCAL_FILES_ONLY = True

## new s3 stuff
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_URL_TTL_SECONDS = int(os.getenv("S3_URL_TTL_SECONDS", "3600"))
S3_PREFIX = os.getenv("S3_PREFIX", "layer-outputs")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL")


_S3 = None

def get_s3():
    global _S3
    if _S3 is not None:
        return _S3

    if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET]):
        raise RuntimeError(
            "Missing S3 env vars: S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, "
            "S3_SECRET_ACCESS_KEY, S3_BUCKET"
        )

    _S3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        region_name=os.getenv("S3_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )
    return _S3

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def upload_png_and_get_url(key: str, png_bytes: bytes) -> str:
    s3 = get_s3()
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=png_bytes,
        ContentType="image/png",
    )

    return f"{S3_PUBLIC_BASE_URL}/{key}"

## end of new S3 stuff



def candidate_model_cache_dirs(model_id: str):
    org, name = model_id.split("/", 1)
    yield CACHE_ROOT / f"models--{org}--{name}"
    yield CACHE_ROOT / f"models--{org.lower()}--{name}"
    yield CACHE_ROOT / f"models--{org}--{name.lower()}"
    yield CACHE_ROOT / f"models--{org.lower()}--{name.lower()}"


def resolve_snapshot_path(model_id: str) -> Path:
    mdir = None
    for cand in candidate_model_cache_dirs(model_id):
        if cand.exists():
            mdir = cand
            break

    if mdir is None:
        existing = []
        if CACHE_ROOT.exists():
            try:
                existing = sorted([p.name for p in CACHE_ROOT.iterdir()])[:40]
            except Exception:
                pass
        raise FileNotFoundError(
            f"Model cache dir not found for {model_id}. "
            f"Make sure Endpoint 'Model' is set to https://huggingface.co/{model_id}. "
            f"Hub entries (first 40): {existing}"
        )

    ref_main = mdir / "refs" / "main"
    if ref_main.exists():
        rev = ref_main.read_text().strip()
        snap = mdir / "snapshots" / rev
        if snap.exists():
            return snap

    snap_root = mdir / "snapshots"
    snaps = sorted(snap_root.glob("*")) if snap_root.exists() else []
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {snap_root}")
    return snaps[0]

_PIPE = None

def load_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Ensure the Serverless endpoint is using a GPU.")

    snapshot_path = resolve_snapshot_path(MODEL_ID)

    # Model card uses bf16 on CUDA. :contentReference[oaicite:2]{index=2}
    _PIPE = QwenImageLayeredPipeline.from_pretrained(
        str(snapshot_path),
        torch_dtype=torch.bfloat16,   #remove for fp8
        local_files_only=LOCAL_FILES_ONLY,
        device_map="cuda",
    )

    _PIPE.set_progress_bar_config(disable=True)
    return _PIPE

# def b64_to_pil_rgba(b64: str) -> Image.Image:
#     data = base64.b64decode(b64)
#     img = Image.open(BytesIO(data))
#     return img.convert("RGBA")


def url_to_pil_rgba(image_url: str) -> Image.Image:
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content))
    img.load()  # force full read (important for serverless)
    return img.convert("RGBA")



def handler(job):
    inp = job.get("input", {}) or {}

    image_url = inp.get("image_url")
    if not image_url:
        raise ValueError("Missing required input: image_url")

    # Match model card defaults/example. :contentReference[oaicite:3]{index=3}
    layers = int(inp.get("layers", 4))
    resolution = int(inp.get("resolution", 640))
    steps = int(inp.get("steps", 50))
    true_cfg_scale = float(inp.get("true_cfg_scale", 4.0))
    negative_prompt = inp.get("negative_prompt", " ")
    cfg_normalize = True
    use_en_prompt = True
    seed = int(inp.get("seed", 777))

    pipe = load_pipe()
    image = url_to_pil_rgba(image_url)

    gen = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        out = pipe(
            image=image,
            generator=gen,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            layers=layers,
            resolution=resolution,
            cfg_normalize=cfg_normalize,
            use_en_prompt=use_en_prompt,
        )

    layer_imgs = out.images[0]  # list of PIL images (RGBA layers) :contentReference[oaicite:4]{index=4}
    
    job_id = inp.get("job_id") or str(uuid.uuid4())

    layer_urls = []
    for i, im in enumerate(layer_imgs):
        key = f"{S3_PREFIX}/{job_id}/layer_{i}.png"
        url = upload_png_and_get_url(key, pil_to_png_bytes(im))
        layer_urls.append(url)

    return {
        "model_id": MODEL_ID,
        "job_id": job_id,
        "seed": seed,
        "layers": layers,
        "resolution": resolution,
        "steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "cfg_normalize": cfg_normalize,
        "use_en_prompt": use_en_prompt,
        "num_returned_layers": len(layer_urls),
        "layer_urls": layer_urls,
        "snapshot_path": str(resolve_snapshot_path(MODEL_ID)),
    }

runpod.serverless.start({"handler": handler})
