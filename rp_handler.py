import os
import base64
from io import BytesIO
from pathlib import Path

import torch
import runpod
from PIL import Image
from diffusers import QwenImageLayeredPipeline

# RunPod Cached Models live here (HF cache layout)
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

# Force offline so we only ever load from the cached snapshot
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def str_to_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default

def normalize_model_id(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return "Qwen/Qwen-Image-Layered"
    if "huggingface.co/" in s:
        s = s.split("huggingface.co/", 1)[1]
    s = s.split(":", 1)[0]
    org, name = s.split("/", 1)
    return f"{org}/{name}"

MODEL_ID = normalize_model_id(os.getenv("MODEL_ID", "Qwen/Qwen-Image-Layered"))
LOCAL_FILES_ONLY = str_to_bool(os.getenv("LOCAL_FILES_ONLY", "true"), default=True)

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
        torch_dtype=torch.bfloat16,
        local_files_only=LOCAL_FILES_ONLY,
        device_map="cuda",
    )

    _PIPE.set_progress_bar_config(disable=True)
    return _PIPE

def pil_to_b64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_pil_rgba(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    img = Image.open(BytesIO(data))
    return img.convert("RGBA")

def handler(job):
    inp = job.get("input", {}) or {}

    image_b64 = inp.get("image_base64")
    if not image_b64:
        raise ValueError("Missing required input: image_base64 (base64-encoded PNG/JPG).")

    # Match model card defaults/example. :contentReference[oaicite:3]{index=3}
    layers = int(inp.get("layers", 4))
    resolution = int(inp.get("resolution", 640))
    steps = int(inp.get("steps", 50))
    true_cfg_scale = float(inp.get("true_cfg_scale", 4.0))
    negative_prompt = inp.get("negative_prompt", " ")
    cfg_normalize = str_to_bool(inp.get("cfg_normalize", True), default=True)
    use_en_prompt = str_to_bool(inp.get("use_en_prompt", True), default=True)

    seed = int(inp.get("seed", 777))

    pipe = load_pipe()
    image = b64_to_pil_rgba(image_b64)

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
    layer_b64 = [pil_to_b64_png(im) for im in layer_imgs]

    return {
        "model_id": MODEL_ID,
        "seed": seed,
        "layers": layers,
        "resolution": resolution,
        "steps": steps,
        "true_cfg_scale": true_cfg_scale,
        "cfg_normalize": cfg_normalize,
        "use_en_prompt": use_en_prompt,
        "num_returned_layers": len(layer_b64),
        "layer_base64_png": layer_b64,
        "snapshot_path": str(resolve_snapshot_path(MODEL_ID)),
    }

runpod.serverless.start({"handler": handler})
