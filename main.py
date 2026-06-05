import base64
import io
import json
import re
import time
import uuid
from pathlib import Path

import httpx
import yaml
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
RESULTS_DIR = ROOT / CONFIG["storage"]["results_dir"]
IMAGES_DIR = ROOT / CONFIG["storage"]["images_dir"]
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Hybridge VLA")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/data/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
async def get_config():
    cfg = load_config()
    return {
        "models": cfg["models"],
        "default_model": cfg["gemini"]["default_model"],
        "default_instruction": cfg.get("default_instruction", ""),
        "available_methods": cfg.get("available_methods", []),
        "prompt_template": cfg.get("prompt_template", ""),
    }


@app.post("/api/infer")
async def infer(
    model: str = Form(...),
    instruction: str = Form(...),
    methods: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
):
    cfg = load_config()
    api_key = cfg["gemini"]["api_key"]

    # Save image
    image_bytes = await image.read()
    image_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    ext = image.filename.rsplit(".", 1)[-1] if "." in image.filename else "jpg"
    image_filename = f"{image_id}.{ext}"
    (IMAGES_DIR / image_filename).write_bytes(image_bytes)

    # Build Gemini request
    b64 = base64.b64encode(image_bytes).decode()
    parts = [
        {"inlineData": {"mimeType": image.content_type or "image/jpeg", "data": b64}},
        {"text": prompt},
    ]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json={"contents": [{"parts": parts}]})
    elapsed = round(time.time() - t0, 2)

    data = resp.json()
    if resp.status_code != 200:
        return {"error": data.get("error", {}).get("message", str(data)), "elapsed": elapsed}

    raw_text = "".join(
        p.get("text", "")
        for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    ) or "(no text response)"

    usage = data.get("usageMetadata", {})
    token_usage = {
        "input": usage.get("promptTokenCount", 0),
        "output": usage.get("candidatesTokenCount", 0),
        "total": usage.get("totalTokenCount", 0),
    }
    model_limit = 1048576
    for m in cfg["models"]:
        if m["id"] == model:
            model_limit = m["input_limit"]
            break
    token_usage["limit"] = model_limit
    token_usage["remaining"] = max(0, model_limit - token_usage["input"])

    # Get image dimensions for bbox normalization
    img = Image.open(io.BytesIO(image_bytes))
    img_w, img_h = img.size

    # Try parse plan JSON
    plan_data = None
    bbox_normalized = True
    try:
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        raw_json = fence.group(1).strip() if fence else raw_text.strip()
        # Fix double-brace issue from template leaking into model output
        raw_json = raw_json.replace("{{", "{").replace("}}", "}")
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict):
            # Auto-normalize bbox if values are in pixel coordinates (any value > 1.0)
            if "steps" in parsed:
                for step in parsed["steps"]:
                    box = step.get("target_bbox")
                    if box and isinstance(box, list) and len(box) == 4:
                        if any(v > 1.0 for v in box):
                            bbox_normalized = False
                            step["target_bbox"] = [
                                round(box[0] / img_w, 6),  # x_center
                                round(box[1] / img_h, 6),  # y_center
                                round(box[2] / img_w, 6),  # width
                                round(box[3] / img_h, 6),  # height
                            ]
            plan_data = parsed
    except (json.JSONDecodeError, AttributeError):
        pass

    # Parse methods list
    methods_list = [m.strip() for m in methods.split(",") if m.strip()]

    result = {
        "id": image_id,
        "timestamp": time.time(),
        "model": model,
        "instruction": instruction,
        "methods": methods_list,
        "prompt": prompt,
        "image": image_filename,
        "image_size": [img_w, img_h],
        "raw_response": raw_text,
        "plan": plan_data,
        "bbox_auto_normalized": not bbox_normalized,
        "token_usage": token_usage,
        "elapsed": elapsed,
    }
    (RESULTS_DIR / f"{image_id}.json").write_text(json.dumps(result, indent=2))

    return result


@app.get("/api/results")
async def list_results(limit: int = 50, offset: int = 0):
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    total = len(files)
    items = []
    for f in files[offset : offset + limit]:
        items.append(json.loads(f.read_text()))
    return {"total": total, "items": items}


@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    path = RESULTS_DIR / f"{result_id}.json"
    if not path.exists():
        return {"error": "Not found"}
    return json.loads(path.read_text())


@app.delete("/api/results/{result_id}")
async def delete_result(result_id: str):
    result_path = RESULTS_DIR / f"{result_id}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        img_path = IMAGES_DIR / result.get("image", "")
        if img_path.exists():
            img_path.unlink()
        result_path.unlink()
        return {"ok": True}
    return {"error": "Not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
