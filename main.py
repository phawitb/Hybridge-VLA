import base64
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


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


# --- Helpers ---

def extract_text(data: dict) -> str:
    return "".join(
        p.get("text", "")
        for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    ) or "(no text response)"


def parse_json_response(raw_text: str) -> dict | None:
    try:
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
        raw_json = fence.group(1).strip() if fence else raw_text.strip()
        raw_json = raw_json.replace("{{", "{").replace("}}", "}")
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def normalize_bboxes(plan: dict, img_w: int, img_h: int) -> bool:
    """Auto-normalize pixel bboxes to [0,1]. Returns True if any were fixed."""
    fixed = False
    for step in plan.get("steps", []):
        box = step.get("target_bbox")
        if box and isinstance(box, list) and len(box) == 4:
            if any(v > 1.0 for v in box):
                fixed = True
                step["target_bbox"] = [
                    round(box[0] / img_w, 6),
                    round(box[1] / img_h, 6),
                    round(box[2] / img_w, 6),
                    round(box[3] / img_h, 6),
                ]
    return fixed


async def call_gemini(client: httpx.AsyncClient, url: str, b64: str, mime: str, text: str) -> tuple[dict, float]:
    """Call Gemini API with image + text. Returns (response_json, elapsed_seconds)."""
    parts = [
        {"inlineData": {"mimeType": mime, "data": b64}},
        {"text": text},
    ]
    t0 = time.time()
    resp = await client.post(url, json={"contents": [{"parts": parts}]})
    elapsed = round(time.time() - t0, 2)
    return resp.json(), elapsed, resp.status_code


async def verify_plan(
    client: httpx.AsyncClient,
    url: str,
    b64: str,
    mime: str,
    instruction: str,
    plan_json: str,
    available_methods: str,
    verify_template: str,
) -> dict | None:
    """Send plan + image to Gemini for verification. Returns parsed result or None."""
    verify_prompt = (
        verify_template
        .replace("{instruction}", instruction)
        .replace("{plan_json}", plan_json)
        .replace("{available_methods}", available_methods)
    )
    parts = [
        {"inlineData": {"mimeType": mime, "data": b64}},
        {"text": verify_prompt},
    ]
    resp = await client.post(url, json={"contents": [{"parts": parts}]})
    if resp.status_code != 200:
        return None
    raw = extract_text(resp.json())
    return parse_json_response(raw)


# --- Routes ---

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
        "verify": cfg.get("verify", {"enabled": True, "max_retries": 3}),
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
    api_key = os.environ["GEMINI_API_KEY"]
    verify_cfg = cfg.get("verify", {"enabled": True, "max_retries": 3})
    max_retries = verify_cfg.get("max_retries", 3)
    verify_enabled = verify_cfg.get("enabled", True)
    verify_template = cfg.get("verify_prompt_template", "")

    # Save image
    image_bytes = await image.read()
    image_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    ext = image.filename.rsplit(".", 1)[-1] if "." in image.filename else "jpg"
    image_filename = f"{image_id}.{ext}"
    (IMAGES_DIR / image_filename).write_bytes(image_bytes)

    # Image info
    img = Image.open(io.BytesIO(image_bytes))
    img_w, img_h = img.size
    b64 = base64.b64encode(image_bytes).decode()
    mime = image.content_type or "image/jpeg"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    methods_list = [m.strip() for m in methods.split(",") if m.strip()]
    methods_str = "\n".join(f"* {m}" for m in methods_list)

    # --- Infer + verify loop ---
    total_elapsed = 0
    all_attempts = []
    plan_data = None
    raw_text = ""
    bbox_auto_fixed = False
    verify_info = {
        "enabled": verify_enabled,
        "passed": False,
        "attempts": 0,
        "max_retries": max_retries,
        "history": [],
    }

    token_usage_total = {"input": 0, "output": 0, "total": 0}

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(1, max_retries + 1):
            attempt_record = {"attempt": attempt}

            # Step 1: Call Gemini for plan
            data, elapsed, status_code = await call_gemini(client, url, b64, mime, prompt)
            total_elapsed += elapsed
            attempt_record["infer_elapsed"] = elapsed

            if status_code != 200:
                err_msg = data.get("error", {}).get("message", str(data))
                attempt_record["error"] = err_msg
                all_attempts.append(attempt_record)
                verify_info["history"].append(attempt_record)
                verify_info["attempts"] = attempt
                # Don't retry on API error
                return {
                    "error": err_msg,
                    "elapsed": total_elapsed,
                    "verify": verify_info,
                }

            raw_text = extract_text(data)
            attempt_record["raw_response"] = raw_text

            # Accumulate token usage
            usage = data.get("usageMetadata", {})
            token_usage_total["input"] += usage.get("promptTokenCount", 0)
            token_usage_total["output"] += usage.get("candidatesTokenCount", 0)
            token_usage_total["total"] += usage.get("totalTokenCount", 0)

            # Step 2: Parse plan
            plan_data = parse_json_response(raw_text)
            if not plan_data or "steps" not in plan_data:
                attempt_record["verify_result"] = {"verified": False, "reason": "Failed to parse valid plan JSON"}
                verify_info["history"].append(attempt_record)
                verify_info["attempts"] = attempt
                continue

            # Step 3: Auto-normalize bbox
            was_fixed = normalize_bboxes(plan_data, img_w, img_h)
            if was_fixed:
                bbox_auto_fixed = True

            # Step 4: Verify (if enabled)
            if not verify_enabled:
                attempt_record["verify_result"] = {"verified": True, "reason": "Verification disabled"}
                verify_info["history"].append(attempt_record)
                verify_info["passed"] = True
                verify_info["attempts"] = attempt
                break

            plan_json_str = json.dumps(plan_data, indent=2)
            v_result = await verify_plan(
                client, url, b64, mime,
                instruction, plan_json_str, methods_str, verify_template,
            )

            if v_result and v_result.get("verified"):
                attempt_record["verify_result"] = v_result
                verify_info["history"].append(attempt_record)
                verify_info["passed"] = True
                verify_info["attempts"] = attempt
                break
            else:
                reason = v_result.get("reason", "Verification call failed") if v_result else "Verification call failed"
                attempt_record["verify_result"] = {"verified": False, "reason": reason}
                verify_info["history"].append(attempt_record)
                verify_info["attempts"] = attempt
                # Will retry on next iteration

    # Token limit info
    model_limit = 1048576
    for m in cfg["models"]:
        if m["id"] == model:
            model_limit = m["input_limit"]
            break
    token_usage_total["limit"] = model_limit
    token_usage_total["remaining"] = max(0, model_limit - token_usage_total["input"])

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
        "bbox_auto_normalized": bbox_auto_fixed,
        "verify": verify_info,
        "token_usage": token_usage_total,
        "elapsed": round(total_elapsed, 2),
    }
    (RESULTS_DIR / f"{image_id}.json").write_text(json.dumps(result, indent=2))

    return result


@app.get("/api/token-usage")
async def token_usage():
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return {"input": 0, "output": 0, "total": 0, "limit": 1048576, "remaining": 1048576}
    latest = json.loads(files[0].read_text())
    return latest.get("token_usage", {"input": 0, "output": 0, "total": 0, "limit": 1048576, "remaining": 1048576})


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
