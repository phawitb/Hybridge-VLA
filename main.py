import asyncio
import base64
import io
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import yaml
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
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
app.mount("/data", StaticFiles(directory=str(ROOT / "data")), name="dataset-files")


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
    robot_cfg = cfg.get("robot", {})
    cams = robot_cfg.get("cameras", {})
    return {
        "models": cfg["models"],
        "default_model": cfg["gemini"]["default_model"],
        "default_instruction": cfg.get("default_instruction", ""),
        "available_methods": cfg.get("available_methods", []),
        "prompt_template": cfg.get("prompt_template", ""),
        "verify": cfg.get("verify", {"enabled": True, "max_retries": 3}),
        "robot_port": robot_cfg.get("port", ""),
        "robot_id": robot_cfg.get("id", ""),
        "camera_top_index": cams.get("top", {}).get("index", 0),
        "camera_wrist_index": cams.get("wrist", {}).get("index", 1),
        "teleop": cfg.get("teleop", {}),
        "click_to_move": cfg.get("click_to_move", {"target_height": 0, "safety_height": 10}),
        "data_collection": cfg.get("data_collection", {
            "repo_id": "phawitbinabik/so101-pick-place", "dataset_root": "./data/so101-pick-place",
            "task": "pick up the pink bow to the green bowl", "num_episodes": 50,
            "encoder_threads": 4, "push_to_hub": False, "resume": False,
            "cameras": {
                "top": {"fps": 30, "width": 320, "height": 240},
                "wrist": {"fps": 30, "width": 320, "height": 240},
            },
        }),
    }


@app.post("/api/config/save")
async def save_config(request: Request):
    """Save configuration fields to config.yaml."""
    data = await request.json()
    cfg = load_config()

    if "robot_port" in data:
        cfg.setdefault("robot", {})["port"] = data["robot_port"]
    if "robot_id" in data:
        cfg.setdefault("robot", {})["id"] = data["robot_id"]
    if "camera_top_index" in data:
        cfg.setdefault("robot", {}).setdefault("cameras", {}).setdefault("top", {})["index"] = int(data["camera_top_index"])
    if "camera_wrist_index" in data:
        cfg.setdefault("robot", {}).setdefault("cameras", {}).setdefault("top", {})  # ensure structure
        cfg["robot"]["cameras"].setdefault("wrist", {})["index"] = int(data["camera_wrist_index"])
    if "default_model" in data:
        cfg.setdefault("gemini", {})["default_model"] = data["default_model"]
    if "available_methods" in data:
        cfg["available_methods"] = data["available_methods"]
    if "prompt_template" in data:
        cfg["prompt_template"] = data["prompt_template"]
    if "default_instruction" in data:
        cfg["default_instruction"] = data["default_instruction"]
    if "teleop" in data:
        tel = data["teleop"]
        cfg.setdefault("teleop", {})
        if "type" in tel:
            cfg["teleop"]["type"] = tel["type"]
        if "port" in tel:
            cfg["teleop"]["port"] = tel["port"]
        if "id" in tel:
            cfg["teleop"]["id"] = tel["id"]
    if "click_to_move" in data:
        cfg["click_to_move"] = {
            "target_height": float(data["click_to_move"].get("target_height", 0)),
            "safety_height": float(data["click_to_move"].get("safety_height", 0)),
        }
    if "data_collection" in data:
        dc = data["data_collection"]
        cams_dc = dc.get("cameras", {})
        cfg["data_collection"] = {
            "repo_id": dc.get("repo_id", "phawitbinabik/so101-pick-place"),
            "dataset_root": dc.get("dataset_root", "./data/so101-pick-place"),
            "task": dc.get("task", ""),
            "num_episodes": int(dc.get("num_episodes", 50)),
            "encoder_threads": int(dc.get("encoder_threads", 4)),
            "push_to_hub": bool(dc.get("push_to_hub", False)),
            "resume": bool(dc.get("resume", False)),
            "cameras": {
                "top": {
                    "fps": int(cams_dc.get("top", {}).get("fps", 30)),
                    "width": int(cams_dc.get("top", {}).get("width", 320)),
                    "height": int(cams_dc.get("top", {}).get("height", 240)),
                },
                "wrist": {
                    "fps": int(cams_dc.get("wrist", {}).get("fps", 30)),
                    "width": int(cams_dc.get("wrist", {}).get("width", 320)),
                    "height": int(cams_dc.get("wrist", {}).get("height", 240)),
                },
            },
        }

    with open(ROOT / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return {"ok": True}


@app.get("/api/cameras/scan")
async def scan_cameras():
    """Try to capture a frame from each camera index, vstack them with index label."""
    import cv2
    frames = []
    idx = 0
    max_idx = 10  # reasonable upper bound
    while idx < max_idx:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            idx += 1
            continue
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            idx += 1
            continue
        # Draw index label at top-right
        label = f"idx: {idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 1.2, 3
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        x = frame.shape[1] - tw - 15
        y = th + 15
        cv2.rectangle(frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thickness)
        frames.append(frame)
        idx += 1

    if not frames:
        return {"ok": False, "error": "No cameras found"}

    # Resize all frames to same height before hstack
    target_h = max(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        if f.shape[0] != target_h:
            ratio = target_h / f.shape[0]
            f = cv2.resize(f, (int(f.shape[1] * ratio), target_h))
        resized.append(f)

    stacked = np.hstack(resized)
    _, jpg = cv2.imencode(".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
    return {"ok": True, "count": len(frames), "image": b64}


@app.post("/api/test/all")
async def test_all():
    """Run all system tests: robot, cameras, Gemini API."""
    results = []

    # 1. Test Robot connection
    try:
        if robot_state["connected"]:
            pos = robot_get_positions()
            results.append({"name": "Robot Connection", "ok": True, "detail": f"Connected. Joints: {pos}"})
        else:
            results.append({"name": "Robot Connection", "ok": False, "detail": "Robot not connected. Press Start in Debug tab to connect."})
    except Exception as e:
        results.append({"name": "Robot Connection", "ok": False, "detail": str(e)})

    # 2. Test Top Camera
    import cv2
    try:
        st = cam_state.get("top")
        if st and st.get("cap") and st["cap"].isOpened():
            with st["lock"]:
                f = st["frame"]
            if f is not None:
                _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
                results.append({"name": "Top Camera", "ok": True, "detail": f"Frame: {f.shape[1]}x{f.shape[0]}", "frame": b64})
            else:
                results.append({"name": "Top Camera", "ok": False, "detail": "Camera opened but no frame captured yet."})
        else:
            results.append({"name": "Top Camera", "ok": False, "detail": "Camera not available. Connect robot first."})
    except Exception as e:
        results.append({"name": "Top Camera", "ok": False, "detail": str(e)})

    # 3. Test Wrist Camera
    try:
        st = cam_state.get("wrist")
        if st and st.get("cap") and st["cap"].isOpened():
            with st["lock"]:
                f = st["frame"]
            if f is not None:
                _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
                results.append({"name": "Wrist Camera", "ok": True, "detail": f"Frame: {f.shape[1]}x{f.shape[0]}", "frame": b64})
            else:
                results.append({"name": "Wrist Camera", "ok": False, "detail": "Camera opened but no frame captured yet."})
        else:
            results.append({"name": "Wrist Camera", "ok": False, "detail": "Camera not available. Connect robot first."})
    except Exception as e:
        results.append({"name": "Wrist Camera", "ok": False, "detail": str(e)})

    # 4. Test Gemini API
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            results.append({"name": "Gemini API", "ok": False, "detail": "GEMINI_API_KEY not set in .env file."})
        else:
            cfg = load_config()
            model = cfg["gemini"]["default_model"]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, json={"contents": [{"parts": [{"text": "Say hello in one word."}]}]})
                if resp.status_code == 200:
                    text = extract_text(resp.json())
                    results.append({"name": "Gemini API", "ok": True, "detail": f"Model: {model}. Response: {text[:80]}"})
                else:
                    err = resp.json().get("error", {}).get("message", resp.text[:200])
                    results.append({"name": "Gemini API", "ok": False, "detail": f"HTTP {resp.status_code}: {err}"})
    except Exception as e:
        results.append({"name": "Gemini API", "ok": False, "detail": str(e)})

    # 5. Test Calibration
    try:
        if calib_state.get("homography"):
            n = len(calib_state.get("points", []))
            results.append({"name": "Calibration", "ok": True, "detail": f"Calibrated with {n} points."})
        else:
            results.append({"name": "Calibration", "ok": False, "detail": "Not calibrated. Go to Calibrate tab."})
    except Exception as e:
        results.append({"name": "Calibration", "ok": False, "detail": str(e)})

    return {"results": results}


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

    retry_feedback = ""  # accumulated feedback from failed verifications

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(1, max_retries + 1):
            attempt_record = {"attempt": attempt}

            # Step 1: Call Gemini for plan (include feedback from previous failed attempt)
            current_prompt = prompt
            if retry_feedback:
                current_prompt = prompt + retry_feedback
            data, elapsed, status_code = await call_gemini(client, url, b64, mime, current_prompt)
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
                retry_feedback = (
                    f"\n\n## Previous Attempt Failed\n"
                    f"Your previous response could not be parsed as valid JSON.\n\n"
                    f"### Your previous response:\n{raw_text[:500]}\n\n"
                    f"Please output ONLY valid JSON. No explanations, no markdown fences."
                )
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
                # Build feedback for next attempt
                retry_feedback = (
                    f"\n\n## Previous Attempt Failed\n"
                    f"Your previous plan was rejected by the verifier.\n\n"
                    f"### Your previous plan:\n{plan_json_str}\n\n"
                    f"### Verifier feedback:\n{reason}\n\n"
                    f"Please fix the issues and generate a corrected plan."
                )

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


# ══════════ Robot + Camera (Camera Calibrate) ══════════

ROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

robot_state = {"robot": None, "connected": False}
cam_state = {}


def connect_robot():
    """Connect robot + start cameras. Raises on failure."""
    import cv2
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    port = robot_cfg.get("port", "/dev/tty.usbmodem5B141122411")
    rcfg = SOFollowerRobotConfig(port=port, id=robot_cfg.get("id", "my_awesome_follower_arm"), cameras={})
    rob = make_robot_from_config(rcfg)
    rob.connect()
    robot_state["robot"] = rob
    robot_state["connected"] = True
    print(f"[Camera Calibrate] Robot connected on {port}")

    # Start cameras
    cams = robot_cfg.get("cameras", {})
    for name, cam_cfg in cams.items():
        cap = cv2.VideoCapture(cam_cfg.get("index", 0))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("w", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("h", 480))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        state = {"cap": cap, "frame": None, "lock": threading.Lock(), "running": True, "stopped": threading.Event()}
        cam_state[name] = state
        ok = cap.isOpened()
        print(f"[Camera Calibrate] Camera '{name}' (idx {cam_cfg.get('index', 0)}): {'OK' if ok else 'FAILED'}")

        def loop(st=state, nm=name, cam_cfg=cam_cfg):
            try:
                while st["running"]:
                    try:
                        c = st["cap"]
                        if c is None or not c.isOpened():
                            c = cv2.VideoCapture(cam_cfg.get("index", 0))
                            c.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("w", 640))
                            c.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("h", 480))
                            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                            st["cap"] = c
                            time.sleep(1)
                            continue
                        # Flush stale buffer frames to get the latest one
                        c.grab()
                        ret, frame = c.retrieve()
                        if ret:
                            with st["lock"]:
                                st["frame"] = frame.copy()
                        else:
                            c.release()
                            st["cap"] = None
                            time.sleep(1)
                    except Exception as e:
                        print(f"[Camera] '{nm}' error: {e}")
                        time.sleep(1)
            finally:
                if st["cap"]:
                    st["cap"].release()
                    st["cap"] = None
                st["stopped"].set()
                print(f"[Camera] '{nm}' thread stopped")

        threading.Thread(target=loop, daemon=True).start()


def disconnect_robot():
    """Disconnect robot + release cameras. Waits for camera threads to finish."""
    # Signal all camera threads to stop
    for name, st in list(cam_state.items()):
        st["running"] = False
    # Wait for threads to release cameras (max 3s each)
    for name, st in list(cam_state.items()):
        st["stopped"].wait(timeout=3)
    cam_state.clear()

    # Disconnect robot
    rob = robot_state["robot"]
    if rob:
        try:
            rob.disconnect()
        except Exception:
            pass
    robot_state["robot"] = None
    robot_state["connected"] = False
    print("[Camera Calibrate] Robot disconnected")


def robot_get_positions():
    rob = robot_state["robot"]
    obs = rob.get_observation()
    return {j: round(float(obs[f"{j}.pos"]), 2) for j in ROBOT_JOINTS}


def robot_send_positions(positions):
    rob = robot_state["robot"]
    obs = rob.get_observation()
    action = {f"{j}.pos": float(obs[f"{j}.pos"]) for j in ROBOT_JOINTS}
    for j, v in positions.items():
        action[f"{j}.pos"] = float(v)
    rob.send_action(action)


@app.websocket("/ws/robot")
async def ws_robot(websocket: WebSocket):
    await websocket.accept()

    if not robot_state["connected"]:
        await websocket.send_json({"type": "error", "message": "Robot not connected"})
        await websocket.close()
        return

    # Send initial positions
    try:
        await websocket.send_json({"type": "init", "positions": robot_get_positions()})
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()
        return

    # Frame + position sender task
    async def send_frames():
        import cv2
        tick = 0
        while True:
            try:
                frames = {}
                for name, st in cam_state.items():
                    with st["lock"]:
                        f = st["frame"]
                    if f is not None:
                        _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        frames[name] = base64.b64encode(jpg.tobytes()).decode("ascii")
                if frames:
                    await websocket.send_json({"type": "frames", "data": frames})
                # Send positions every ~200ms (every 2nd tick)
                tick += 1
                if tick % 2 == 0:
                    try:
                        await websocket.send_json({"type": "positions", "data": robot_get_positions()})
                    except Exception:
                        pass
                await asyncio.sleep(0.1)
            except Exception:
                break

    sender = asyncio.create_task(send_frames())
    try:
        while True:
            data = await websocket.receive_json()
            t = data.get("type")
            if t == "move":
                robot_send_positions(data["data"])
            elif t == "read":
                await websocket.send_json({"type": "positions", "data": robot_get_positions()})
            elif t == "torque":
                rob = robot_state["robot"]
                if data["enabled"]:
                    rob.bus.enable_torque()
                else:
                    rob.bus.disable_torque()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS Robot] error: {e}")
    finally:
        sender.cancel()


# ══════════ Teleop ══════════

teleop_state = {"process": None, "running": False, "log_lines": [], "log_lock": threading.Lock()}


@app.get("/api/robot/status")
async def robot_status():
    # Check if teleop process died
    if teleop_state["running"] and teleop_state["process"]:
        if teleop_state["process"].poll() is not None:
            teleop_state["running"] = False
            teleop_state["process"] = None
    return {"connected": robot_state["connected"], "teleop": teleop_state["running"]}


@app.post("/api/robot/start")
async def robot_start():
    if robot_state["connected"]:
        return {"ok": True, "message": "Already connected"}
    try:
        connect_robot()
        return {"ok": True, "message": "Robot connected"}
    except ImportError:
        return {"ok": False, "error": "lerobot or cv2 not installed"}
    except Exception as e:
        robot_state["robot"] = None
        robot_state["connected"] = False
        return {"ok": False, "error": str(e)}


@app.post("/api/robot/stop")
async def robot_stop():
    if not robot_state["connected"]:
        return {"ok": True, "message": "Already disconnected"}
    disconnect_robot()
    return {"ok": True, "message": "Robot disconnected"}


def _teleop_log(text):
    """Append a line to teleop log buffer."""
    with teleop_state["log_lock"]:
        teleop_state["log_lines"].append(text)
        if len(teleop_state["log_lines"]) > 500:
            teleop_state["log_lines"] = teleop_state["log_lines"][-500:]
    print(f"[Teleop] {text}")


def _drain_teleop_output(proc):
    """Drain subprocess output and auto-answer calibration prompts.

    Uses character-by-character reading because calibration prompts
    (Python input()) don't end with newline, so readline() would block.
    """
    buf = ""
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            c = ch.decode("utf-8", errors="replace")
            if c == "\n":
                if buf.strip():
                    _teleop_log(buf)
                buf = ""
            else:
                buf += c
                # Check for calibration prompt (ends with ": " not newline)
                if "press enter" in buf.lower() and buf.rstrip().endswith(":"):
                    _teleop_log(buf)
                    try:
                        proc.stdin.write(b"\n")
                        proc.stdin.flush()
                        _teleop_log("Auto-answered calibration prompt")
                    except Exception:
                        pass
                    buf = ""
    except Exception:
        pass
    if buf.strip():
        _teleop_log(buf)


@app.post("/api/teleop/start")
async def teleop_start():
    if teleop_state["running"]:
        return {"ok": True, "message": "Already running"}
    # Stop slider-mode robot first (they share the follower port)
    if robot_state["connected"]:
        try:
            disconnect_robot()
        except Exception as e:
            return {"ok": False, "error": f"Failed to disconnect robot: {e}"}

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})

    # Validate ports exist
    robot_port = robot_cfg.get("port", "")
    leader_port = teleop_cfg.get("port", "")
    if not os.path.exists(robot_port):
        return {"ok": False, "error": f"Follower port not found: {robot_port}"}
    if not os.path.exists(leader_port):
        return {"ok": False, "error": f"Leader port not found: {leader_port}"}

    cmd = [
        "/opt/miniconda3/envs/lerobot/bin/lerobot-teleoperate",
        f"--robot.type=so101_follower",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--teleop.type={teleop_cfg.get('type', 'so101_leader')}",
        f"--teleop.port={leader_port}",
        f"--teleop.id={teleop_cfg.get('id', 'my_awesome_leader_arm')}",
    ]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        teleop_state["process"] = p
        teleop_state["running"] = True
        teleop_state["log_lines"] = []
        # Drain pipe in background to prevent buffer deadlock
        threading.Thread(target=_drain_teleop_output, args=(p,), daemon=True).start()
        print(f"[Teleop] Started: {' '.join(cmd)}")
        return {"ok": True, "message": "Teleop started"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/teleop/stop")
async def teleop_stop():
    if not teleop_state["running"]:
        return {"ok": True, "message": "Already stopped"}
    p = teleop_state["process"]
    if p:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("[Teleop] Process still alive after SIGKILL")
    teleop_state["process"] = None
    teleop_state["running"] = False
    print("[Teleop] Stopped")
    return {"ok": True, "message": "Teleop stopped"}


@app.get("/api/teleop/status")
async def teleop_status():
    """Get teleop running state and recent logs."""
    p = teleop_state["process"]
    if p and p.poll() is not None:
        teleop_state["running"] = False
        teleop_state["process"] = None
    with teleop_state["log_lock"]:
        recent_logs = list(teleop_state["log_lines"][-50:])
    return {"running": teleop_state["running"], "logs": recent_logs}


# ══════════ Visualize Data ══════════


@app.get("/api/datasets")
async def list_datasets():
    """List available LeRobot datasets in data/ directory."""
    data_dir = ROOT / "data"
    datasets = []
    if data_dir.exists():
        for d in sorted(data_dir.iterdir()):
            info_path = d / "meta" / "info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text())
                    datasets.append({
                        "name": d.name,
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type", ""),
                    })
                except Exception:
                    pass
    return {"ok": True, "datasets": datasets}


@app.get("/api/datasets/{name}/info")
async def dataset_info(name: str):
    """Get full info.json + tasks for a dataset."""
    info_path = ROOT / "data" / name / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}
    info = json.loads(info_path.read_text())
    # Read tasks
    tasks_path = ROOT / "data" / name / "meta" / "tasks.parquet"
    task_list = []
    if tasks_path.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(tasks_path))
            task_list = tbl.to_pydict().get("task", [])
        except Exception:
            pass
    return {"ok": True, "info": info, "tasks": task_list}


@app.get("/api/datasets/{name}/episodes")
async def dataset_episodes(name: str):
    """Get per-episode metadata from episodes parquet."""
    ep_dir = ROOT / "data" / name / "meta" / "episodes"
    episodes = []
    if ep_dir.exists():
        try:
            import pyarrow.parquet as pq
            for pf in sorted(ep_dir.rglob("*.parquet")):
                tbl = pq.read_table(str(pf))
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    episodes.append({k: v[i] for k, v in d.items()})
        except Exception:
            pass
    return {"ok": True, "episodes": episodes}


@app.get("/api/datasets/{name}/frames/{episode_index}")
async def dataset_frames(name: str, episode_index: int):
    """Get frame data (action, state) for a specific episode."""
    data_dir = ROOT / "data" / name / "data"
    if not data_dir.exists():
        return {"ok": False, "error": "Data not found"}
    try:
        import pyarrow.parquet as pq
        rows = []
        for pf in sorted(data_dir.rglob("*.parquet")):
            tbl = pq.read_table(str(pf))
            d = tbl.to_pydict()
            for i in range(len(d.get("episode_index", []))):
                if d["episode_index"][i] == episode_index:
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        # Convert numpy/list to plain python
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    rows.append(row)
        return {"ok": True, "frames": rows}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ══════════ Data Collection ══════════

datacollect_state = {
    "process": None,
    "running": False,
    "started_at": None,
    "log_lines": [],
    "log_lock": threading.Lock(),
}


def _datacollect_reader(proc, state):
    """Background thread to read subprocess stdout, store log lines, and auto-answer prompts."""
    buf = ""
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            c = ch.decode("utf-8", errors="replace")
            if c == "\n":
                if buf.strip():
                    with state["log_lock"]:
                        state["log_lines"].append(buf)
                        if len(state["log_lines"]) > 500:
                            state["log_lines"] = state["log_lines"][-500:]
                    print(f"[DataCollect] {buf}")
                buf = ""
            else:
                buf += c
                # Auto-answer calibration prompts
                if "press enter" in buf.lower() and buf.rstrip().endswith(":"):
                    with state["log_lock"]:
                        state["log_lines"].append(buf)
                    print(f"[DataCollect] {buf}")
                    try:
                        proc.stdin.write(b"\n")
                        proc.stdin.flush()
                        print("[DataCollect] Auto-answered calibration prompt")
                    except Exception:
                        pass
                    buf = ""
    except Exception:
        pass
    if buf.strip():
        with state["log_lock"]:
            state["log_lines"].append(buf)
        print(f"[DataCollect] {buf}")


@app.post("/api/datacollect/verify")
async def datacollect_verify():
    """Run pre-flight hardware checks."""
    import shutil

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    checks = []

    # 1. Robot port
    robot_port = robot_cfg.get("port", "")
    robot_ok = os.path.exists(robot_port)
    checks.append({"name": "Robot port", "detail": robot_port, "ok": robot_ok,
                    "error": "" if robot_ok else f"Port not found: {robot_port}"})

    # 2. Teleop port
    teleop_port = teleop_cfg.get("port", "")
    teleop_ok = os.path.exists(teleop_port)
    checks.append({"name": "Teleop port", "detail": teleop_port, "ok": teleop_ok,
                    "error": "" if teleop_ok else f"Port not found: {teleop_port}"})

    # 3. Cameras
    import cv2
    cams = robot_cfg.get("cameras", {})
    for cam_name, cam_cfg in cams.items():
        idx = cam_cfg.get("index", 0)
        cap = cv2.VideoCapture(idx)
        cam_ok = False
        if cap.isOpened():
            ret, _ = cap.read()
            cam_ok = ret
        cap.release()
        checks.append({"name": f"Camera: {cam_name}", "detail": f"index {idx}", "ok": cam_ok,
                        "error": "" if cam_ok else f"Cannot read from camera index {idx}"})

    # 4. Disk space
    stat = shutil.disk_usage(str(ROOT))
    free_gb = stat.free / (1024 ** 3)
    disk_ok = free_gb > 1.0
    checks.append({"name": "Disk space", "detail": f"{free_gb:.1f} GB free", "ok": disk_ok,
                    "error": "" if disk_ok else "Less than 1 GB free"})

    # 5. lerobot-record binary
    lerobot_bin = "/opt/miniconda3/envs/lerobot/bin/lerobot-record"
    bin_ok = os.path.exists(lerobot_bin)
    checks.append({"name": "lerobot-record", "detail": lerobot_bin, "ok": bin_ok,
                    "error": "" if bin_ok else f"Binary not found: {lerobot_bin}"})

    all_ok = all(c["ok"] for c in checks)
    return {"ok": all_ok, "checks": checks}


@app.post("/api/datacollect/start")
async def datacollect_start():
    """Start lerobot-record subprocess."""
    if datacollect_state["running"]:
        return {"ok": False, "error": "Already recording"}
    # Stop teleop and slider-mode robot if running (they share ports & cameras)
    if teleop_state["running"]:
        await teleop_stop()
    if robot_state["connected"]:
        disconnect_robot()
    # Wait for cameras to fully release
    time.sleep(1)

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    dc_cfg = cfg.get("data_collection", {})
    cams = robot_cfg.get("cameras", {})

    dc_cams = dc_cfg.get("cameras", {})

    # Build cameras config string for lerobot (use per-camera resolution)
    cam_parts = []
    for cam_name, cam_c in cams.items():
        dc_cam = dc_cams.get(cam_name, {})
        cam_fps = dc_cam.get("fps", 30)
        cam_w = dc_cam.get("width", 320)
        cam_h = dc_cam.get("height", 240)
        cam_parts.append(
            f"{cam_name}: {{type: opencv, index_or_path: {cam_c.get('index', 0)}, "
            f"width: {cam_w}, height: {cam_h}, fps: {cam_fps}}}"
        )
    cameras_str = "{" + ", ".join(cam_parts) + "}"
    # Use top camera fps for dataset fps (primary)
    fps = dc_cams.get("top", {}).get("fps", 30)

    cmd = [
        "/opt/miniconda3/envs/lerobot/bin/lerobot-record",
        f"--robot.type=so101_follower",
        f"--robot.port={robot_cfg.get('port', '')}",
        f"--robot.id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--robot.cameras={cameras_str}",
        f"--teleop.type={teleop_cfg.get('type', 'so101_leader')}",
        f"--teleop.port={teleop_cfg.get('port', '')}",
        f"--teleop.id={teleop_cfg.get('id', 'my_awesome_leader_arm')}",
        f"--display_data=false",
        f"--dataset.repo_id={dc_cfg.get('repo_id', 'user/my-dataset')}",
        f"--dataset.root={dc_cfg.get('dataset_root', './data/dataset')}",
        f"--dataset.push_to_hub={'true' if dc_cfg.get('push_to_hub', False) else 'false'}",
        f"--dataset.num_episodes={dc_cfg.get('num_episodes', 50)}",
        f"--dataset.single_task={dc_cfg.get('task', 'default task')}",
        f"--dataset.streaming_encoding=true",
        f"--dataset.fps={fps}",
        f"--dataset.encoder_threads={dc_cfg.get('encoder_threads', 4)}",
    ]
    if dc_cfg.get("resume", False):
        cmd.append("--resume=true")
    else:
        # Remove existing dataset directory to avoid FileExistsError
        import shutil
        dataset_root = Path(dc_cfg.get("dataset_root", "./data/dataset"))
        if not dataset_root.is_absolute():
            dataset_root = ROOT / dataset_root
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
            print(f"[DataCollect] Removed existing dataset dir: {dataset_root}")

    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        datacollect_state["process"] = p
        datacollect_state["running"] = True
        datacollect_state["started_at"] = time.time()
        datacollect_state["log_lines"] = []
        # Start log reader thread
        threading.Thread(target=_datacollect_reader, args=(p, datacollect_state), daemon=True).start()
        print(f"[DataCollect] Started: {' '.join(cmd)}")
        return {"ok": True, "message": "Recording started"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datacollect/stop")
async def datacollect_stop():
    """Stop lerobot-record subprocess."""
    if not datacollect_state["running"]:
        return {"ok": True, "message": "Not recording"}
    p = datacollect_state["process"]
    if p:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    datacollect_state["process"] = None
    datacollect_state["running"] = False
    datacollect_state["started_at"] = None
    print("[DataCollect] Stopped")
    return {"ok": True, "message": "Recording stopped"}


@app.get("/api/datacollect/status")
async def datacollect_status():
    """Get current recording status."""
    elapsed = 0
    if datacollect_state["started_at"] and datacollect_state["running"]:
        elapsed = time.time() - datacollect_state["started_at"]
    # Check if process exited
    p = datacollect_state["process"]
    if p and p.poll() is not None:
        datacollect_state["running"] = False
        datacollect_state["process"] = None
        datacollect_state["started_at"] = None
    with datacollect_state["log_lock"]:
        recent_logs = list(datacollect_state["log_lines"][-50:])
    return {
        "running": datacollect_state["running"],
        "elapsed": round(elapsed, 1),
        "logs": recent_logs,
    }


# ══════════ Interpolation IK from calibration points ══════════


def interpolate_joints_from_pixel(pixel_uv, points, height_cm=0.0):
    """Interpolate joint angles from 4 calibration points using bilinear interpolation.

    For each joint, fit: value = a + b*u + c*v + d*u*v
    using the 4 calibration point pairs (pixel → joint_value).

    height_cm: gripper height above calibrated surface (cm).
               Adjusts shoulder_lift to raise the arm.
    """
    if len(points) < 4:
        return None

    u, v = pixel_uv
    pixels = np.array([p["pixel"] for p in points[:4]], dtype=np.float64)

    # Build coefficient matrix: [1, u, v, u*v] for each calibration point
    A = np.column_stack([
        np.ones(4),
        pixels[:, 0],
        pixels[:, 1],
        pixels[:, 0] * pixels[:, 1],
    ])

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    result = {}

    for jname in joint_names:
        values = np.array([p["joints"][jname] for p in points[:4]], dtype=np.float64)
        try:
            coeffs = np.linalg.solve(A, values)
            result[jname] = round(float(coeffs[0] + coeffs[1] * u + coeffs[2] * v + coeffs[3] * u * v), 2)
        except np.linalg.LinAlgError:
            return None

    # Adjust height: raise arm by offsetting shoulder_lift
    # SO101 effective arm length ≈ 0.25m (upper_arm + lower_arm)
    # Δshoulder_lift ≈ arctan(height / arm_reach) in degrees
    if height_cm > 0:
        h = height_cm / 100.0  # convert to meters
        arm_len = 0.25
        offset_deg = math.degrees(math.atan2(h, arm_len))
        result["shoulder_lift"] = round(result["shoulder_lift"] - offset_deg, 2)

    return result


# ══════════ Calibration ══════════

CALIB_FILE = ROOT / "data" / "calibration.json"

calib_state = {
    "points": [],       # list of {pixel: [u,v], joints: {...}, robot_xy: [x,y]}
    "homography": None,  # 3x3 matrix (list of lists)
    "z_fixed": None,
}


def load_calibration():
    """Load calibration from file if exists."""
    global calib_state
    if CALIB_FILE.exists():
        calib_state = json.loads(CALIB_FILE.read_text())


def save_calibration():
    """Save calibration to file."""
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIB_FILE.write_text(json.dumps(calib_state, indent=2))


# Load calibration on startup
load_calibration()


@app.get("/api/calibrate/status")
async def calibrate_status():
    return {
        "points": calib_state["points"],
        "calibrated": calib_state["homography"] is not None,
        "z_fixed": calib_state["z_fixed"],
    }


@app.post("/api/calibrate/capture")
async def calibrate_capture():
    """Capture a snapshot from the top camera for calibration."""
    import cv2
    st = cam_state.get("top")
    if not st:
        return {"ok": False, "error": "Top camera not available"}
    with st["lock"]:
        f = st["frame"]
    if f is None:
        return {"ok": False, "error": "No frame available"}
    _, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
    return {"ok": True, "image": b64}


@app.post("/api/calibrate/save-point")
async def calibrate_save_point():
    """Save the current robot position for the latest pixel point."""
    if not robot_state["connected"]:
        return {"ok": False, "error": "Robot not connected"}

    joints = robot_get_positions()

    return {
        "ok": True,
        "joints": joints,
    }


@app.post("/api/calibrate/compute")
async def calibrate_compute():
    """Validate calibration points and save."""
    points = calib_state["points"]
    if len(points) < 4:
        return {"ok": False, "error": f"Need 4 points, have {len(points)}"}

    for i, p in enumerate(points):
        if "joints" not in p:
            return {"ok": False, "error": f"Point {i + 1} missing robot position"}

    calib_state["homography"] = True  # flag as calibrated
    save_calibration()

    return {"ok": True}


@app.post("/api/calibrate/reset")
async def calibrate_reset():
    """Reset calibration data."""
    calib_state["points"] = []
    calib_state["homography"] = None
    calib_state["z_fixed"] = None
    save_calibration()
    return {"ok": True}



@app.post("/api/calibrate/move-to")
async def calibrate_move_to(request: Request):
    """Click on top view → move robot with smooth safe path.

    If the direct path stays above safety_height → move directly.
    Otherwise → smooth arc that lifts shoulder_lift with a sine curve,
    keeping the arm above safety_height during transit.
    """
    if not robot_state["connected"]:
        return {"ok": False, "error": "Robot not connected"}
    if not calib_state["homography"]:
        return {"ok": False, "error": "Not calibrated"}

    data = await request.json()
    pixel = data.get("pixel")
    if not pixel:
        return {"ok": False, "error": "No pixel coordinate"}

    height_cm = data.get("height_cm", 0.0)
    safety_cm = data.get("safety_height_cm", 0.0)
    n_steps = int(data.get("n_steps", 15))

    target = interpolate_joints_from_pixel(pixel, calib_state["points"], height_cm=height_cm)
    if target is None:
        return {"ok": False, "error": "Interpolation failed — need 4 valid calibration points"}

    current = robot_get_positions()

    # Preserve current gripper position — reach does not touch gripper
    target["gripper"] = current["gripper"]

    # Compute the safety threshold for shoulder_lift
    # Lower shoulder_lift = arm higher.  sl_safe is the max allowed value.
    safe_target = interpolate_joints_from_pixel(pixel, calib_state["points"], height_cm=safety_cm)
    sl_safe = safe_target["shoulder_lift"] if safe_target else target["shoulder_lift"]

    sl_start = current["shoulder_lift"]
    sl_end = target["shoulder_lift"]

    # Check if the direct linear path is already safe
    # (both endpoints and everything in between stay ≤ sl_safe)
    path_safe = (sl_start <= sl_safe) and (sl_end <= sl_safe)

    if path_safe:
        # Direct move — no safety concern
        robot_send_positions(target)
        return {
            "ok": True, "pixel": pixel, "target_joints": target,
            "path": "direct", "safety_height_cm": safety_cm,
        }

    # ── Smooth arc path ──
    # All joints: linear interpolation current → target
    # shoulder_lift: linear - bump(t)
    #   bump(t) = bump_height × sin(πt)   → 0 at endpoints, max at midpoint
    # bump_height = just enough to clear safety + small margin

    sl_max = max(sl_start, sl_end)
    overshoot = max(0.0, sl_max - sl_safe)
    bump_height = min(overshoot + 2.0, 10.0)  # cap at 10° to prevent folding backward

    joint_names = list(current.keys())
    step_ms = 40

    for i in range(1, n_steps + 1):
        t = i / n_steps
        wp = {}
        for j in joint_names:
            wp[j] = round(current[j] + t * (target[j] - current[j]), 2)

        bump = bump_height * math.sin(math.pi * t)
        wp["shoulder_lift"] = round(wp["shoulder_lift"] - bump, 2)

        robot_send_positions(wp)
        await asyncio.sleep(step_ms / 1000.0)

    # Final — ensure exact target
    robot_send_positions(target)

    return {
        "ok": True, "pixel": pixel, "target_joints": target,
        "path": "arc", "n_steps": n_steps,
        "bump_height": round(bump_height, 2),
        "safety_height_cm": safety_cm,
    }


# ── Calibrate JSON API (receive JSON body) ──

@app.post("/api/calibrate/save-all")
async def calibrate_save_all(request: Request):
    """Save all calibration points at once: {points: [{pixel, joints, robot_xy, robot_z}, ...]}"""
    data = await request.json()
    calib_state["points"] = data.get("points", [])
    save_calibration()
    return {"ok": True}


# ══════════ Run Step ══════════

@app.post("/api/run/step")
async def run_step(request: Request):
    """Execute a single plan step on the robot.

    method_id → action:
      ik_reach_object_v1   → move gripper to target_bbox center (via calibration)
      act_gripper_grasp_v1 → close gripper (gripper=0)
      act_gripper_release_v1 → open gripper (gripper=100)
      act_drawer_open_v1   → (placeholder) move to bbox then pull back
      act_drawer_close_v1  → (placeholder) move to bbox then push forward
      act_door_open_v1     → (placeholder)
      act_door_close_v1    → (placeholder)
    """
    if not robot_state["connected"]:
        return {"ok": False, "error": "Robot not connected"}

    data = await request.json()
    method = data.get("method_id", "")
    bbox = data.get("target_bbox")  # [x_center, y_center, w, h] normalized 0-1

    if method == "ik_reach_object_v1":
        # Move to the bbox center using calibration
        if not calib_state.get("homography"):
            return {"ok": False, "error": "Not calibrated — go to Calibrate tab first"}
        if not bbox or len(bbox) < 2:
            return {"ok": False, "error": "No target_bbox"}

        # Convert normalized bbox center → pixel coordinates
        # Need image dimensions from the top camera
        cfg = load_config()
        cam_cfg = cfg.get("robot", {}).get("cameras", {}).get("top", {})
        img_w = cam_cfg.get("w", 640)
        img_h = cam_cfg.get("h", 480)
        pixel = [bbox[0] * img_w, bbox[1] * img_h]

        height_cm = float(data.get("height_cm", 0.0))
        safety_cm = float(data.get("safety_height_cm", 0.0))

        target = interpolate_joints_from_pixel(pixel, calib_state["points"], height_cm=height_cm)
        if target is None:
            return {"ok": False, "error": "Interpolation failed"}

        current = robot_get_positions()

        # Preserve current gripper position — IK reach does not touch gripper
        target["gripper"] = current["gripper"]

        safe_target = interpolate_joints_from_pixel(pixel, calib_state["points"], height_cm=safety_cm)
        sl_safe = safe_target["shoulder_lift"] if safe_target else target["shoulder_lift"]
        sl_start = current["shoulder_lift"]
        sl_end = target["shoulder_lift"]
        path_safe = (sl_start <= sl_safe) and (sl_end <= sl_safe)

        if path_safe:
            robot_send_positions(target)
        else:
            sl_max = max(sl_start, sl_end)
            overshoot = max(0.0, sl_max - sl_safe)
            bump_height = min(overshoot + 2.0, 10.0)
            n_steps = 15
            joint_names = list(current.keys())
            for i in range(1, n_steps + 1):
                t = i / n_steps
                wp = {}
                for j in joint_names:
                    wp[j] = round(current[j] + t * (target[j] - current[j]), 2)
                bump = bump_height * math.sin(math.pi * t)
                wp["shoulder_lift"] = round(wp["shoulder_lift"] - bump, 2)
                robot_send_positions(wp)
                await asyncio.sleep(0.04)
            robot_send_positions(target)

        return {"ok": True, "action": "reach", "pixel": pixel, "target_joints": target}

    elif method == "act_gripper_grasp_v1":
        joints = robot_get_positions()
        joints["gripper"] = 0.0
        robot_send_positions(joints)
        await asyncio.sleep(0.5)
        return {"ok": True, "action": "grasp"}

    elif method == "act_gripper_release_v1":
        joints = robot_get_positions()
        joints["gripper"] = 100.0
        robot_send_positions(joints)
        await asyncio.sleep(0.5)
        return {"ok": True, "action": "release"}

    elif method in ("act_drawer_open_v1", "act_drawer_close_v1",
                     "act_door_open_v1", "act_door_close_v1"):
        # Placeholder — for now just reach the target
        if bbox and len(bbox) >= 2 and calib_state.get("homography"):
            cfg = load_config()
            cam_cfg = cfg.get("robot", {}).get("cameras", {}).get("top", {})
            pixel = [bbox[0] * cam_cfg.get("w", 640), bbox[1] * cam_cfg.get("h", 480)]
            target = interpolate_joints_from_pixel(pixel, calib_state["points"], height_cm=float(data.get("height_cm", 0.0)))
            if target:
                robot_send_positions(target)
                await asyncio.sleep(0.5)
        return {"ok": True, "action": method, "note": "placeholder — reach only"}

    else:
        return {"ok": False, "error": f"Unknown method: {method}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
