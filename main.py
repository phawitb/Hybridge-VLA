import asyncio
import base64
import io
import json
import math
import os
import random
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
from fastapi.responses import FileResponse, HTMLResponse, Response
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


def _read_env():
    """Read .env file as dict."""
    env = {}
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _write_env(env: dict):
    """Write dict to .env file."""
    lines = [f"{k}={v}" for k, v in env.items()]
    (ROOT / ".env").write_text("\n".join(lines) + "\n")


@app.get("/api/env")
async def get_env():
    """Get env tokens (masked)."""
    env = _read_env()
    masked = {}
    for k, v in env.items():
        if v and len(v) > 8:
            masked[k] = v[:4] + "*" * (len(v) - 8) + v[-4:]
        elif v:
            masked[k] = "****"
        else:
            masked[k] = ""
    return {"ok": True, "env": masked, "keys": list(env.keys())}


@app.post("/api/env/save")
async def save_env(request: Request):
    """Save env tokens to .env file."""
    data = await request.json()
    env = _read_env()
    for key in ["GEMINI_API_KEY", "HF_TOKEN", "WB_API_KEY"]:
        if key in data:
            val = data[key].strip()
            # Skip masked values (don't overwrite with mask)
            if val and "*" not in val:
                env[key] = val
            elif not val:
                env.pop(key, None)
    _write_env(env)
    # Reload env vars
    load_dotenv(ROOT / ".env", override=True)
    return {"ok": True}


@app.get("/api/wb/status")
async def wb_status():
    """Check W&B login status."""
    env = _read_env()
    api_key = env.get("WB_API_KEY", "")
    if not api_key:
        return {"ok": True, "logged_in": False, "message": "No API key configured"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.wandb.ai/graphql",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": "{viewer{username}}"},
            )
            if r.status_code == 200:
                data = r.json()
                username = data.get("data", {}).get("viewer", {}).get("username", "")
                if username:
                    return {"ok": True, "logged_in": True, "username": username}
            return {"ok": True, "logged_in": False, "message": "Invalid API key"}
    except Exception as e:
        return {"ok": True, "logged_in": False, "message": str(e)}


@app.post("/api/wb/login")
async def wb_login(request: Request):
    """Login to W&B by verifying API key via HTTP and writing to netrc."""
    env = _read_env()
    api_key = env.get("WB_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "No WB_API_KEY in .env"}
    try:
        # Verify key via W&B API
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.wandb.ai/graphql",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": "{viewer{username}}"},
            )
            if r.status_code != 200:
                return {"ok": False, "error": f"API returned status {r.status_code}"}
            data = r.json()
            username = data.get("data", {}).get("viewer", {}).get("username", "")
            if not username:
                return {"ok": False, "error": "Invalid API key"}
        # Write to netrc so wandb/training scripts can authenticate
        import netrc as _netrc
        netrc_path = Path.home() / ".netrc"
        try:
            nrc = _netrc.netrc(str(netrc_path)) if netrc_path.exists() else _netrc.netrc()
        except Exception:
            nrc = None
        # Write/update the entry manually
        lines = []
        if netrc_path.exists():
            lines = netrc_path.read_text().splitlines()
        # Remove existing api.wandb.ai block
        new_lines = []
        skip = False
        for line in lines:
            if line.strip().startswith("machine") and "api.wandb.ai" in line:
                skip = True
                continue
            if skip and line.strip().startswith(("login", "password")):
                continue
            skip = False
            new_lines.append(line)
        # Append new entry
        new_lines.append("")
        new_lines.append("machine api.wandb.ai")
        new_lines.append(f"  login user")
        new_lines.append(f"  password {api_key}")
        netrc_path.write_text("\n".join(new_lines) + "\n")
        netrc_path.chmod(0o600)
        return {"ok": True, "message": f"Logged in as {username}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
        "camera_wrist_index": cams.get("wrist", {}).get("index", 0),
        "teleop": cfg.get("teleop", {}),
        "click_to_move": cfg.get("click_to_move", {"target_height": 0, "safety_height": 10}),
        "hf_repo_name": cfg.get("hf_repo_name", ""),
        "training_server": cfg.get("training_server", "http://100.87.242.52:8000"),
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
    if "hf_repo_name" in data:
        cfg["hf_repo_name"] = data["hf_repo_name"]
    if "training_server" in data:
        cfg["training_server"] = data["training_server"]
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


# ── Fake Camera ──
fakecam_state = {
    "running": False, "threads": [], "stop_event": None,
    "params": {},
    "cams": [
        {"input_idx": 0, "frame": None, "frame_lock": threading.Lock()},
        {"input_idx": 1, "frame": None, "frame_lock": threading.Lock()},
    ],
    "width": 640, "height": 480,
}


@app.post("/api/fakecam/start")
async def fakecam_start(request: Request):
    """Start fake cameras: reads from 2 input cameras, applies same augmentation, streams via WebSocket."""
    if fakecam_state["running"]:
        return {"ok": False, "error": "Already running"}
    data = await request.json()
    input_idx_0 = int(data.get("input_idx_0", 0))
    input_idx_1 = int(data.get("input_idx_1", 1))
    params = data.get("params", {})
    width = int(data.get("width", 640))
    height = int(data.get("height", 480))
    fakecam_state["cams"][0]["input_idx"] = input_idx_0
    fakecam_state["cams"][1]["input_idx"] = input_idx_1
    fakecam_state["params"] = params
    fakecam_state["width"] = width
    fakecam_state["height"] = height
    stop_event = threading.Event()
    fakecam_state["stop_event"] = stop_event

    def _run_cam(cam_slot):
        import cv2
        cam_state = fakecam_state["cams"][cam_slot]
        input_idx = cam_state["input_idx"]
        try:
            cap = cv2.VideoCapture(input_idx, cv2.CAP_AVFOUNDATION)
        except Exception:
            cap = cv2.VideoCapture(input_idx)
        if not cap.isOpened():
            print(f"[FakeCam{cam_slot}] Cannot open camera {input_idx}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fail_count = 0
        while not stop_event.is_set():
            try:
                ret, frame = cap.read()
            except Exception:
                ret = False
            if not ret:
                fail_count += 1
                if fail_count > 100:
                    print(f"[FakeCam{cam_slot}] Too many read failures, stopping")
                    break
                time.sleep(0.01)
                continue
            fail_count = 0
            fh, fw = frame.shape[:2]
            if fw != width or fh != height:
                frame = cv2.resize(frame, (width, height))
            cam_p = _build_cam_params(fakecam_state["params"])
            light_p = _build_light_params(fakecam_state["params"])
            if cam_p or light_p:
                frame = _aug_frame(frame, cam_p, light_p)
            with cam_state["frame_lock"]:
                cam_state["frame"] = frame
            time.sleep(1 / 30)
        cap.release()
        cam_state["frame"] = None

    threads = []
    for i in range(2):
        t = threading.Thread(target=_run_cam, args=(i,), daemon=True)
        threads.append(t)
    fakecam_state["threads"] = threads
    fakecam_state["running"] = True
    for t in threads:
        t.start()
    return {"ok": True}


@app.post("/api/fakecam/stop")
async def fakecam_stop():
    if not fakecam_state["running"]:
        return {"ok": False, "error": "Not running"}
    fakecam_state["stop_event"].set()
    fakecam_state["running"] = False
    for c in fakecam_state["cams"]:
        c["frame"] = None
    return {"ok": True}


@app.post("/api/fakecam/update-params")
async def fakecam_update_params(request: Request):
    data = await request.json()
    fakecam_state["params"] = data.get("params", {})
    return {"ok": True}


@app.get("/api/fakecam/params")
async def fakecam_get_params():
    """Return current fakecam augmentation params (for fakecam_inject.py --from-server)."""
    return fakecam_state["params"]


@app.post("/api/fakecam/save-params")
async def fakecam_save_params(request: Request):
    """Save current fakecam params to fakecam_params.json."""
    data = await request.json()
    params = data.get("params", fakecam_state["params"])
    path = ROOT / "fakecam_params.json"
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return {"ok": True, "path": str(path)}


@app.get("/api/fakecam/status")
async def fakecam_status():
    return {
        "running": fakecam_state["running"],
        "width": fakecam_state["width"],
        "height": fakecam_state["height"],
        "cams": [
            {"input_idx": c["input_idx"]}
            for c in fakecam_state["cams"]
        ],
    }


@app.websocket("/ws/fakecam/{cam_id}")
async def ws_fakecam(websocket: WebSocket, cam_id: int = 0):
    """Stream augmented camera frames as JPEG over WebSocket. cam_id: 0 or 1."""
    import cv2
    await websocket.accept()
    if cam_id < 0 or cam_id > 1:
        await websocket.close()
        return
    cam_state = fakecam_state["cams"][cam_id]
    try:
        while True:
            if not fakecam_state["running"]:
                await asyncio.sleep(0.1)
                continue
            with cam_state["frame_lock"]:
                frame = cam_state["frame"]
            if frame is None:
                await asyncio.sleep(0.03)
                continue
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            await websocket.send_bytes(jpg.tobytes())
            await asyncio.sleep(1 / 25)
    except WebSocketDisconnect:
        pass


def _build_cam_params(p):
    """Convert UI slider values to _aug_frame camera params."""
    cam = {}
    rotation = float(p.get("rotation", 0))
    translate_x = float(p.get("translate_x", 0))
    translate_y = float(p.get("translate_y", 0))
    scale = float(p.get("scale", 1.0))
    shear = float(p.get("shear", 0))
    if rotation:
        cam["angle"] = rotation
    if translate_x or translate_y:
        cam["tx"] = translate_x / 100  # UI sends percentage
        cam["ty"] = translate_y / 100
        cam["scale"] = scale
        cam["shear"] = shear / 100
    elif scale != 1.0 or shear:
        cam["tx"] = 0
        cam["ty"] = 0
        cam["scale"] = scale
        cam["shear"] = shear / 100
    return cam if cam else None


def _build_light_params(p):
    """Convert UI slider values to _aug_frame light params."""
    lp = {}
    brightness = float(p.get("brightness", 1.0))
    contrast = float(p.get("contrast", 1.0))
    saturation = float(p.get("saturation", 1.0))
    noise = float(p.get("noise", 0))
    blur = int(p.get("blur", 0))
    if brightness != 1.0:
        lp["brightness"] = brightness
    if contrast != 1.0:
        lp["contrast"] = contrast
    if saturation != 1.0:
        lp["saturation"] = saturation
    if noise > 0:
        lp["noise_s"] = noise
    if blur > 0:
        k = blur * 2 + 1  # must be odd
        lp["blur_k"] = k
        lp["blur_s"] = blur * 0.4
    return lp if lp else None


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
                    # Read tasks
                    task_list = []
                    tasks_path = d / "meta" / "tasks.parquet"
                    if tasks_path.exists():
                        try:
                            import pyarrow.parquet as pq
                            tbl = pq.read_table(str(tasks_path))
                            task_list = tbl.to_pydict().get("task", [])
                        except Exception:
                            pass
                    datasets.append({
                        "name": d.name,
                        "total_episodes": info.get("total_episodes", 0),
                        "total_frames": info.get("total_frames", 0),
                        "fps": info.get("fps", 0),
                        "robot_type": info.get("robot_type", ""),
                        "tasks": task_list,
                    })
                except Exception:
                    pass
    return {"ok": True, "datasets": datasets}


@app.post("/api/datasets/combine")
async def combine_datasets(request: Request):
    """Combine multiple datasets into a new one."""
    import shutil

    body = await request.json()
    source_names = body.get("datasets", [])
    new_name = body.get("name", "").strip()
    if not source_names or len(source_names) < 2:
        return {"ok": False, "error": "Select at least 2 datasets"}
    if not new_name:
        return {"ok": False, "error": "Enter a name for the new dataset"}
    # Sanitize name
    new_name = new_name.replace(" ", "-").replace("/", "-")

    new_dir = ROOT / "data" / new_name
    if new_dir.exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        # Validate all sources exist and collect info
        sources = []
        for sname in source_names:
            sdir = ROOT / "data" / sname
            info_path = sdir / "meta" / "info.json"
            if not info_path.exists():
                return {"ok": False, "error": f"Dataset '{sname}' not found"}
            info = json.loads(info_path.read_text())
            sources.append({"name": sname, "dir": sdir, "info": info})

        # Use first source as template for info.json structure
        base_info = dict(sources[0]["info"])
        fps = base_info.get("fps", 30)

        # Collect all tasks (deduplicate)
        all_tasks = []
        task_set = set()
        for src in sources:
            tp = src["dir"] / "meta" / "tasks.parquet"
            if tp.exists():
                try:
                    tbl = pq.read_table(str(tp))
                    for t in tbl.to_pydict().get("task", []):
                        if t not in task_set:
                            task_set.add(t)
                            all_tasks.append(t)
                except Exception:
                    pass
        task_to_idx = {t: i for i, t in enumerate(all_tasks)}

        # Merge data and episode metadata from all sources
        combined_data = []
        combined_ep = []
        new_ep_idx = 0
        global_frame_idx = 0

        for src in sources:
            sdir = src["dir"]

            # Read episode metadata
            ep_meta_list = []
            ep_dir = sdir / "meta" / "episodes"
            if ep_dir.exists():
                for pf in sorted(ep_dir.rglob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                    except Exception:
                        continue
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        ep_meta_list.append({k: v[i] for k, v in d.items()})
            ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

            # Read data frames
            data_dir = sdir / "data"
            src_frames = {}  # ep -> [rows]
            if data_dir.exists():
                for pf in sorted(data_dir.rglob("*.parquet")):
                    try:
                        tbl = pq.read_table(str(pf))
                    except Exception:
                        continue
                    d = tbl.to_pydict()
                    for i in range(len(d.get("episode_index", []))):
                        ep = d["episode_index"][i]
                        row = {}
                        for k, v in d.items():
                            val = v[i]
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            row[k] = val
                        src_frames.setdefault(ep, []).append(row)

            for ep in src_frames:
                src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

            # Read tasks for this source to remap task_index
            src_tasks = []
            tp = sdir / "meta" / "tasks.parquet"
            if tp.exists():
                try:
                    tbl = pq.read_table(str(tp))
                    src_tasks = tbl.to_pydict().get("task", [])
                except Exception:
                    pass

            # Process each episode
            for old_ep in sorted(src_frames.keys()):
                frames = src_frames[old_ep]
                ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
                ep_global_start = global_frame_idx

                for fi, row in enumerate(frames):
                    new_row = dict(row)
                    new_row["episode_index"] = new_ep_idx
                    new_row["frame_index"] = fi
                    new_row["index"] = global_frame_idx
                    # Remap task_index
                    old_ti = row.get("task_index", 0)
                    if old_ti < len(src_tasks) and src_tasks[old_ti] in task_to_idx:
                        new_row["task_index"] = task_to_idx[src_tasks[old_ti]]
                    else:
                        new_row["task_index"] = 0
                    combined_data.append(new_row)
                    global_frame_idx += 1

                # Build new episode meta
                new_meta = {}
                new_meta["episode_index"] = new_ep_idx
                if ep_meta and "tasks" in ep_meta:
                    new_meta["tasks"] = ep_meta["tasks"]
                elif src_tasks:
                    new_meta["tasks"] = [src_tasks[0]]
                else:
                    new_meta["tasks"] = all_tasks[:1] if all_tasks else [""]
                new_meta["length"] = len(frames)
                new_meta["data/chunk_index"] = 0
                new_meta["data/file_index"] = 0
                new_meta["dataset_from_index"] = ep_global_start
                new_meta["dataset_to_index"] = global_frame_idx

                # Video timestamps — reference original video files
                for cam in ["observation.images.top", "observation.images.wrist"]:
                    cam_key = f"videos/{cam}"
                    if ep_meta:
                        orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0)
                        orig_file = ep_meta.get(f"{cam_key}/file_index", 0)
                        src_vid = sdir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                    else:
                        src_vid = None
                    new_meta[f"{cam_key}/from_timestamp"] = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                    new_meta[f"{cam_key}/to_timestamp"] = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                    new_meta[f"{cam_key}/chunk_index"] = 0
                    new_meta[f"{cam_key}/file_index"] = 0
                    new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid and src_vid.exists() else None

                # Copy stats if available
                if ep_meta:
                    for k, v in ep_meta.items():
                        if k.startswith("stats/"):
                            new_meta[k] = v
                    if "meta/episodes/chunk_index" in ep_meta:
                        new_meta["meta/episodes/chunk_index"] = 0
                    if "meta/episodes/file_index" in ep_meta:
                        new_meta["meta/episodes/file_index"] = 0

                combined_ep.append(new_meta)
                new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data found in source datasets"}

        # Create output directory structure
        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        # Copy video files — each source gets its own file index
        for cam in ["observation.images.top", "observation.images.wrist"]:
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids = []
            src_vid_set = set()
            for ep_meta in combined_ep:
                sv = ep_meta.get(f"_src_vid_{cam}")
                if sv and sv not in src_vid_set:
                    src_vid_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                dst = vid_dir / f"file-{fi:03d}.mp4"
                shutil.copy2(sv, str(dst))
                vid_map[sv] = fi
            cam_key = f"videos/{cam}"
            for ep_meta in combined_ep:
                sv = ep_meta.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    ep_meta[f"{cam_key}/file_index"] = vid_map[sv]

        # Write data parquet
        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

        # Write episode meta parquet
        for ep in combined_ep:
            for k in list(ep.keys()):
                if k.startswith("_src_vid_"):
                    del ep[k]
        cols = {k: [r[k] for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

        # Write tasks parquet (task strings as index, matching LeRobot format)
        import pandas as pd
        tasks_df = pd.DataFrame({"task_index": list(range(len(all_tasks)))}, index=pd.Index(all_tasks, name="task"))
        tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        # Write info.json
        new_info = dict(base_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["total_tasks"] = len(all_tasks)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
        _compute_stats_json(combined_data, new_dir)

        print(f"[Dataset] Combined {len(source_names)} datasets into {new_name} ({new_ep_idx} eps, {len(combined_data)} frames)")
        return {
            "ok": True,
            "name": new_name,
            "episodes": new_ep_idx,
            "frames": len(combined_data),
            "tasks": all_tasks,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/random-subset")
async def random_subset_dataset(name: str, request: Request):
    """Create a new dataset by randomly sampling episodes from an existing one."""
    import shutil
    import random

    body = await request.json()
    num_episodes = int(body.get("num_episodes", 10))
    new_name = body.get("new_name", "").strip()
    if not new_name:
        return {"ok": False, "error": "Enter a name for the new dataset"}
    new_name = new_name.replace(" ", "-").replace("/", "-")
    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
        return {"ok": False, "error": "Invalid name. Use only letters, numbers, dash, underscore."}

    src_dir = ROOT / "data" / name
    new_dir = ROOT / "data" / new_name
    info_path = src_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": f"Dataset '{name}' not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        src_info = json.loads(info_path.read_text())
        total_eps = src_info.get("total_episodes", 0)
        if num_episodes > total_eps:
            return {"ok": False, "error": f"Requested {num_episodes} episodes but dataset only has {total_eps}"}
        if num_episodes < 1:
            return {"ok": False, "error": "Need at least 1 episode"}

        # Randomly select episodes
        selected_eps = sorted(random.sample(range(total_eps), num_episodes))

        # Read tasks
        all_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                tbl = pq.read_table(str(tp))
                all_tasks = tbl.to_pydict().get("task", [])
            except Exception:
                pass

        # Read episode metadata
        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        # Read data frames
        data_dir = src_dir / "data"
        src_frames = {}
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    if ep not in selected_eps:
                        continue
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)

        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        # Build new dataset
        combined_data = []
        combined_ep = []
        new_ep_idx = 0
        global_frame_idx = 0

        for old_ep in selected_eps:
            frames = src_frames.get(old_ep, [])
            if not frames:
                continue
            ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)
            ep_global_start = global_frame_idx

            for fi, row in enumerate(frames):
                new_row = dict(row)
                new_row["episode_index"] = new_ep_idx
                new_row["frame_index"] = fi
                new_row["index"] = global_frame_idx
                combined_data.append(new_row)
                global_frame_idx += 1

            new_meta = {}
            new_meta["episode_index"] = new_ep_idx
            if ep_meta and "tasks" in ep_meta:
                new_meta["tasks"] = ep_meta["tasks"]
            elif all_tasks:
                new_meta["tasks"] = [all_tasks[0]]
            else:
                new_meta["tasks"] = [""]
            new_meta["length"] = len(frames)
            new_meta["data/chunk_index"] = 0
            new_meta["data/file_index"] = 0
            new_meta["dataset_from_index"] = ep_global_start
            new_meta["dataset_to_index"] = global_frame_idx

            for cam in ["observation.images.top", "observation.images.wrist"]:
                cam_key = f"videos/{cam}"
                if ep_meta:
                    orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0)
                    orig_file = ep_meta.get(f"{cam_key}/file_index", 0)
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"
                else:
                    src_vid = None
                new_meta[f"{cam_key}/from_timestamp"] = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                new_meta[f"{cam_key}/to_timestamp"] = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                new_meta[f"{cam_key}/chunk_index"] = 0
                new_meta[f"{cam_key}/file_index"] = 0
                new_meta[f"_src_vid_{cam}"] = str(src_vid) if src_vid and src_vid.exists() else None

            if ep_meta:
                for k, v in ep_meta.items():
                    if k.startswith("stats/"):
                        new_meta[k] = v
                if "meta/episodes/chunk_index" in ep_meta:
                    new_meta["meta/episodes/chunk_index"] = 0
                if "meta/episodes/file_index" in ep_meta:
                    new_meta["meta/episodes/file_index"] = 0

            combined_ep.append(new_meta)
            new_ep_idx += 1

        if not combined_data:
            return {"ok": False, "error": "No data found for selected episodes"}

        # Create output directory
        new_dir.mkdir(parents=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True)

        # Copy video files
        for cam in ["observation.images.top", "observation.images.wrist"]:
            vid_dir = new_dir / "videos" / cam / "chunk-000"
            vid_dir.mkdir(parents=True)
            src_vids = []
            src_vid_set = set()
            for ep_meta in combined_ep:
                sv = ep_meta.get(f"_src_vid_{cam}")
                if sv and sv not in src_vid_set:
                    src_vid_set.add(sv)
                    src_vids.append(sv)
            vid_map = {}
            for fi, sv in enumerate(src_vids):
                dst = vid_dir / f"file-{fi:03d}.mp4"
                shutil.copy2(sv, str(dst))
                vid_map[sv] = fi
            cam_key = f"videos/{cam}"
            for ep_meta in combined_ep:
                sv = ep_meta.pop(f"_src_vid_{cam}", None)
                if sv and sv in vid_map:
                    ep_meta[f"{cam_key}/file_index"] = vid_map[sv]

        # Write data parquet
        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

        # Write episode meta parquet
        for ep in combined_ep:
            for k in list(ep.keys()):
                if k.startswith("_src_vid_"):
                    del ep[k]
        cols = {k: [r[k] for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

        # Write tasks parquet
        import pandas as pd
        if all_tasks:
            tasks_df = pd.DataFrame({"task_index": list(range(len(all_tasks)))}, index=pd.Index(all_tasks, name="task"))
            tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        # Write info.json
        new_info = dict(src_info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
        _compute_stats_json(combined_data, new_dir)

        print(f"[Dataset] Random subset from {name}: {new_name} ({new_ep_idx} eps, {len(combined_data)} frames)")
        return {
            "ok": True,
            "name": new_name,
            "episodes": new_ep_idx,
            "frames": len(combined_data),
            "selected": selected_eps,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


@app.delete("/api/datasets/{name}")
async def delete_dataset(name: str):
    """Delete an entire dataset directory."""
    import shutil
    dataset_dir = ROOT / "data" / name
    if not dataset_dir.exists() or not (dataset_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    try:
        shutil.rmtree(dataset_dir)
        print(f"[Dataset] Deleted dataset: {name}")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datasets/{name}/rename")
async def rename_dataset(name: str, request: Request):
    """Rename a dataset directory."""
    data = await request.json()
    new_name = data.get("new_name", "").strip()
    if not new_name:
        return {"ok": False, "error": "New name is required"}
    if new_name == name:
        return {"ok": False, "error": "New name is the same as current name"}
    # Validate name (alphanumeric, dash, underscore)
    if not re.match(r'^[a-zA-Z0-9_\-]+$', new_name):
        return {"ok": False, "error": "Invalid name. Use only letters, numbers, dash, underscore."}
    dataset_dir = ROOT / "data" / name
    new_dir = ROOT / "data" / new_name
    if not dataset_dir.exists():
        return {"ok": False, "error": "Dataset not found"}
    if new_dir.exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}
    try:
        dataset_dir.rename(new_dir)
        print(f"[Dataset] Renamed: {name} -> {new_name}")
        return {"ok": True, "new_name": new_name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Push to Hugging Face ──

hf_push_state = {
    "running": False,
    "progress": 0,
    "status": "",
    "error": None,
    "logs": [],
}


def _run_hf_push(dataset_name: str, repo_id: str):
    """Background thread: push dataset to Hugging Face Hub."""
    state = hf_push_state
    state["running"] = True
    state["progress"] = 10
    state["status"] = "Preparing upload..."
    state["error"] = None
    state["logs"] = []

    dataset_dir = ROOT / "data" / dataset_name

    try:
        # Build list of files to upload (exclude augmentation.json and tmp dirs)
        upload_files = []
        for f in sorted(dataset_dir.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(dataset_dir)
            # Skip augmentation metadata & tmp streaming dirs
            if rel.name == "augmentation.json":
                continue
            parts = rel.parts
            if parts[0].startswith("tmp"):
                continue
            upload_files.append((str(f), str(rel)))

        if not upload_files:
            state["error"] = "No files to upload"
            state["status"] = "Error: no files"
            return

        state["status"] = f"Uploading {len(upload_files)} files to {repo_id}..."
        state["progress"] = 20
        state["logs"].append(f"$ Preparing {len(upload_files)} files for {repo_id}")

        # Use huggingface_hub via lerobot env python
        script = f"""
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi()
repo_id = {repr(repo_id)}
print(f"Creating/checking repo: {{repo_id}}", flush=True)
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
print("Repo ready. Preparing commit...", flush=True)
ops = []
files = {upload_files!r}
for local, remote in files:
    ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local))
    print(f"ADD {{remote}}", flush=True)
print(f"Committing {{len(ops)}} files...", flush=True)
api.create_commit(
    repo_id=repo_id,
    repo_type="dataset",
    operations=ops,
    commit_message="Upload dataset via Hybridge VLA",
)
print("DONE", flush=True)
"""
        lerobot_python = "/opt/miniconda3/envs/lerobot/bin/python"
        if not Path(lerobot_python).exists():
            lerobot_python = "python3"

        # Pass HF_TOKEN from .env to subprocess
        sub_env = dict(os.environ)
        hf_token = _read_env().get("HF_TOKEN", "")
        if hf_token:
            sub_env["HF_TOKEN"] = hf_token

        proc = subprocess.Popen(
            [lerobot_python, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=sub_env,
        )

        total = len(upload_files)
        uploaded = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            state["logs"].append(line)
            # Keep log buffer bounded
            if len(state["logs"]) > 500:
                state["logs"] = state["logs"][-300:]
            if line.startswith("ADD "):
                uploaded += 1
                pct = 20 + int(uploaded / total * 70)
                state["progress"] = min(pct, 90)
                state["status"] = f"Uploading ({uploaded}/{total}): {line[4:]}"
            elif line == "DONE":
                state["progress"] = 100
                state["status"] = f"Pushed {total} files to {repo_id}"

        proc.wait()
        if proc.returncode != 0:
            state["logs"].append(f"[EXIT CODE {proc.returncode}]")
            state["error"] = f"Process exited with code {proc.returncode}"
            state["status"] = f"Error: exit code {proc.returncode}"
        else:
            state["progress"] = 100
            state["status"] = f"Done! Pushed to {repo_id}"
            state["logs"].append(f"Successfully pushed to {repo_id}")
            print(f"[HF Push] Pushed {dataset_name} -> {repo_id} ({total} files)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.post("/api/datasets/{name}/push-hf")
async def push_to_hf(name: str, request: Request):
    """Push dataset to Hugging Face Hub."""
    if hf_push_state["running"]:
        return {"ok": False, "error": "Push already in progress"}
    dataset_dir = ROOT / "data" / name
    if not dataset_dir.exists() or not (dataset_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Dataset not found"}
    body = await request.json()
    repo_id = body.get("repo_id", "").strip()
    if not repo_id or "/" not in repo_id:
        return {"ok": False, "error": "Invalid repo_id (format: username/dataset-name)"}
    threading.Thread(target=_run_hf_push, args=(name, repo_id), daemon=True).start()
    return {"ok": True, "message": f"Pushing to {repo_id}..."}


@app.get("/api/hf-push/status")
async def hf_push_status():
    """Get HF push progress."""
    return dict(hf_push_state)


# ── Download from Hugging Face ──

hf_download_state = {
    "running": False,
    "progress": 0,
    "status": "",
    "error": None,
    "logs": [],
}


def _run_hf_download(repo_id: str, local_name: str):
    """Background thread: download dataset from Hugging Face Hub."""
    state = hf_download_state
    state["running"] = True
    state["progress"] = 10
    state["status"] = f"Downloading {repo_id}..."
    state["error"] = None
    state["logs"] = []

    dataset_dir = ROOT / "data" / local_name

    try:
        script = f"""
import sys, os
from huggingface_hub import snapshot_download
repo_id = {repr(repo_id)}
local_dir = {repr(str(dataset_dir))}
print(f"Downloading {{repo_id}} to {{local_dir}}", flush=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
)
print("DONE", flush=True)
"""
        lerobot_python = "/opt/miniconda3/envs/lerobot/bin/python"
        if not Path(lerobot_python).exists():
            lerobot_python = "python3"

        sub_env = dict(os.environ)
        hf_token = _read_env().get("HF_TOKEN", "")
        if hf_token:
            sub_env["HF_TOKEN"] = hf_token

        proc = subprocess.Popen(
            [lerobot_python, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=sub_env,
        )

        state["progress"] = 20
        state["logs"].append(f"$ Downloading {repo_id} -> data/{local_name}")

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            state["logs"].append(line)
            if len(state["logs"]) > 500:
                state["logs"] = state["logs"][-300:]
            if "Downloading" in line or "Fetching" in line:
                state["status"] = line[:120]
                state["progress"] = min(state["progress"] + 2, 85)
            elif line == "DONE":
                state["progress"] = 100
                state["status"] = f"Downloaded {repo_id} to data/{local_name}"

        proc.wait()
        if proc.returncode != 0:
            state["logs"].append(f"[EXIT CODE {proc.returncode}]")
            state["error"] = f"Process exited with code {proc.returncode}"
            state["status"] = f"Error: exit code {proc.returncode}"
        else:
            state["progress"] = 100
            state["status"] = f"Done! Downloaded to data/{local_name}"
            state["logs"].append(f"Successfully downloaded {repo_id}")
            print(f"[HF Download] {repo_id} -> data/{local_name}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.post("/api/datasets/download-hf")
async def download_from_hf(request: Request):
    """Download a dataset from Hugging Face Hub."""
    if hf_download_state["running"]:
        return {"ok": False, "error": "Download already in progress"}
    body = await request.json()
    repo_id = body.get("repo_id", "").strip()
    if not repo_id or "/" not in repo_id:
        return {"ok": False, "error": "Invalid repo_id (format: username/dataset-name)"}

    # Extract local name from repo_id (part after /)
    local_name = repo_id.split("/", 1)[1]
    local_name = local_name.replace(" ", "-")

    # Check if dataset already exists locally
    dataset_dir = ROOT / "data" / local_name
    if dataset_dir.exists() and (dataset_dir / "meta" / "info.json").exists():
        return {
            "ok": False,
            "error": f"Dataset '{local_name}' already exists locally. Please delete it first before downloading."
        }

    threading.Thread(target=_run_hf_download, args=(repo_id, local_name), daemon=True).start()
    return {"ok": True, "message": f"Downloading {repo_id}...", "local_name": local_name}


@app.get("/api/hf-download/status")
async def hf_download_status():
    """Get HF download progress."""
    return dict(hf_download_state)


# ── Training Server Proxy ──

TRAIN_LOG_DIR = ROOT / "data" / "training_logs"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _training_server_url() -> str:
    cfg = load_config()
    return cfg.get("training_server", "http://100.87.242.52:8000").rstrip("/")


def _save_job_local(job_id: str, status_data: dict = None, log_lines: list = None):
    """Save/update job data locally for history."""
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    # Load existing or create new
    if job_path.exists():
        existing = json.loads(job_path.read_text())
    else:
        existing = {"job_id": job_id, "logs": []}
    if status_data:
        existing["status"] = status_data.get("status", existing.get("status"))
        existing["returncode"] = status_data.get("returncode", existing.get("returncode"))
        for k in ("cmd", "config", "job_name", "pid", "started_at", "finished_at", "log_file", "wb_url"):
            if k in status_data:
                existing[k] = status_data[k]
    if log_lines and isinstance(log_lines, list):
        # Append new lines (deduplicate by keeping the longer set)
        if len(log_lines) > len(existing.get("logs", [])):
            existing["logs"] = log_lines
    existing["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    job_path.write_text(json.dumps(existing, ensure_ascii=False, indent=1))


@app.post("/api/training/download")
async def training_download(request: Request):
    """Proxy: download dataset on training server."""
    body = await request.json()
    url = _training_server_url()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(f"{url}/download", json=body)
            if not r.text.strip():
                return {"ok": False, "error": f"Server returned empty response (HTTP {r.status_code})"}
            data = r.json()
            if data.get("job_id"):
                _save_job_local(data["job_id"], {"status": "downloading", "cmd": data.get("cmd"), "config": body})
            return data
        except httpx.TimeoutException:
            return {"ok": False, "error": "Training server timeout (120s)."}
        except Exception as e:
            return {"ok": False, "error": str(e)}


@app.post("/api/training/train")
async def training_start(request: Request):
    """Proxy: start training on remote server. Job may start immediately or be queued."""
    try:
        body = await request.json()
    except Exception as e:
        return {"ok": False, "error": f"Invalid request body: {e}"}
    url = _training_server_url()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(f"{url}/train", json=body)
            try:
                data = r.json()
            except Exception:
                if r.status_code == 500:
                    # Server may have started the job but crashed building the response.
                    # Poll /jobs to find the newly started job.
                    return await _recover_after_500(client, url, body)
                return {"ok": False, "error": f"Server returned non-JSON (HTTP {r.status_code}): {r.text[:200]}"}
            if data.get("job_id"):
                status = data.get("status", "running")
                save_data = {"status": status, "cmd": data.get("cmd"), "config": body, "job_name": body.get("job_name")}
                if data.get("wb_url"):
                    save_data["wb_url"] = data["wb_url"]
                _save_job_local(data["job_id"], save_data)
            return data
        except httpx.TimeoutException:
            return {"ok": False, "error": "Training server timeout (120s). Server may be busy — check queue."}
        except Exception as e:
            return {"ok": False, "error": f"Proxy error: {type(e).__name__}: {e}"}


async def _recover_after_500(client, url, body):
    """After a 500 from /train, check if the job actually started."""
    import asyncio
    await asyncio.sleep(1)
    try:
        r = await client.get(f"{url}/jobs")
        jobs = r.json()
        # Find the most recent running job
        running = [(jid, j) for jid, j in jobs.items() if j.get("status") == "running"]
        if running:
            jid = running[-1][0]
            _save_job_local(jid, {"status": "running", "config": body, "job_name": body.get("job_name")})
            return {"job_id": jid, "status": "running",
                    "note": "Server returned 500 but job started successfully."}
        # Check queue
        qr = await client.get(f"{url}/queue")
        qjobs = qr.json().get("jobs", [])
        if qjobs:
            qj = qjobs[-1]
            jid = qj.get("job_id", "")
            _save_job_local(jid, {"status": "queued", "config": body, "job_name": body.get("job_name")})
            return {"job_id": jid, "status": "queued",
                    "note": "Server returned 500 but job was queued successfully."}
    except Exception:
        pass
    return {"ok": False, "error": "Server returned 500. Job may have started — check server status."}


@app.get("/api/training/queue")
async def training_queue():
    """Proxy: list queued jobs on training server."""
    url = _training_server_url()
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{url}/queue")
            data = r.json()
            # Enrich queued jobs with local config
            enriched = []
            for qj in data.get("jobs", []):
                jid = qj.get("job_id") or qj.get("id", "")
                if jid:
                    job_path = TRAIN_LOG_DIR / f"{jid}.json"
                    if job_path.exists():
                        local = json.loads(job_path.read_text())
                        qj["config"] = qj.get("config") or local.get("config") or {}
                        qj["job_name"] = qj.get("job_name") or local.get("job_name")
                enriched.append(qj)
            return {"ok": True, "queue_length": data.get("queue_length", len(enriched)), "jobs": enriched}
        except Exception as e:
            return {"ok": False, "error": str(e), "queue_length": 0, "jobs": []}


@app.delete("/api/training/queue/{job_id}")
async def training_cancel_queued(job_id: str):
    """Proxy: cancel a queued job before it starts."""
    url = _training_server_url()
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.delete(f"{url}/queue/{job_id}")
            data = r.json()
            _save_job_local(job_id, {"status": "cancelled"})
            return data
        except Exception as e:
            return {"error": str(e)}


@app.get("/api/training/jobs")
async def training_jobs():
    """Proxy: list training jobs + queue, merged with local history."""
    url = _training_server_url()
    server_jobs = {}
    queued_ids: set = set()
    server_online = False
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{url}/jobs")
            server_jobs = r.json()
            server_online = True
            # Save each to local (updates status from server)
            for jid, jdata in server_jobs.items():
                _save_job_local(jid, jdata)
        except Exception:
            pass
        # Also fetch queue to include queued jobs
        if server_online:
            try:
                qr = await client.get(f"{url}/queue")
                qdata = qr.json()
                for qj in qdata.get("jobs", []):
                    qid = qj.get("job_id") or qj.get("id", "")
                    if qid and qid not in server_jobs:
                        queued_ids.add(qid)
                        server_jobs[qid] = {"status": "queued", "returncode": None}
                        _save_job_local(qid, {"status": "queued"})
            except Exception:
                pass

    # Merge with local history — enrich server jobs with locally saved config
    merged = dict(server_jobs)
    for f in sorted(TRAIN_LOG_DIR.glob("*.json"), reverse=True):
        try:
            local = json.loads(f.read_text())
            jid = local.get("job_id", f.stem)
            config = local.get("config") or {}
            if jid in merged:
                # Server job exists but may lack config — enrich from local
                if not merged[jid].get("config"):
                    merged[jid]["config"] = config
                if not merged[jid].get("job_name"):
                    merged[jid]["job_name"] = local.get("job_name")
                if not merged[jid].get("dataset_repo_id"):
                    merged[jid]["dataset_repo_id"] = config.get("dataset_repo_id") or config.get("repo_id", "")
            else:
                local_status = local.get("status", "unknown")
                # If server is online but doesn't list this job, it's no longer active
                if server_online and local_status in ("running", "queued", "downloading"):
                    local_status = "stopped"
                    _save_job_local(jid, {"status": "stopped"})
                merged[jid] = {
                    "status": local_status,
                    "returncode": local.get("returncode"),
                    "job_name": local.get("job_name"),
                    "config": config,
                    "dataset_repo_id": config.get("dataset_repo_id") or config.get("repo_id", ""),
                    "last_updated": local.get("last_updated"),
                    "wb_url": local.get("wb_url"),
                    "local_only": True,
                }
        except Exception:
            pass
    return {"server_online": server_online, "jobs": merged}


@app.get("/api/training/jobs/{job_id}/status")
async def training_job_status(job_id: str):
    """Proxy: get training job status, enriched with local config."""
    url = _training_server_url()
    server_data = None
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{url}/jobs/{job_id}/status")
            data = r.json()
            if "detail" not in data:
                _save_job_local(job_id, data)
                server_data = data
        except Exception:
            pass
    # Always enrich with local data (config, job_name, etc.)
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if job_path.exists():
        local = json.loads(job_path.read_text())
        result = {
            "job_id": job_id,
            "status": (server_data or {}).get("status") or local.get("status", "unknown"),
            "returncode": (server_data or {}).get("returncode", local.get("returncode")),
            "config": local.get("config") or {},
            "job_name": local.get("job_name"),
            "cmd": local.get("cmd"),
            "started_at": (server_data or {}).get("started_at") or local.get("started_at"),
            "finished_at": (server_data or {}).get("finished_at") or local.get("finished_at"),
            "wb_url": local.get("wb_url"),
        }
        if not server_data:
            result["local_only"] = True
        return result
    if server_data:
        return {"job_id": job_id, **server_data}
    return {"error": "Job not found"}


@app.get("/api/training/jobs/{job_id}/logs")
async def training_job_logs(job_id: str, tail: int = 100):
    """Proxy: get training job logs. Falls back to local."""
    url = _training_server_url()
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(f"{url}/jobs/{job_id}/logs", params={"tail": tail})
            data = r.json()
            if "detail" not in data:
                lines = data.get("lines", [])
                _save_job_local(job_id, log_lines=lines)
                return data
        except Exception:
            pass
    # Fallback to local
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if job_path.exists():
        local = json.loads(job_path.read_text())
        lines = local.get("logs", [])
        if tail:
            lines = lines[-tail:]
        return {"job_id": job_id, "lines": lines, "local_only": True}
    return {"error": "Log not found"}


@app.delete("/api/training/jobs/{job_id}")
async def training_stop_job(job_id: str):
    """Proxy: stop a training job."""
    url = _training_server_url()
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.delete(f"{url}/jobs/{job_id}")
            data = r.json()
            _save_job_local(job_id, {"status": "stopped"})
            return data
        except Exception as e:
            return {"error": str(e)}


@app.post("/api/training/kill-all")
async def training_kill_all():
    """Kill all running jobs and clear the queue on the remote server."""
    url = _training_server_url()
    results = {"killed": [], "cancelled": [], "errors": []}
    async with httpx.AsyncClient(timeout=15) as client:
        # 1. Get running jobs and kill them
        try:
            r = await client.get(f"{url}/jobs")
            for jid, jdata in r.json().items():
                if jdata.get("status") == "running":
                    try:
                        await client.delete(f"{url}/jobs/{jid}")
                        _save_job_local(jid, {"status": "stopped"})
                        results["killed"].append(jid)
                    except Exception as e:
                        results["errors"].append(f"kill {jid}: {e}")
        except Exception as e:
            results["errors"].append(f"list jobs: {e}")
        # 2. Get queued jobs and cancel them
        try:
            qr = await client.get(f"{url}/queue")
            for qj in qr.json().get("jobs", []):
                qid = qj.get("job_id") or qj.get("id", "")
                if qid:
                    try:
                        await client.delete(f"{url}/queue/{qid}")
                        _save_job_local(qid, {"status": "cancelled"})
                        results["cancelled"].append(qid)
                    except Exception as e:
                        results["errors"].append(f"cancel {qid}: {e}")
        except Exception as e:
            results["errors"].append(f"list queue: {e}")
    return {"ok": True, **results}


@app.delete("/api/training/history/{job_id}")
async def delete_job_history(job_id: str):
    """Delete a local job history entry."""
    job_path = TRAIN_LOG_DIR / f"{job_id}.json"
    if job_path.exists():
        job_path.unlink()
        return {"ok": True}
    return {"ok": False, "error": "Not found"}


@app.get("/api/datasets/{name}/segments/{ep}")
async def get_segments(name: str, ep: int):
    """Get saved segments for an episode."""
    seg_path = ROOT / "data" / name / "meta" / "segments.json"
    if seg_path.exists():
        try:
            data = json.loads(seg_path.read_text())
            segs = data.get(str(ep))
            if segs:
                return {"ok": True, "segments": segs}
        except Exception:
            pass
    return {"ok": True, "segments": None}


@app.post("/api/datasets/{name}/segments/{ep}")
async def save_segments(name: str, ep: int, request: Request):
    """Save segments for an episode."""
    body = await request.json()
    segments = body.get("segments", [])
    seg_path = ROOT / "data" / name / "meta" / "segments.json"
    data = {}
    if seg_path.exists():
        try:
            data = json.loads(seg_path.read_text())
        except Exception:
            pass
    data[str(ep)] = segments
    seg_path.write_text(json.dumps(data, indent=2))
    return {"ok": True}


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
    # Read augmentation metadata if exists
    aug_path = ROOT / "data" / name / "meta" / "augmentation.json"
    aug_info = None
    if aug_path.exists():
        try:
            aug_info = json.loads(aug_path.read_text())
        except Exception:
            pass
    return {"ok": True, "info": info, "tasks": task_list, "augmentation": aug_info}


@app.get("/api/datasets/{name}/episodes")
async def dataset_episodes(name: str):
    """Get per-episode metadata from episodes parquet."""
    ep_dir = ROOT / "data" / name / "meta" / "episodes"
    episodes = []
    if ep_dir.exists():
        try:
            import pyarrow.parquet as pq
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    episodes.append({k: v[i] for k, v in d.items()})
        except Exception:
            pass
    return {"ok": True, "episodes": episodes}


def _match_model_to_dataset(model_name: str, ds_names: list, hf_user: str) -> str | None:
    """Find the most likely source dataset for a model by name similarity.

    Strategy: strip known prefixes from model name, then find the dataset
    whose name is the longest match within the remaining string.
    e.g. 'smolvla_V1-4tasks-augnormal5x' -> 'V1-4tasks-augnormal5x'
    """
    # Common model name prefixes to strip
    prefixes = ["smolvla_", "smolvla-", "pi0_", "pi0-", "act_", "act-",
                "diffusion_", "diffusion-", "vla_", "vla-", "model_", "model-"]
    stripped = model_name
    for p in prefixes:
        if stripped.lower().startswith(p):
            stripped = stripped[len(p):]
            break

    # Score each dataset: prefer exact match, then longest common substring
    best_ds = None
    best_score = 0
    for ds in ds_names:
        # Exact match after stripping prefix
        if stripped == ds:
            return f"{hf_user}/{ds}"
        # Check if dataset name appears in model name
        if ds in model_name:
            score = len(ds)
            if score > best_score:
                best_score = score
                best_ds = ds
        # Check if stripped model name appears in dataset name
        elif stripped in ds:
            score = len(stripped)
            if score > best_score:
                best_score = score
                best_ds = ds
    if best_ds and best_score >= 2:
        return f"{hf_user}/{best_ds}"
    return None


@app.get("/api/hf-models")
async def list_hf_models():
    """List SmolVLA models from user's HF repo."""
    cfg = load_config()
    hf_user = cfg.get("hf_repo_name", "")
    if not hf_user:
        return {"ok": False, "error": "HF repo name not set in config"}
    try:
        from huggingface_hub import HfApi
        env = _read_env()
        token = env.get("HF_TOKEN", None)
        api = HfApi(token=token)
        models = list(api.list_models(author=hf_user))
        # Also list datasets to match model -> dataset
        datasets = list(api.list_datasets(author=hf_user))
        ds_names = [d.id.split("/")[-1] for d in datasets]

        models_dir = ROOT / "models"
        result = []
        for m in models:
            model_name = m.modelId.split("/")[-1] if "/" in m.modelId else m.modelId
            # Match model to dataset: strip common prefixes and find longest substring match
            matched_ds = _match_model_to_dataset(model_name, ds_names, hf_user)
            local_exists = (models_dir / model_name).exists()
            result.append({
                "id": m.modelId,
                "name": model_name,
                "last_modified": m.lastModified.isoformat() if m.lastModified else None,
                "tags": m.tags or [],
                "private": m.private,
                "matched_dataset": matched_ds,
                "local_exists": local_exists,
            })
        return {"ok": True, "models": result, "hf_user": hf_user}
    except ImportError:
        return {"ok": False, "error": "huggingface_hub not installed"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


model_download_state = {
    "running": False,
    "status": "",
    "error": None,
    "model_name": "",
    "log_lines": [],
}


def _run_model_download(repo_id: str, model_name: str):
    """Background thread: download model from Hugging Face Hub."""
    state = model_download_state
    state["running"] = True
    state["status"] = f"Downloading {repo_id}..."
    state["error"] = None
    state["model_name"] = model_name
    state["log_lines"] = [f"Downloading {repo_id}..."]

    model_dir = ROOT / "models" / model_name
    try:
        script = f"""
import sys, os
from huggingface_hub import snapshot_download
repo_id = {repr(repo_id)}
local_dir = {repr(str(model_dir))}
print(f"Downloading {{repo_id}} to {{local_dir}}", flush=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=local_dir,
)
print("DONE", flush=True)
"""
        lerobot_python = "/opt/miniconda3/envs/lerobot/bin/python"
        if not Path(lerobot_python).exists():
            lerobot_python = "python3"

        sub_env = dict(os.environ)
        hf_token = _read_env().get("HF_TOKEN", "")
        if hf_token:
            sub_env["HF_TOKEN"] = hf_token

        proc = subprocess.Popen(
            [lerobot_python, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=sub_env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                state["status"] = line
                state["log_lines"].append(line)
        proc.wait()
        if proc.returncode != 0:
            state["error"] = f"Download failed (exit code {proc.returncode})"
            state["status"] = "Failed"
        else:
            state["status"] = "Done"
            state["error"] = None
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "Failed"
    finally:
        state["running"] = False


@app.post("/api/model-download")
async def model_download(request: Request):
    """Download a model from HuggingFace to ./models/"""
    if model_download_state["running"]:
        return {"ok": False, "error": "Download already in progress"}
    body = await request.json()
    repo_id = body.get("repo_id", "")
    model_name = body.get("model_name", "")
    if not repo_id or not model_name:
        return {"ok": False, "error": "Missing repo_id or model_name"}
    threading.Thread(target=_run_model_download, args=(repo_id, model_name), daemon=True).start()
    return {"ok": True}


@app.get("/api/model-download/status")
async def model_download_status():
    return dict(model_download_state)


@app.get("/api/hf-datasets")
async def list_hf_datasets():
    """List datasets from user's HF account."""
    cfg = load_config()
    hf_user = cfg.get("hf_repo_name", "")
    if not hf_user:
        return {"ok": False, "error": "HF repo name not set in config"}
    try:
        from huggingface_hub import HfApi
        env = _read_env()
        token = env.get("HF_TOKEN", None)
        api = HfApi(token=token)
        datasets = list(api.list_datasets(author=hf_user))
        # Check which ones already exist locally
        local_names = set()
        data_dir = ROOT / "data"
        if data_dir.exists():
            for d in data_dir.iterdir():
                if (d / "meta" / "info.json").exists():
                    local_names.add(d.name)
        result = []
        for ds in datasets:
            ds_name = ds.id.split("/")[-1] if "/" in ds.id else ds.id
            result.append({
                "id": ds.id,
                "name": ds_name,
                "last_modified": ds.lastModified.isoformat() if ds.lastModified else None,
                "private": ds.private,
                "local_exists": ds_name in local_names,
            })
        return {"ok": True, "datasets": result, "hf_user": hf_user}
    except ImportError:
        return {"ok": False, "error": "huggingface_hub not installed"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/dataset-tasks/{repo_id:path}")
async def dataset_tasks(repo_id: str):
    """Get task list for a dataset. Try local data/ first, then HF."""
    # repo_id like "phawitbinabik/V1-4tasks-augnormal5x"
    ds_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    # Try local
    tasks_path = ROOT / "data" / ds_name / "meta" / "tasks.parquet"
    if tasks_path.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(tasks_path))
            d = tbl.to_pydict()
            tasks = d.get("task", [])
            # If task is in index (LeRobot format), read from pandas
            if not tasks:
                import pandas as pd
                df = pd.read_parquet(str(tasks_path))
                tasks = df.index.tolist()
            return {"ok": True, "tasks": tasks, "source": "local"}
        except Exception:
            pass
    # Try HF
    try:
        from huggingface_hub import hf_hub_download
        env = _read_env()
        token = env.get("HF_TOKEN", None)
        import tempfile
        tmp = hf_hub_download(repo_id=repo_id, filename="meta/tasks.parquet",
                              repo_type="dataset", token=token)
        import pandas as pd
        df = pd.read_parquet(tmp)
        # LeRobot format: task strings are index
        if "task" in df.columns:
            tasks = df["task"].tolist()
        else:
            tasks = df.index.tolist()
        return {"ok": True, "tasks": tasks, "source": "hf"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/datasets/{name}/video-thumbnail")
async def video_thumbnail(name: str, cam: str = "observation.images.top", chunk: int = 0, file: int = 0, t: float = 0.0):
    """Extract a single frame from a video at time t and return as JPEG."""
    import subprocess
    vid_path = ROOT / "data" / name / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    if not vid_path.exists():
        return Response(status_code=404)
    try:
        cmd = [
            "ffmpeg", "-ss", str(t), "-i", str(vid_path),
            "-frames:v", "1", "-f", "image2", "-vcodec", "mjpeg",
            "-q:v", "3", "pipe:1"
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        if proc.returncode != 0 or not proc.stdout:
            return Response(status_code=500)
        return Response(content=proc.stdout, media_type="image/jpeg")
    except Exception:
        return Response(status_code=500)


@app.get("/api/datasets/{name}/aug-grid")
async def aug_grid_data(name: str):
    """Get augmentation grid data: groups of source episodes with their augmented copies."""
    aug_path = ROOT / "data" / name / "meta" / "augmentation.json"
    if not aug_path.exists():
        return {"ok": False, "error": "No augmentation info"}
    try:
        aug = json.loads(aug_path.read_text())
        episodes = aug.get("episodes", [])
        source_ds = aug.get("source_dataset", "")
        # Group by source_episode
        groups = {}
        for ep in episodes:
            src = ep.get("source_episode", 0)
            if src not in groups:
                groups[src] = {"original": None, "augmented": []}
            if ep.get("is_original", False):
                groups[src]["original"] = ep
            else:
                groups[src]["augmented"].append(ep)
        # Build response: list of groups sorted by source_episode
        result = []
        for src_ep in sorted(groups.keys()):
            g = groups[src_ep]
            result.append({
                "source_episode": src_ep,
                "original": g["original"],
                "augmented": g["augmented"],
            })
        return {"ok": True, "source_dataset": source_ds, "groups": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
            try:
                tbl = pq.read_table(str(pf))
            except Exception:
                continue  # skip corrupted files
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


@app.post("/api/datasets/{name}/delete-episodes")
async def delete_episodes(name: str, request: Request):
    """Delete selected episodes from a LeRobot dataset and re-index."""
    body = await request.json()
    episodes_to_delete = set(body.get("episodes", []))
    if not episodes_to_delete:
        return {"ok": False, "error": "No episodes specified"}

    dataset_dir = ROOT / "data" / name
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        info = json.loads(info_path.read_text())
        original_total = info.get("total_episodes", 0)

        # --- 1. Filter episode metadata parquet ---
        ep_dir = dataset_dir / "meta" / "episodes"
        all_ep_rows = []
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    if d["episode_index"][i] not in episodes_to_delete:
                        all_ep_rows.append({k: v[i] for k, v in d.items()})

        # Build old→new episode index mapping
        remaining_old_indices = sorted(set(r["episode_index"] for r in all_ep_rows))
        ep_remap = {old: new for new, old in enumerate(remaining_old_indices)}

        # Re-index episode metadata
        for row in all_ep_rows:
            row["episode_index"] = ep_remap[row["episode_index"]]

        # --- 2. Filter and re-index data parquet ---
        data_dir = dataset_dir / "data"
        all_data_rows = []
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    if d["episode_index"][i] not in episodes_to_delete:
                        row = {}
                        for k, v in d.items():
                            val = v[i]
                            if hasattr(val, "tolist"):
                                val = val.tolist()
                            row[k] = val
                        all_data_rows.append(row)

        # Re-index data rows
        new_total_frames = len(all_data_rows)
        for idx, row in enumerate(all_data_rows):
            row["episode_index"] = ep_remap[row["episode_index"]]
            row["index"] = idx

        # Recalculate dataset_from_index / dataset_to_index for episodes
        ep_frame_ranges = {}
        for row in all_data_rows:
            ep = row["episode_index"]
            gidx = row["index"]
            if ep not in ep_frame_ranges:
                ep_frame_ranges[ep] = [gidx, gidx]
            else:
                ep_frame_ranges[ep][0] = min(ep_frame_ranges[ep][0], gidx)
                ep_frame_ranges[ep][1] = max(ep_frame_ranges[ep][1], gidx)
        for row in all_ep_rows:
            ep = row["episode_index"]
            if ep in ep_frame_ranges:
                row["dataset_from_index"] = ep_frame_ranges[ep][0]
                row["dataset_to_index"] = ep_frame_ranges[ep][1] + 1

        # --- 3. Write filtered episode metadata ---
        # Remove old files
        if ep_dir.exists():
            import shutil
            shutil.rmtree(ep_dir)
        ep_out = ep_dir / "chunk-000"
        ep_out.mkdir(parents=True, exist_ok=True)
        if all_ep_rows:
            cols = {k: [r[k] for r in all_ep_rows] for k in all_ep_rows[0]}
            pq.write_table(pa.table(cols), str(ep_out / "file-000.parquet"))

        # --- 4. Write filtered data ---
        if data_dir.exists():
            import shutil
            shutil.rmtree(data_dir)
        data_out = data_dir / "chunk-000"
        data_out.mkdir(parents=True, exist_ok=True)
        if all_data_rows:
            cols = {k: [r[k] for r in all_data_rows] for k in all_data_rows[0]}
            pq.write_table(pa.table(cols), str(data_out / "file-000.parquet"))

        # --- 5. Update info.json ---
        new_total_episodes = len(remaining_old_indices)
        info["total_episodes"] = new_total_episodes
        info["total_frames"] = new_total_frames
        info["splits"] = {"train": f"0:{new_total_episodes}"}
        info_path.write_text(json.dumps(info, indent=4))

        # --- 6. Delete stats.json (will be regenerated on next train) ---
        stats_path = dataset_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats_path.unlink()

        # Note: video files are left as-is — timestamps in episode metadata
        # still point to the correct segments within the shared mp4 files.

        deleted_count = original_total - new_total_episodes
        print(f"[Dataset] Deleted {deleted_count} episode(s) from {name}, {new_total_episodes} remaining")
        return {
            "ok": True,
            "deleted": deleted_count,
            "remaining_episodes": new_total_episodes,
            "remaining_frames": new_total_frames,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def _detect_segments(frames_action, fps):
    """Detect pick/place segments from gripper data (port of JS vizDetectSegments).

    Returns list of dicts: {start, end, phase} where phase is 'idle', 'pick', or 'place'.
    start/end are frame indices.
    """
    import numpy as np
    n = len(frames_action)
    if n < 5:
        return [{"start": 0, "end": n - 1, "phase": "pick"}]

    GI = 5  # gripper index

    def smooth(arr, hw):
        out = np.empty(len(arr))
        for i in range(len(arr)):
            lo, hi = max(0, i - hw), min(len(arr) - 1, i + hw)
            out[i] = np.mean(arr[lo:hi + 1])
        return out

    gripper = np.array([a[GI] if len(a) > GI else 0.0 for a in frames_action])
    arm_vel = np.zeros(n)
    for i in range(1, n):
        v = 0.0
        for j in range(6):
            a0 = frames_action[i - 1][j] if j < len(frames_action[i - 1]) else 0.0
            a1 = frames_action[i][j] if j < len(frames_action[i]) else 0.0
            v += abs(a1 - a0)
        arm_vel[i] = v

    sg = smooth(gripper, 3)
    sv = smooth(arm_vel, 5)

    # Velocity threshold for idle detection
    sv_sorted = np.sort(sv)
    vel_thresh = max(sv_sorted[int(n * 0.85)] * 0.12, 0.3)

    # Active range
    active_start, active_end = 0, n - 1
    for i in range(n):
        if sv[i] > vel_thresh:
            active_start = i
            break
    for i in range(n - 1, -1, -1):
        if sv[i] > vel_thresh:
            active_end = i
            break

    # Gripper peaks
    rest = sg[0]
    dist = np.abs(sg - rest)
    max_dist = np.max(dist)
    closed_thresh = max_dist * 0.25

    p1 = active_start
    for i in range(active_start, active_end + 1):
        if dist[i] > dist[p1]:
            p1 = i
    min_sep = int(n * 0.15)
    p2 = n - 1 if p1 < n / 2 else 0
    for i in range(n):
        if abs(i - p1) >= min_sep and dist[i] > dist[p2]:
            p2 = i
    if p1 > p2:
        p1, p2 = p2, p1

    # Valley between peaks → split in half
    valley_start, valley_end = -1, -1
    for i in range(p1, p2 + 1):
        if dist[i] <= closed_thresh:
            if valley_start < 0:
                valley_start = i
            valley_end = i
    if valley_start >= 0:
        split_frame = (valley_start + valley_end) // 2
    else:
        split_frame = (p1 + p2) // 2

    segs = []
    if active_start > 0:
        segs.append({"start": 0, "end": active_start - 1, "phase": "idle"})
    segs.append({"start": active_start, "end": split_frame, "phase": "pick"})
    segs.append({"start": split_frame + 1, "end": active_end, "phase": "place"})
    if active_end < n - 1:
        segs.append({"start": active_end + 1, "end": n - 1, "phase": "idle"})
    return segs


def _parse_object_from_task(task):
    """Extract object name from task description."""
    import re
    # "pick up the pink bow to the green bowl"
    m = re.search(r"pick\s+up\s+the\s+(.+?)\s+to\s+the\s+", task, re.I)
    if m:
        return m.group(1)
    # "Pick up the object and place it in the bin."
    m = re.search(r"pick\s+up\s+the\s+(.+?)\s+and\s+place\s+", task, re.I)
    if m:
        return m.group(1)
    # "pick up the pink bow"
    m = re.search(r"pick\s+up\s+the\s+(.+)", task, re.I)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"(?:move|put|bring|grab)\s+the\s+(.+?)\s+(?:to|into|onto)\s+", task, re.I)
    if m:
        return m.group(1)
    return "object"


@app.post("/api/datasets/{name}/split")
async def split_dataset(name: str):
    """Split a dataset by pick/place segments, creating new sub-datasets."""
    import shutil

    dataset_dir = ROOT / "data" / name
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "error": "Dataset not found"}

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        info = json.loads(info_path.read_text())
        fps = info.get("fps", 30)

        # Read task
        tasks_path = dataset_dir / "meta" / "tasks.parquet"
        task = ""
        if tasks_path.exists():
            tbl = pq.read_table(str(tasks_path))
            tasks = tbl.to_pydict().get("task", [])
            if tasks:
                task = tasks[0]
        obj = _parse_object_from_task(task)
        pick_label = f"pick up the {obj}"
        place_label = f"place the {obj}"

        # Read all episode metadata
        ep_dir = dataset_dir / "meta" / "episodes"
        all_ep_meta = []
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    all_ep_meta.append({k: v[i] for k, v in d.items()})

        # Read all data frames grouped by episode
        data_dir = dataset_dir / "data"
        all_frames = {}  # ep_index -> [rows]
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    all_frames.setdefault(ep, []).append(row)

        # Sort frames within each episode by frame_index
        for ep in all_frames:
            all_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        # Load saved segments or auto-detect
        saved_segs = {}
        seg_path = dataset_dir / "meta" / "segments.json"
        if seg_path.exists():
            try:
                saved_segs = json.loads(seg_path.read_text())
            except Exception:
                pass

        ep_segments = {}
        for ep in sorted(all_frames.keys()):
            saved = saved_segs.get(str(ep))
            if saved:
                # Convert saved segments (from frontend format) to backend format
                ep_segments[ep] = [
                    {"start": s["start"], "end": s["end"], "phase": s["phase"]}
                    for s in saved
                ]
            else:
                actions = [f.get("action", [0] * 6) for f in all_frames[ep]]
                segs = _detect_segments(actions, fps)
                ep_segments[ep] = segs

        # Build sub-datasets for pick and place
        phase_configs = [
            ("pick", pick_label),
            ("place", place_label),
        ]

        created = []

        for phase, phase_label in phase_configs:
            slug = phase_label.replace(" ", "-")
            new_name = f"{name}_{slug}"
            new_dir = ROOT / "data" / new_name

            # Skip if already exists
            if new_dir.exists():
                shutil.rmtree(new_dir)

            new_dir.mkdir(parents=True)
            (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (new_dir / "data" / "chunk-000").mkdir(parents=True)

            new_data_rows = []
            new_ep_rows = []
            new_ep_idx = 0
            global_frame_idx = 0

            for ep in sorted(all_frames.keys()):
                frames = all_frames[ep]
                segs = ep_segments[ep]
                seg = next((s for s in segs if s["phase"] == phase), None)
                if not seg:
                    continue

                ep_meta = next((m for m in all_ep_meta if m["episode_index"] == ep), None)
                start, end = seg["start"], seg["end"]
                seg_frames = frames[start:end + 1]
                seg_length = len(seg_frames)

                if seg_length == 0:
                    continue

                # Re-index data frames
                ep_global_start = global_frame_idx
                for fi, row in enumerate(seg_frames):
                    new_row = dict(row)
                    new_row["episode_index"] = new_ep_idx
                    new_row["frame_index"] = fi
                    new_row["index"] = global_frame_idx
                    new_row["timestamp"] = fi / fps
                    new_row["task_index"] = 0
                    new_data_rows.append(new_row)
                    global_frame_idx += 1

                # Build episode meta
                new_ep_meta = {}
                new_ep_meta["episode_index"] = new_ep_idx
                new_ep_meta["tasks"] = [phase_label]
                new_ep_meta["length"] = seg_length
                new_ep_meta["data/chunk_index"] = 0
                new_ep_meta["data/file_index"] = 0
                new_ep_meta["dataset_from_index"] = ep_global_start
                new_ep_meta["dataset_to_index"] = global_frame_idx

                # Video timestamps: segment range within original video
                for cam in ["observation.images.top", "observation.images.wrist"]:
                    cam_key = f"videos/{cam}"
                    if ep_meta:
                        orig_from = ep_meta.get(f"{cam_key}/from_timestamp", 0.0)
                        chunk_idx = ep_meta.get(f"{cam_key}/chunk_index", 0)
                        file_idx = ep_meta.get(f"{cam_key}/file_index", 0)
                    else:
                        orig_from, chunk_idx, file_idx = 0.0, 0, 0
                    new_ep_meta[f"{cam_key}/chunk_index"] = chunk_idx
                    new_ep_meta[f"{cam_key}/file_index"] = file_idx
                    new_ep_meta[f"{cam_key}/from_timestamp"] = orig_from + start / fps
                    new_ep_meta[f"{cam_key}/to_timestamp"] = orig_from + (end + 1) / fps

                # Copy stats from original if available (omit for simplicity)
                if ep_meta:
                    for k, v in ep_meta.items():
                        if k.startswith("stats/"):
                            new_ep_meta[k] = v
                    if "meta/episodes/chunk_index" in ep_meta:
                        new_ep_meta["meta/episodes/chunk_index"] = 0
                    if "meta/episodes/file_index" in ep_meta:
                        new_ep_meta["meta/episodes/file_index"] = 0

                new_ep_rows.append(new_ep_meta)
                new_ep_idx += 1

            if not new_data_rows:
                shutil.rmtree(new_dir)
                continue

            # Write data parquet
            cols = {k: [r[k] for r in new_data_rows] for k in new_data_rows[0]}
            pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

            # Write episode meta parquet
            if new_ep_rows:
                cols = {k: [r[k] for r in new_ep_rows] for k in new_ep_rows[0]}
                pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

            # Write tasks parquet (task strings as index, matching LeRobot format)
            import pandas as pd
            tasks_df = pd.DataFrame({"task_index": [0]}, index=pd.Index([phase_label], name="task"))
            tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

            # Copy video files
            for cam in ["observation.images.top", "observation.images.wrist"]:
                src_vid_dir = dataset_dir / "videos" / cam
                dst_vid_dir = new_dir / "videos" / cam
                if src_vid_dir.exists():
                    shutil.copytree(str(src_vid_dir), str(dst_vid_dir))

            # Write info.json
            new_info = dict(info)
            new_info["total_episodes"] = new_ep_idx
            new_info["total_frames"] = len(new_data_rows)
            new_info["total_tasks"] = 1
            new_info["splits"] = {"train": f"0:{new_ep_idx}"}
            (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
            _compute_stats_json(new_data_rows, new_dir)

            created.append({
                "name": new_name,
                "label": phase_label,
                "episodes": new_ep_idx,
                "frames": len(new_data_rows),
            })
            print(f"[Dataset] Created split: {new_name} ({new_ep_idx} eps, {len(new_data_rows)} frames)")

        return {"ok": True, "datasets": created}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ══════════ Data Augmentation ══════════

augment_state = {
    "running": False,
    "progress": 0,
    "status": "",
    "error": None,
}


def _compute_stats_json(data_rows, new_dir):
    """Compute aggregate stats.json for a dataset from data rows."""
    # Keys to compute stats for
    stat_keys = ["action", "observation.state", "timestamp", "frame_index",
                 "episode_index", "index", "task_index"]
    stats = {}
    for key in stat_keys:
        vals = []
        for row in data_rows:
            v = row.get(key)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                vals.append(v)
            else:
                vals.append([v])
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        stats[key] = {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "count": [len(arr)],
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q10": np.quantile(arr, 0.10, axis=0).tolist(),
            "q50": np.quantile(arr, 0.50, axis=0).tolist(),
            "q90": np.quantile(arr, 0.90, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }
    # Image stats: use placeholder [0..255] range (actual pixel stats require decoding all frames)
    for cam in ["observation.images.top", "observation.images.wrist"]:
        n_pixels = 240 * 320 * 3
        stats[cam] = {
            "min": [0.0] * 3,
            "max": [255.0] * 3,
            "mean": [127.5] * 3,
            "std": [73.9] * 3,
            "count": [len(data_rows)],
            "q01": [0.0] * 3,
            "q10": [25.5] * 3,
            "q50": [127.5] * 3,
            "q90": [229.5] * 3,
            "q99": [255.0] * 3,
        }
    (new_dir / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))


# Intensity level scales (1=very subtle, 5=very strong)
# Each value is a multiplier applied to the max range of each parameter
_INTENSITY_SCALE = {1: 0.25, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}


def _augp_camera(opts, intensity=3):
    """Generate random camera shift parameters."""
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if opts.get("perspective"):
        p["persp"] = np.random.uniform(-0.05 * s, 0.05 * s, (4, 2))
    if opts.get("affine"):
        p["tx"] = random.uniform(-0.04 * s, 0.04 * s)
        p["ty"] = random.uniform(-0.04 * s, 0.04 * s)
        p["shear"] = random.uniform(-0.03 * s, 0.03 * s)
        p["scale"] = random.uniform(1 - 0.05 * s, 1 + 0.05 * s)
    if opts.get("rotation"):
        p["angle"] = random.uniform(-4 * s, 4 * s)
    return p or None


def _augp_light(opts, intensity=3):
    """Generate random light/quality parameters."""
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if opts.get("brightness"):
        p["brightness"] = random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("contrast"):
        p["contrast"] = random.uniform(1 - 0.25 * s, 1 + 0.25 * s)
    if opts.get("saturation"):
        p["saturation"] = random.uniform(1 - 0.35 * s, 1 + 0.35 * s)
    if opts.get("color_jitter"):
        h = int(8 * s)
        sv = int(15 * s)
        p["hue"] = random.randint(-h, h)
        p["sat_off"] = random.randint(-sv, sv)
        p["val_off"] = random.randint(-sv, sv)
    if opts.get("shadow"):
        p["shadow_dir"] = random.choice(["left", "right", "top", "bottom"])
        p["shadow_a"] = random.uniform(0.15 * s, 0.4 * s)
    if opts.get("noise"):
        p["noise_s"] = random.uniform(4 * s, 12 * s)
    if opts.get("blur"):
        p["blur_k"] = random.choice([3, 5] if s <= 1.0 else [3, 5, 7])
        p["blur_s"] = random.uniform(0.4 * s, 1.2 * s)
    return p or None


def _augp_robot(opts, n_frames, fps, intensity=3):
    """Generate random robot state noise parameters."""
    s = _INTENSITY_SCALE.get(intensity, 1.0)
    p = {}
    if opts.get("random_start") and n_frames > 30:
        p["trim"] = random.randint(1, max(1, int(n_frames * 0.12 * s)))
    if opts.get("initial_noise"):
        p["joint_offset"] = [random.gauss(0, 1.5 * s) for _ in range(5)] + [0.0]
    if opts.get("trajectory_jitter"):
        p["jitter_s"] = random.uniform(0.3 * s, 1.0 * s)
    p["fps"] = fps
    return p or None


# --- Language augmentation ---

_PICK_SYNS = ["pick up", "grab", "lift", "take", "grasp"]
_PLACE_SYNS = ["place", "put", "set down", "drop", "lay"]
_PREP_SYNS = {"to": ["to", "into", "onto", "in", "on"],
              "from": ["from", "out of", "off"]}


def _aug_task_text(task, opts):
    """Generate language variation of task text."""
    if not opts or not opts.get("enabled"):
        return task
    result = task
    do_syn = opts.get("synonym") or opts.get("paraphrase")
    if do_syn:
        # Replace verb
        for verb, syns in [("pick up", _PICK_SYNS), ("place", _PLACE_SYNS)]:
            if verb in result.lower():
                rep = random.choice(syns)
                result = re.sub(re.escape(verb), rep, result, count=1, flags=re.I)
        # Replace preposition
        for prep, syns in _PREP_SYNS.items():
            pat = rf"\b{prep}\b"
            if re.search(pat, result, re.I):
                result = re.sub(pat, random.choice(syns), result, count=1, flags=re.I)
    if opts.get("typo") and random.random() < 0.35:
        words = result.split()
        if len(words) > 2:
            idx = random.randint(1, len(words) - 1)
            w = list(words[idx])
            if len(w) > 2:
                pos = random.randint(0, len(w) - 2)
                op = random.choice(["swap", "del", "dup"])
                if op == "swap":
                    w[pos], w[pos + 1] = w[pos + 1], w[pos]
                elif op == "del":
                    w.pop(pos)
                else:
                    w.insert(pos, w[pos])
                words[idx] = "".join(w)
            result = " ".join(words)
    return result


# --- Frame augmentation ---

def _aug_frame(frame, cam_p, light_p):
    """Apply camera + light augmentation to one frame."""
    import cv2
    h, w = frame.shape[:2]
    f = frame

    if cam_p:
        if "persp" in cam_p:
            pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2 = (pts1 + cam_p["persp"] * [w, h]).astype(np.float32)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            f = cv2.warpPerspective(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if "tx" in cam_p:
            M = np.float32([[cam_p.get("scale", 1), cam_p.get("shear", 0), cam_p["tx"] * w],
                            [0, cam_p.get("scale", 1), cam_p.get("ty", 0) * h]])
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if "angle" in cam_p:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), cam_p["angle"], 1.0)
            f = cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if light_p:
        ff = f.astype(np.float32)
        if "brightness" in light_p:
            ff *= light_p["brightness"]
        if "contrast" in light_p:
            mean = ff.mean(axis=(0, 1), keepdims=True)
            ff = (ff - mean) * light_p["contrast"] + mean
        ff = np.clip(ff, 0, 255).astype(np.uint8)
        if "saturation" in light_p:
            hsv = cv2.cvtColor(ff, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= light_p["saturation"]
            ff = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "hue" in light_p:
            hsv = cv2.cvtColor(ff, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + light_p["hue"]) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] + light_p["sat_off"], 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + light_p["val_off"], 0, 255)
            ff = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if "shadow_a" in light_p:
            hh, ww = ff.shape[:2]
            a = light_p["shadow_a"]
            d = light_p["shadow_dir"]
            if d in ("left", "right"):
                g = np.linspace(1 - a, 1, ww) if d == "left" else np.linspace(1, 1 - a, ww)
                shadow = np.tile(g, (hh, 1))
            else:
                g = np.linspace(1 - a, 1, hh) if d == "top" else np.linspace(1, 1 - a, hh)
                shadow = np.tile(g.reshape(-1, 1), (1, ww))
            ff = (ff.astype(np.float32) * shadow[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
        if "noise_s" in light_p:
            noise = np.random.normal(0, light_p["noise_s"], ff.shape)
            ff = np.clip(ff.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if "blur_k" in light_p:
            ff = cv2.GaussianBlur(ff, (light_p["blur_k"], light_p["blur_k"]), light_p["blur_s"])
        f = ff
    return f


# --- Robot state augmentation ---

def _aug_episode_data(rows, robot_p):
    """Augment robot state/action data for one episode."""
    if not robot_p:
        return rows
    rows = [dict(r) for r in rows]

    # Random start trim
    trim = robot_p.get("trim", 0)
    if trim > 0 and trim < len(rows):
        rows = rows[trim:]
    fps = robot_p.get("fps", 30)

    offset = robot_p.get("joint_offset", None)
    jitter = robot_p.get("jitter_s", 0)

    for i, r in enumerate(rows):
        r["frame_index"] = i
        r["timestamp"] = i / fps
        action = list(r.get("action", []))
        state = list(r.get("observation.state", []))
        for j in range(min(len(action), 5)):  # skip gripper j=5
            if offset:
                action[j] += offset[j]
                state[j] += offset[j]
            if jitter > 0:
                jit = random.gauss(0, jitter)
                action[j] += jit
                state[j] += jit
        r["action"] = action
        r["observation.state"] = state
    return rows


# --- Video augmentation ---

def _aug_process_video(src_path, dst_path, from_ts, to_ts, fps, cam_p, light_p, vid_w, vid_h):
    """Read video segment via ffmpeg, augment frames, write to new file."""
    duration = to_ts - from_ts
    if duration <= 0 or not Path(src_path).exists():
        return 0.0, 0.0, 0

    # Read frames from source using ffmpeg pipe (handles av1 and all codecs)
    read_cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", str(from_ts), "-i", str(src_path),
        "-t", str(duration),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}",
        "pipe:1",
    ]
    # Write frames to output using ffmpeg pipe (h264 for broad compatibility)
    write_cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{vid_w}x{vid_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        str(dst_path),
    ]

    try:
        reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE)
        writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE)

        frame_size = vid_w * vid_h * 3
        count = 0
        while True:
            raw = reader.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(vid_h, vid_w, 3)
            if cam_p or light_p:
                frame = _aug_frame(frame, cam_p, light_p)
            writer.stdin.write(frame.tobytes())
            count += 1

        reader.stdout.close()
        writer.stdin.close()
        reader.wait()
        writer.wait()

        new_to = count / fps if fps > 0 else 0.0
        return 0.0, new_to, count
    except Exception as e:
        print(f"[Augment] Video processing error: {e}")
        return 0.0, 0.0, 0


# --- Main augmentation runner ---

def _run_augmentation(src_name, new_name, target_episodes, techniques, intensity=3):
    """Background thread: create augmented dataset."""
    import shutil
    import cv2

    state = augment_state
    state["running"] = True
    state["progress"] = 0
    state["status"] = "Loading source dataset..."
    state["error"] = None

    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        src_dir = ROOT / "data" / src_name
        new_dir = ROOT / "data" / new_name
        info = json.loads((src_dir / "meta" / "info.json").read_text())
        fps = info.get("fps", 30)
        vid_w = info.get("features", {}).get("observation.images.top", {}).get("shape", [240, 320, 3])[1]
        vid_h = info.get("features", {}).get("observation.images.top", {}).get("shape", [240, 320, 3])[0]

        # Read tasks
        src_tasks = []
        tp = src_dir / "meta" / "tasks.parquet"
        if tp.exists():
            try:
                src_tasks = pq.read_table(str(tp)).to_pydict().get("task", [])
            except Exception:
                pass

        # Read episode metadata
        ep_meta_list = []
        ep_dir = src_dir / "meta" / "episodes"
        if ep_dir.exists():
            for pf in sorted(ep_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep_meta_list.append({k: v[i] for k, v in d.items()})
        ep_meta_list.sort(key=lambda r: r.get("episode_index", 0))

        # Read all data frames by episode
        src_frames = {}
        data_dir = src_dir / "data"
        if data_dir.exists():
            for pf in sorted(data_dir.rglob("*.parquet")):
                try:
                    tbl = pq.read_table(str(pf))
                except Exception:
                    continue
                d = tbl.to_pydict()
                for i in range(len(d.get("episode_index", []))):
                    ep = d["episode_index"][i]
                    row = {}
                    for k, v in d.items():
                        val = v[i]
                        if hasattr(val, "tolist"):
                            val = val.tolist()
                        row[k] = val
                    src_frames.setdefault(ep, []).append(row)
        for ep in src_frames:
            src_frames[ep].sort(key=lambda r: r.get("frame_index", 0))

        n_src_eps = len(src_frames)

        # Distribute target_episodes evenly across source episodes
        # Each source ep gets at least 1 copy (original) + augmented copies
        sorted_eps = sorted(src_frames.keys())
        copies_per_ep = {}  # ep -> total copies (including original)
        base = target_episodes // n_src_eps
        remainder = target_episodes % n_src_eps
        for i, ep in enumerate(sorted_eps):
            copies_per_ep[ep] = base + (1 if i < remainder else 0)

        total_steps = target_episodes
        current_step = 0

        # Create output dirs
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (new_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in ["observation.images.top", "observation.images.wrist"]:
            (new_dir / "videos" / cam / "chunk-000").mkdir(parents=True, exist_ok=True)

        combined_data = []
        combined_ep = []
        aug_episode_info = []  # per-episode augmentation details
        all_new_tasks = list(src_tasks)
        task_set = set(src_tasks)
        new_ep_idx = 0
        global_frame_idx = 0

        cam_opts = techniques.get("camera_shift", {})
        light_opts = techniques.get("light_quality", {})
        robot_opts = techniques.get("robot_noise", {})
        lang_opts = techniques.get("language", {})

        for old_ep in sorted_eps:
            n_copies = copies_per_ep[old_ep]
            for copy_idx in range(n_copies):
                is_original = (copy_idx == 0)
                frames = src_frames[old_ep]
                ep_meta = next((m for m in ep_meta_list if m["episode_index"] == old_ep), None)

                state["status"] = f"Episode {new_ep_idx + 1}/{target_episodes} (src ep {old_ep}, copy {copy_idx + 1}/{n_copies})"
                current_step += 1
                state["progress"] = int(current_step / total_steps * 100)

                # Generate augmentation params per episode (all random each time)
                cam_p = None if is_original else (_augp_camera(cam_opts, intensity) if cam_opts.get("enabled") else None)
                light_p = None if is_original else (_augp_light(light_opts, intensity) if light_opts.get("enabled") else None)
                robot_p = None if is_original else (
                    _augp_robot(robot_opts, len(frames), fps, intensity) if robot_opts.get("enabled") else None
                )

                # Augment data
                aug_rows = _aug_episode_data(frames, robot_p) if not is_original else [dict(r) for r in frames]

                # Language augmentation — per-episode variation
                if is_original:
                    ep_task_list = ep_meta["tasks"] if ep_meta and "tasks" in ep_meta else src_tasks[:1]
                else:
                    orig_tasks = ep_meta["tasks"] if ep_meta and "tasks" in ep_meta else src_tasks[:1]
                    ep_task_list = [_aug_task_text(t, lang_opts) for t in orig_tasks]
                # Track new tasks
                for t in ep_task_list:
                    if t not in task_set:
                        task_set.add(t)
                        all_new_tasks.append(t)
                task_to_idx = {t: i for i, t in enumerate(all_new_tasks)}

                ep_global_start = global_frame_idx
                for fi, row in enumerate(aug_rows):
                    nr = dict(row)
                    nr["episode_index"] = new_ep_idx
                    nr["frame_index"] = fi
                    nr["index"] = global_frame_idx
                    nr["timestamp"] = fi / fps
                    # Remap task_index
                    old_ti = row.get("task_index", 0)
                    if old_ti < len(ep_task_list):
                        nr["task_index"] = task_to_idx.get(ep_task_list[old_ti], 0)
                    elif ep_task_list:
                        nr["task_index"] = task_to_idx.get(ep_task_list[0], 0)
                    combined_data.append(nr)
                    global_frame_idx += 1

                # Process videos
                new_meta = {"episode_index": new_ep_idx}
                new_meta["tasks"] = ep_task_list
                new_meta["length"] = len(aug_rows)
                new_meta["data/chunk_index"] = 0
                new_meta["data/file_index"] = 0
                new_meta["dataset_from_index"] = ep_global_start
                new_meta["dataset_to_index"] = global_frame_idx

                for cam in ["observation.images.top", "observation.images.wrist"]:
                    cam_key = f"videos/{cam}"
                    orig_chunk = ep_meta.get(f"{cam_key}/chunk_index", 0) if ep_meta else 0
                    orig_file = ep_meta.get(f"{cam_key}/file_index", 0) if ep_meta else 0
                    orig_from = ep_meta.get(f"{cam_key}/from_timestamp", 0.0) if ep_meta else 0.0
                    orig_to = ep_meta.get(f"{cam_key}/to_timestamp", 0.0) if ep_meta else 0.0
                    src_vid = src_dir / "videos" / cam / f"chunk-{orig_chunk:03d}" / f"file-{orig_file:03d}.mp4"

                    dst_vid = new_dir / "videos" / cam / "chunk-000" / f"file-{new_ep_idx:03d}.mp4"
                    new_meta[f"{cam_key}/chunk_index"] = 0
                    new_meta[f"{cam_key}/file_index"] = new_ep_idx

                    if is_original and src_vid.exists():
                        # Copy original video segment (no augmentation)
                        new_from, new_to, _ = _aug_process_video(
                            src_vid, dst_vid, orig_from, orig_to, fps, None, None, vid_w, vid_h
                        )
                    elif src_vid.exists() and (cam_p or light_p):
                        new_from, new_to, _ = _aug_process_video(
                            src_vid, dst_vid, orig_from, orig_to, fps, cam_p, light_p, vid_w, vid_h
                        )
                    elif src_vid.exists():
                        # No image augmentation, just copy segment
                        new_from, new_to, _ = _aug_process_video(
                            src_vid, dst_vid, orig_from, orig_to, fps, None, None, vid_w, vid_h
                        )
                    else:
                        new_from, new_to = 0.0, 0.0

                    # Apply random_start trim to video timestamps
                    trim = robot_p.get("trim", 0) if robot_p else 0
                    if trim > 0:
                        new_from += trim / fps

                    new_meta[f"{cam_key}/from_timestamp"] = new_from
                    new_meta[f"{cam_key}/to_timestamp"] = new_to

                # Copy stats from original
                if ep_meta:
                    for k, v in ep_meta.items():
                        if k.startswith("stats/"):
                            new_meta[k] = v
                new_meta["meta/episodes/chunk_index"] = 0
                new_meta["meta/episodes/file_index"] = 0

                # Track augmentation details per episode
                ep_aug_info = {
                    "episode_index": new_ep_idx,
                    "copy_index": copy_idx,
                    "source_episode": old_ep,
                    "is_original": is_original,
                }
                if not is_original:
                    params_detail = {}
                    if cam_p:
                        cd = {}
                        if "angle" in cam_p: cd["rotation"] = round(cam_p["angle"], 1)
                        if "persp" in cam_p: cd["perspective"] = True
                        if "tx" in cam_p: cd["translate"] = [round(cam_p["tx"], 3), round(cam_p.get("ty", 0), 3)]
                        if "scale" in cam_p: cd["scale"] = round(cam_p["scale"], 2)
                        if "shear" in cam_p: cd["shear"] = round(cam_p["shear"], 3)
                        params_detail["camera"] = cd
                    if light_p:
                        ld = {}
                        if "brightness" in light_p: ld["brightness"] = round(light_p["brightness"], 2)
                        if "contrast" in light_p: ld["contrast"] = round(light_p["contrast"], 2)
                        if "saturation" in light_p: ld["saturation"] = round(light_p["saturation"], 2)
                        if "hue" in light_p: ld["color_jitter"] = True
                        if "shadow_a" in light_p: ld["shadow"] = f"{light_p['shadow_dir']} {round(light_p['shadow_a'], 2)}"
                        if "noise_s" in light_p: ld["noise_sigma"] = round(light_p["noise_s"], 1)
                        if "blur_k" in light_p: ld["blur_kernel"] = light_p["blur_k"]
                        params_detail["light"] = ld
                    if robot_p:
                        rd = {}
                        if "trim" in robot_p: rd["trim_frames"] = robot_p["trim"]
                        if "joint_offset" in robot_p: rd["initial_noise"] = [round(v, 2) for v in robot_p["joint_offset"]]
                        if "jitter_s" in robot_p: rd["jitter_sigma"] = round(robot_p["jitter_s"], 2)
                        params_detail["robot"] = rd
                    if lang_opts.get("enabled") and ep_task_list:
                        params_detail["language"] = {"augmented_task": ep_task_list[0]}
                    ep_aug_info["params"] = params_detail
                # Source video reference for original comparison
                if ep_meta:
                    ep_aug_info["source_video"] = {}
                    for sv_cam in ["observation.images.top", "observation.images.wrist"]:
                        sv_key = f"videos/{sv_cam}"
                        sv_label = sv_cam.split(".")[-1]
                        ep_aug_info["source_video"][sv_label] = {
                            "chunk_index": ep_meta.get(f"{sv_key}/chunk_index", 0),
                            "file_index": ep_meta.get(f"{sv_key}/file_index", 0),
                            "from_timestamp": ep_meta.get(f"{sv_key}/from_timestamp", 0.0),
                            "to_timestamp": ep_meta.get(f"{sv_key}/to_timestamp", 0.0),
                        }
                aug_episode_info.append(ep_aug_info)

                combined_ep.append(new_meta)
                new_ep_idx += 1

        # Write output
        state["status"] = "Writing dataset..."
        state["progress"] = 95

        # Data parquet
        cols = {k: [r[k] for r in combined_data] for k in combined_data[0]}
        pq.write_table(pa.table(cols), str(new_dir / "data" / "chunk-000" / "file-000.parquet"))

        # Episode meta parquet
        cols = {k: [r[k] for r in combined_ep] for k in combined_ep[0]}
        pq.write_table(pa.table(cols), str(new_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"))

        # Tasks parquet (task strings as index, matching LeRobot format)
        import pandas as pd
        tasks_df = pd.DataFrame({"task_index": list(range(len(all_new_tasks)))}, index=pd.Index(all_new_tasks, name="task"))
        tasks_df.to_parquet(str(new_dir / "meta" / "tasks.parquet"))

        # Info
        new_info = dict(info)
        new_info["total_episodes"] = new_ep_idx
        new_info["total_frames"] = len(combined_data)
        new_info["total_tasks"] = len(all_new_tasks)
        new_info["splits"] = {"train": f"0:{new_ep_idx}"}
        # Update video codec to h264 (re-encoded with libx264 via ffmpeg)
        for feat_key in ["observation.images.top", "observation.images.wrist"]:
            feat = new_info.get("features", {}).get(feat_key, {})
            if "info" in feat:
                feat["info"]["video.codec"] = "h264"
        (new_dir / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

        # Compute aggregate stats.json
        state["status"] = "Computing stats..."
        _compute_stats_json(combined_data, new_dir)

        # Save augmentation metadata
        aug_meta = {
            "source_dataset": src_name,
            "target_episodes": target_episodes,
            "intensity": intensity,
            "techniques": techniques,
            "episodes": aug_episode_info,
        }
        (new_dir / "meta" / "augmentation.json").write_text(json.dumps(aug_meta, indent=2))

        state["progress"] = 100
        state["status"] = f"Done! {new_ep_idx} episodes, {len(combined_data)} frames"
        print(f"[Augment] Created {new_name}: {new_ep_idx} eps, {len(combined_data)} frames")

    except Exception as e:
        import traceback
        traceback.print_exc()
        state["error"] = str(e)
        state["status"] = f"Error: {e}"
    finally:
        state["running"] = False


@app.post("/api/datasets/{name}/augment")
async def start_augment(name: str, request: Request):
    """Start data augmentation in background."""
    if augment_state["running"]:
        return {"ok": False, "error": "Augmentation already running"}
    body = await request.json()
    new_name = body.get("name", "").strip().replace(" ", "-").replace("/", "-")
    target_episodes = body.get("target_episodes")
    intensity = body.get("intensity", 3)
    techniques = body.get("techniques", {})

    if not new_name:
        return {"ok": False, "error": "Enter a dataset name"}
    src_dir = ROOT / "data" / name
    if not (src_dir / "meta" / "info.json").exists():
        return {"ok": False, "error": "Source dataset not found"}
    if (ROOT / "data" / new_name).exists():
        return {"ok": False, "error": f"Dataset '{new_name}' already exists"}
    if not target_episodes or target_episodes < 1:
        return {"ok": False, "error": "Invalid target episodes"}
    if intensity not in (1, 2, 3, 4, 5):
        intensity = 3

    threading.Thread(
        target=_run_augmentation,
        args=(name, new_name, target_episodes, techniques, intensity),
        daemon=True,
    ).start()
    return {"ok": True, "message": "Augmentation started"}


@app.get("/api/augment/status")
async def augment_status():
    return {
        "running": augment_state["running"],
        "progress": augment_state["progress"],
        "status": augment_state["status"],
        "error": augment_state["error"],
    }


# ══════════ Data Collection ══════════

datacollect_state = {
    "process": None,
    "running": False,
    "started_at": None,
    "log_lines": [],
    "log_lock": threading.Lock(),
    "completed": False,  # True when process exits naturally (all episodes done)
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

    # 3. Cameras — capture frames and hstack for preview
    import cv2
    cams = robot_cfg.get("cameras", {})
    cam_frames = []
    for cam_name, cam_cfg in cams.items():
        idx = cam_cfg.get("index", 0)
        cap = cv2.VideoCapture(idx)
        cam_ok = False
        frame = None
        if cap.isOpened():
            ret, frame = cap.read()
            cam_ok = ret
        cap.release()
        checks.append({"name": f"Camera: {cam_name}", "detail": f"index {idx}", "ok": cam_ok,
                        "error": "" if cam_ok else f"Cannot read from camera index {idx}"})
        if cam_ok and frame is not None:
            # Draw camera name label
            label = cam_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.9, 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(frame, (5, 5), (tw + 20, th + 20), (0, 0, 0), -1)
            cv2.putText(frame, label, (12, th + 12), font, scale, (255, 255, 255), thickness)
            cam_frames.append(frame)

    # Build hstacked camera preview
    camera_preview = None
    if cam_frames:
        target_h = max(f.shape[0] for f in cam_frames)
        resized = []
        for f in cam_frames:
            if f.shape[0] != target_h:
                ratio = target_h / f.shape[0]
                f = cv2.resize(f, (int(f.shape[1] * ratio), target_h))
            resized.append(f)
        stacked = np.hstack(resized)
        _, buf = cv2.imencode(".jpg", stacked, [cv2.IMWRITE_JPEG_QUALITY, 85])
        camera_preview = base64.b64encode(buf.tobytes()).decode()

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
    result = {"ok": all_ok, "checks": checks}
    if camera_preview:
        result["camera_preview"] = camera_preview
    return result


def _build_datacollect_cmd() -> list[str]:
    """Build the lerobot-record command from current config (without --resume)."""
    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    dc_cfg = cfg.get("data_collection", {})
    cams = robot_cfg.get("cameras", {})
    dc_cams = dc_cfg.get("cameras", {})

    cam_parts = []
    for cam_name, cam_c in cams.items():
        dc_cam = dc_cams.get(cam_name, {})
        cam_fps = dc_cam.get("fps", 30)
        cam_w = dc_cam.get("width", 320)
        cam_h = dc_cam.get("height", 240)
        fourcc = dc_cam.get("fourcc", "MJPG")
        part = (f"{cam_name}: {{type: opencv, index_or_path: {cam_c.get('index', 0)}, "
                f"width: {cam_w}, height: {cam_h}, fps: {cam_fps}")
        if fourcc:
            part += f", fourcc: '{fourcc}'"
        part += "}"
        cam_parts.append(part)
    cameras_str = "{" + ", ".join(cam_parts) + "}"
    fps = dc_cams.get("top", {}).get("fps", 30)

    return [
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


@app.get("/api/datacollect/command")
async def datacollect_command():
    """Return the lerobot-record command that would be run."""
    cmd = _build_datacollect_cmd()
    # Add --resume if applicable
    cfg = load_config()
    dc_cfg = cfg.get("data_collection", {})
    dataset_root = Path(dc_cfg.get("dataset_root", "./data/dataset"))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()
    if dc_cfg.get("resume", False) and dataset_exists:
        cmd.append("--resume=true")
    # Shell-quote args that contain spaces or braces for copy-paste
    import shlex
    quoted = [shlex.quote(arg) if any(c in arg for c in ' {}') else arg for arg in cmd]
    return {"ok": True, "command": " \\\n  ".join(quoted)}


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

    cmd = _build_datacollect_cmd()

    import shutil
    cfg = load_config()
    dc_cfg = cfg.get("data_collection", {})
    dataset_root = Path(dc_cfg.get("dataset_root", "./data/dataset"))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    dataset_exists = dataset_root.exists() and (dataset_root / "meta" / "info.json").exists()

    if dc_cfg.get("resume", False) and dataset_exists:
        cmd.append("--resume=true")
    elif not dc_cfg.get("resume", False):
        # Remove existing dataset directory to avoid FileExistsError
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
            print(f"[DataCollect] Removed existing dataset dir: {dataset_root}")

    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        datacollect_state["process"] = p
        datacollect_state["running"] = True
        datacollect_state["started_at"] = time.time()
        datacollect_state["log_lines"] = []
        datacollect_state["completed"] = False
        # Start log reader thread
        threading.Thread(target=_datacollect_reader, args=(p, datacollect_state), daemon=True).start()
        print(f"[DataCollect] Started: {' '.join(cmd)}")
        return {"ok": True, "message": "Recording started"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/datacollect/stop")
async def datacollect_stop():
    """Stop lerobot-record subprocess gracefully using SIGINT."""
    if not datacollect_state["running"]:
        return {"ok": True, "message": "Not recording"}
    p = datacollect_state["process"]
    if p:
        # Use SIGINT (Ctrl+C) instead of SIGTERM to let lerobot-record
        # finalize data files properly before exiting
        import signal as sig
        p.send_signal(sig.SIGINT)
        try:
            p.wait(timeout=30)  # Give more time for data finalization
        except subprocess.TimeoutExpired:
            print("[DataCollect] SIGINT timeout, sending SIGTERM...")
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[DataCollect] SIGTERM timeout, killing...")
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
    # Check if process exited naturally (all episodes done)
    p = datacollect_state["process"]
    if p and p.poll() is not None:
        exit_code = p.returncode
        datacollect_state["running"] = False
        datacollect_state["process"] = None
        datacollect_state["started_at"] = None
        datacollect_state["completed"] = True
        print(f"[DataCollect] Process exited with code {exit_code}")
    with datacollect_state["log_lock"]:
        recent_logs = list(datacollect_state["log_lines"][-50:])
    return {
        "running": datacollect_state["running"],
        "elapsed": round(elapsed, 1),
        "logs": recent_logs,
        "completed": datacollect_state["completed"],
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


# ── Evaluate ──

EVAL_DIR = ROOT / "data" / "eval_sessions"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CONFIG_FILE = EVAL_DIR / "eval_config.json"


@app.get("/api/eval/config")
async def eval_get_config():
    if EVAL_CONFIG_FILE.exists():
        try:
            return {"ok": True, "config": json.loads(EVAL_CONFIG_FILE.read_text())}
        except Exception:
            pass
    return {"ok": True, "config": {}}


@app.post("/api/eval/config")
async def eval_save_config(request: Request):
    body = await request.json()
    EVAL_CONFIG_FILE.write_text(json.dumps(body, indent=2))
    return {"ok": True}

eval_state = {"running": False, "process": None, "log_lines": [], "session_id": None}

EVAL_RESET_POS_FILE = EVAL_DIR / "reset_position.json"


def _eval_load_reset_position():
    """Load saved reset position or return None."""
    if EVAL_RESET_POS_FILE.exists():
        try:
            return json.loads(EVAL_RESET_POS_FILE.read_text())
        except Exception:
            pass
    return None


def _eval_move_to_reset():
    """Connect robot, move to saved reset position, disconnect. Returns (ok, error)."""
    pos = _eval_load_reset_position()
    if not pos:
        return True, None  # No reset position saved, skip

    try:
        connect_robot()
        # Move gradually to reset position
        for _ in range(30):  # ~1 second at 30 iterations
            robot_send_positions(pos)
            time.sleep(0.033)
        disconnect_robot()
        return True, None
    except Exception as e:
        try:
            disconnect_robot()
        except Exception:
            pass
        return False, str(e)


def _eval_reader(proc, state):
    """Background thread to read eval subprocess output."""
    for line in iter(proc.stdout.readline, b''):
        text = line.decode("utf-8", errors="replace").rstrip()
        state["log_lines"].append(text)
        if len(state["log_lines"]) > 2000:
            state["log_lines"] = state["log_lines"][-1500:]
    proc.wait()
    state["running"] = False


@app.get("/api/eval/reset-position")
async def eval_get_reset_position():
    """Get saved reset position."""
    pos = _eval_load_reset_position()
    return {"ok": True, "position": pos}


@app.post("/api/eval/reset-position")
async def eval_save_reset_position(request: Request):
    """Save current robot position as the eval reset position."""
    body = await request.json()
    pos = body.get("position")
    if pos:
        # Directly provided position
        EVAL_RESET_POS_FILE.write_text(json.dumps(pos, indent=2))
        return {"ok": True, "position": pos}
    # Read from robot — auto-connect if needed
    auto_connected = False
    try:
        if not robot_state["connected"]:
            connect_robot()
            auto_connected = True
        pos = robot_get_positions()
        EVAL_RESET_POS_FILE.write_text(json.dumps(pos, indent=2))
        if auto_connected:
            disconnect_robot()
        return {"ok": True, "position": pos}
    except Exception as e:
        if auto_connected:
            try:
                disconnect_robot()
            except Exception:
                pass
        return {"ok": False, "error": str(e)}


@app.delete("/api/eval/reset-position")
async def eval_delete_reset_position():
    """Delete saved reset position."""
    if EVAL_RESET_POS_FILE.exists():
        EVAL_RESET_POS_FILE.unlink()
    return {"ok": True}


@app.post("/api/eval/move-to-reset")
async def eval_move_to_reset():
    """Move robot to saved reset position."""
    pos = _eval_load_reset_position()
    if not pos:
        return {"ok": False, "error": "No reset position saved"}
    ok, err = _eval_move_to_reset()
    if ok:
        return {"ok": True}
    return {"ok": False, "error": err}


@app.post("/api/eval/fakecam-preview")
async def eval_fakecam_preview(request: Request):
    """Capture one frame from each camera and return original + augmented as base64 JPEG."""
    import cv2
    import base64

    body = await request.json()
    params = body.get("params", {})

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    cameras = robot_cfg.get("cameras", {})

    cam_p = _build_cam_params(params)
    light_p = _build_light_params(params)

    result = []
    for cam_name, cam_cfg in cameras.items():
        idx = cam_cfg.get("index", 0)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            result.append({"name": cam_name, "error": f"Cannot open camera {idx}"})
            continue
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            result.append({"name": cam_name, "error": f"Cannot read from camera {idx}"})
            continue
        # Resize for preview (keep small for storage in session JSON)
        preview_w = 320
        h, w = frame.shape[:2]
        preview_h = int(h * preview_w / w)
        small = cv2.resize(frame, (preview_w, preview_h))
        # Encode original
        _, orig_buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        orig_b64 = base64.b64encode(orig_buf).decode()
        # Apply augmentation (if any params set)
        if cam_p or light_p:
            aug = _aug_frame(small, cam_p, light_p)
        else:
            aug = small
        _, aug_buf = cv2.imencode(".jpg", aug, [cv2.IMWRITE_JPEG_QUALITY, 70])
        aug_b64 = base64.b64encode(aug_buf).decode()
        result.append({"name": cam_name, "original": orig_b64, "augmented": aug_b64})

    return {"ok": True, "cameras": result}


@app.post("/api/eval/start")
async def eval_start(request: Request):
    """Start lerobot-rollout process for evaluation."""
    if eval_state["running"]:
        return {"ok": False, "error": "Already running"}
    body = await request.json()
    model_id = body.get("model_id", "")
    model_name = body.get("model_name", "")
    task = body.get("task", "")
    fps = int(body.get("fps", 30))
    cam_width = int(body.get("width", 320))
    cam_height = int(body.get("height", 240))
    fourcc = body.get("fourcc", "").strip()
    use_fakecam = body.get("use_fakecam", False)

    if not model_name:
        return {"ok": False, "error": "No model selected"}

    # Stop teleop/robot if running
    if teleop_state["running"]:
        await teleop_stop()
    if robot_state["connected"]:
        disconnect_robot()
        time.sleep(0.2)

    # Move robot to reset position before starting rollout
    reset_pos = _eval_load_reset_position()
    if reset_pos:
        ok, err = _eval_move_to_reset()
        if not ok:
            return {"ok": False, "error": f"Failed to move to reset position: {err}"}

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    cams = robot_cfg.get("cameras", {})

    # Build cameras config string
    cam_parts = []
    fourcc_part = f", fourcc: {fourcc}" if fourcc else ""
    for cam_name, cam_c in cams.items():
        cam_parts.append(
            f"{cam_name}: {{type: opencv, index_or_path: {cam_c.get('index', 0)}, "
            f"width: {cam_width}, height: {cam_height}, fps: {fps}{fourcc_part}}}"
        )
    cameras_str = "{" + ", ".join(cam_parts) + "}"

    model_dir = f"./models/{model_name}"

    cmd = [
        "/opt/miniconda3/envs/lerobot/bin/lerobot-rollout",
        f"--robot.type=so101_follower",
        f"--robot.port={robot_cfg.get('port', '')}",
        f"--robot.id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--robot.cameras={cameras_str}",
        f"--policy.path={model_dir}",
        f"--fps={fps}",
        f"--display_data=false",
    ]
    if task:
        cmd.append(f"--task={task}")

    if use_fakecam:
        cmd = ["python", "fakecam_inject.py", "--params-file", "fakecam_params.json", "--"] + cmd

    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        eval_state["process"] = p
        eval_state["running"] = True
        eval_state["log_lines"] = []
        threading.Thread(target=_eval_reader, args=(p, eval_state), daemon=True).start()
        return {"ok": True, "pid": p.pid, "cmd": " ".join(cmd)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/eval/stop")
async def eval_stop():
    """Stop the running eval process. lerobot returns robot to initial position on its own."""
    if not eval_state["running"] and not eval_state["process"]:
        return {"ok": True, "message": "Not running"}

    p = eval_state["process"]
    if p:
        import signal as sig
        try:
            os.killpg(os.getpgid(p.pid), sig.SIGINT)
        except (ProcessLookupError, OSError):
            try:
                p.send_signal(sig.SIGINT)
            except (ProcessLookupError, OSError):
                pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), sig.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    p.kill()
                except (ProcessLookupError, OSError):
                    pass
    eval_state["running"] = False
    return {"ok": True}


@app.get("/api/eval/status")
async def eval_status():
    """Get eval process status and logs."""
    return {
        "running": eval_state["running"],
        "lines": eval_state["log_lines"][-100:],
    }


@app.post("/api/eval/sessions")
async def eval_save_session(request: Request):
    """Save an evaluation session."""
    body = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return {"ok": False, "error": "No session_id"}
    path = EVAL_DIR / f"{session_id}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2))
    return {"ok": True}


@app.get("/api/eval/sessions")
async def eval_list_sessions():
    """List saved eval sessions."""
    sessions = []
    for f in sorted(EVAL_DIR.glob("*.json"), reverse=True):
        try:
            d = json.loads(f.read_text())
            sessions.append({
                "id": d.get("session_id", f.stem),
                "model": d.get("model_name", ""),
                "task": d.get("task", ""),
                "target": d.get("target_runs", 0),
                "completed": d.get("completed", 0),
                "success": d.get("success", 0),
                "fail": d.get("fail", 0),
                "created": d.get("created", ""),
            })
        except Exception:
            pass
    return {"ok": True, "sessions": sessions}


@app.get("/api/eval/sessions/{session_id}")
async def eval_get_session(session_id: str):
    """Get a specific eval session."""
    path = EVAL_DIR / f"{session_id}.json"
    if not path.exists():
        return {"ok": False, "error": "Session not found"}
    return {"ok": True, **json.loads(path.read_text())}


@app.delete("/api/eval/sessions/{session_id}")
async def eval_delete_session(session_id: str):
    """Delete an eval session."""
    path = EVAL_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()
    return {"ok": True}


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
