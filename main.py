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
                            st["cap"] = c
                            time.sleep(1)
                            continue
                        ret, frame = c.read()
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

teleop_state = {"process": None, "running": False}


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


@app.post("/api/teleop/start")
async def teleop_start():
    if teleop_state["running"]:
        return {"ok": True, "message": "Already running"}
    # Stop slider-mode robot first (they share the follower port)
    if robot_state["connected"]:
        disconnect_robot()

    cfg = load_config()
    robot_cfg = cfg.get("robot", {})
    teleop_cfg = cfg.get("teleop", {})
    cmd = [
        "/opt/miniconda3/envs/lerobot/bin/lerobot-teleoperate",
        f"--robot.type=so101_follower",
        f"--robot.port={robot_cfg.get('port', '/dev/tty.usbmodem5B141122411')}",
        f"--robot.id={robot_cfg.get('id', 'my_awesome_follower_arm')}",
        f"--teleop.type={teleop_cfg.get('type', 'so101_leader')}",
        f"--teleop.port={teleop_cfg.get('port', '/dev/tty.usbmodem5B140300651')}",
    ]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        teleop_state["process"] = p
        teleop_state["running"] = True
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
    teleop_state["process"] = None
    teleop_state["running"] = False
    print("[Teleop] Stopped")
    return {"ok": True, "message": "Teleop stopped"}


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
    """Click on top view → move robot to that position using interpolation from calibration points."""
    if not robot_state["connected"]:
        return {"ok": False, "error": "Robot not connected"}
    if not calib_state["homography"]:
        return {"ok": False, "error": "Not calibrated"}

    data = await request.json()
    pixel = data.get("pixel")  # [u, v]
    if not pixel:
        return {"ok": False, "error": "No pixel coordinate"}

    height_cm = data.get("height_cm", 0.0)
    joints = interpolate_joints_from_pixel(pixel, calib_state["points"], height_cm=height_cm)
    if joints is None:
        return {"ok": False, "error": "Interpolation failed — need 4 valid calibration points"}

    robot_send_positions(joints)
    return {
        "ok": True,
        "pixel": pixel,
        "target_joints": joints,
    }


# ── Calibrate JSON API (receive JSON body) ──

@app.post("/api/calibrate/save-all")
async def calibrate_save_all(request: Request):
    """Save all calibration points at once: {points: [{pixel, joints, robot_xy, robot_z}, ...]}"""
    data = await request.json()
    calib_state["points"] = data.get("points", [])
    save_calibration()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
