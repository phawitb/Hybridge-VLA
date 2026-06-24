#!/usr/bin/env python3
"""
Fake Camera Inject — Monkey-patch cv2.VideoCapture to apply augmentation
to all camera frames without modifying inference code.

Usage:
    python fakecam_inject.py -- lerobot-rollout --robot.port=... --policy.path=...

    # With custom params (JSON):
    python fakecam_inject.py --params '{"rotation":5,"brightness":0.8}' -- lerobot-rollout ...

    # Load params from Hybridge VLA server:
    python fakecam_inject.py --from-server http://localhost:8000 -- lerobot-rollout ...

    # Load params from file:
    python fakecam_inject.py --params-file fakecam_params.json -- lerobot-rollout ...
"""
import sys
import json
import os
import time
import argparse
import subprocess


def build_aug_frame():
    """Return a standalone _aug_frame function with no external dependencies."""
    import cv2
    import numpy as np

    def _aug_frame(frame, cam_p, light_p):
        h, w = frame.shape[:2]
        f = frame
        if cam_p:
            if "persp" in cam_p:
                pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                pts2 = (pts1 + cam_p["persp"] * np.array([w, h])).astype(np.float32)
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
            if "noise_s" in light_p:
                noise = np.random.normal(0, light_p["noise_s"], ff.shape)
                ff = np.clip(ff.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            if "blur_k" in light_p:
                ff = cv2.GaussianBlur(ff, (light_p["blur_k"], light_p["blur_k"]), light_p["blur_s"])
            f = ff
        return f

    return _aug_frame


def build_cam_params(p):
    cam = {}
    rotation = float(p.get("rotation", 0))
    translate_x = float(p.get("translate_x", 0))
    translate_y = float(p.get("translate_y", 0))
    scale = float(p.get("scale", 1.0))
    shear = float(p.get("shear", 0))
    if rotation:
        cam["angle"] = rotation
    if translate_x or translate_y:
        cam["tx"] = translate_x / 100
        cam["ty"] = translate_y / 100
        cam["scale"] = scale
        cam["shear"] = shear / 100
    elif scale != 1.0 or shear:
        cam["tx"] = 0
        cam["ty"] = 0
        cam["scale"] = scale
        cam["shear"] = shear / 100
    return cam if cam else None


def build_light_params(p):
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
        k = blur * 2 + 1
        lp["blur_k"] = k
        lp["blur_s"] = blur * 0.4
    return lp if lp else None


def load_params_from_server(url):
    """Fetch current fakecam params from Hybridge VLA server."""
    import urllib.request
    resp = urllib.request.urlopen(f"{url.rstrip('/')}/api/fakecam/params")
    return json.loads(resp.read().decode())


def patch_black_cameras(black_indices, width=320, height=240):
    """Monkey-patch cv2.VideoCapture so that cameras opened with indices in
    `black_indices` always return a black frame instead of real camera data."""
    import cv2
    import numpy as np

    black_indices = set(int(i) for i in black_indices)
    print(f"[FakeCam] Black-camera mode for indices: {black_indices}")

    _OriginalVideoCapture = cv2.VideoCapture

    class BlackCameraCapture(_OriginalVideoCapture):
        def __init__(self, *args, **kwargs):
            self._is_black = False
            self._black_frame = None
            if args and isinstance(args[0], int) and args[0] in black_indices:
                self._is_black = True
                self._black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                print(f"[FakeCam] Camera index {args[0]} → black frame ({width}x{height})")
                # Don't open real device — just init base with no args
                super().__init__()
            else:
                super().__init__(*args, **kwargs)

        def isOpened(self):
            if self._is_black:
                return True
            return super().isOpened()

        def read(self):
            if self._is_black:
                return True, self._black_frame.copy()
            return super().read()

        def release(self):
            if self._is_black:
                return
            return super().release()

        def set(self, propId, value):
            if self._is_black:
                return True
            return super().set(propId, value)

        def get(self, propId):
            if self._is_black:
                if propId == cv2.CAP_PROP_FRAME_WIDTH:
                    return float(width)
                if propId == cv2.CAP_PROP_FRAME_HEIGHT:
                    return float(height)
                if propId == cv2.CAP_PROP_FPS:
                    return 30.0
                return 0.0
            return super().get(propId)

    cv2.VideoCapture = BlackCameraCapture
    print("[FakeCam] cv2.VideoCapture replaced with BlackCameraCapture")


def patch_opencv(params, params_file=None, server_url=None, reload_interval=2.0):
    """Replace cv2.VideoCapture with a subclass that applies augmentation.

    cv2.VideoCapture is a C++ extension type — patching its .read() method
    at the class level is unreliable because C extensions may resolve methods
    via tp_methods slots instead of __dict__. Instead we replace the class
    itself with a Python subclass whose read() is a normal Python method.
    """
    import cv2
    import threading

    aug_frame = build_aug_frame()

    # Shared state for hot-reload
    state = {
        "cam_p": build_cam_params(params),
        "light_p": build_light_params(params),
        "file_mtime": None,
    }

    if not state["cam_p"] and not state["light_p"]:
        print("[FakeCam] No augmentation params — running without augmentation")

    print(f"[FakeCam] Augmentation active:")
    if state["cam_p"]:
        print(f"  Camera: {state['cam_p']}")
    if state["light_p"]:
        print(f"  Light:  {state['light_p']}")

    # Track file mtime for change detection
    if params_file and os.path.exists(params_file):
        state["file_mtime"] = os.path.getmtime(params_file)

    def _reload():
        """Background thread: reload params from file or server periodically."""
        while True:
            time.sleep(reload_interval)
            try:
                new_params = None
                if params_file and os.path.exists(params_file):
                    mtime = os.path.getmtime(params_file)
                    if mtime != state["file_mtime"]:
                        state["file_mtime"] = mtime
                        with open(params_file) as f:
                            new_params = json.load(f)
                        print(f"[FakeCam] Hot-reload: params updated from file")
                elif server_url:
                    new_params = load_params_from_server(server_url)

                if new_params is not None:
                    state["cam_p"] = build_cam_params(new_params)
                    state["light_p"] = build_light_params(new_params)
            except Exception:
                pass

    # Start hot-reload thread if source is available
    if params_file or server_url:
        src = params_file or server_url
        t = threading.Thread(target=_reload, daemon=True)
        t.start()
        print(f"[FakeCam] Hot-reload enabled (checking every {reload_interval}s from {src})")

    # Save the original class
    _OriginalVideoCapture = cv2.VideoCapture

    class AugmentedVideoCapture(_OriginalVideoCapture):
        """VideoCapture subclass that applies fakecam augmentation on read()."""

        def read(self):
            ret, frame = super().read()
            if ret and frame is not None:
                cam_p = state["cam_p"]
                light_p = state["light_p"]
                if cam_p or light_p:
                    frame = aug_frame(frame, cam_p, light_p)
            return ret, frame

    # Replace cv2.VideoCapture so any new instance uses our subclass
    cv2.VideoCapture = AugmentedVideoCapture
    print("[FakeCam] cv2.VideoCapture replaced with AugmentedVideoCapture")


def main():
    # Ensure prints appear immediately in piped output
    sys.stdout.reconfigure(line_buffering=True)

    # Split args at '--'
    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        our_args = argv[:split_idx]
        cmd_args = argv[split_idx + 1:]
    else:
        our_args = argv
        cmd_args = []

    parser = argparse.ArgumentParser(description="Fake Camera Inject")
    parser.add_argument("--params", type=str, help="JSON string of augmentation params")
    parser.add_argument("--params-file", type=str, help="Path to JSON file with params")
    parser.add_argument("--from-server", type=str, help="Load params from Hybridge VLA server URL")
    parser.add_argument("--save", type=str, help="Save resolved params to file and exit")
    parser.add_argument("--black-cameras", type=str, help="Comma-separated camera indices to replace with black frames")
    parser.add_argument("--black-cam-width", type=int, default=320, help="Width for black camera frames")
    parser.add_argument("--black-cam-height", type=int, default=240, help="Height for black camera frames")
    args = parser.parse_args(our_args)

    # Resolve params
    has_aug = args.from_server or args.params_file or args.params
    params = {}
    if args.from_server:
        try:
            params = load_params_from_server(args.from_server)
            print(f"[FakeCam] Loaded params from server: {args.from_server}")
        except Exception as e:
            print(f"[FakeCam] Failed to load from server: {e}")
            sys.exit(1)
    elif args.params_file:
        with open(args.params_file) as f:
            params = json.load(f)
        print(f"[FakeCam] Loaded params from file: {args.params_file}")
    elif args.params:
        params = json.loads(args.params)
    elif not args.black_cameras:
        # Try loading from default file (only required if no --black-cameras)
        default_path = os.path.join(os.path.dirname(__file__), "fakecam_params.json")
        if os.path.exists(default_path):
            with open(default_path) as f:
                params = json.load(f)
            print(f"[FakeCam] Loaded params from {default_path}")
            has_aug = True
        else:
            print("[FakeCam] No params specified. Use --params, --params-file, or --from-server")
            print("[FakeCam] Or save params from the web UI first")
            sys.exit(1)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(params, f, indent=2)
        print(f"[FakeCam] Params saved to {args.save}")
        return

    if not cmd_args:
        print("[FakeCam] No command specified after '--'")
        print("Usage: python fakecam_inject.py --params '{...}' -- lerobot-rollout ...")
        sys.exit(1)

    # Resolve reload source for hot-reload
    reload_file = None
    reload_server = None
    if args.params_file:
        reload_file = args.params_file
    elif args.from_server:
        reload_server = args.from_server
    else:
        # Check default file
        default_path = os.path.join(os.path.dirname(__file__) or '.', "fakecam_params.json")
        if os.path.exists(default_path):
            reload_file = default_path

    # Patch black cameras first (so augmentation wraps on top if needed)
    if args.black_cameras:
        black_indices = [int(i.strip()) for i in args.black_cameras.split(",")]
        patch_black_cameras(black_indices, width=args.black_cam_width, height=args.black_cam_height)

    # Patch OpenCV for augmentation (with hot-reload)
    if has_aug:
        patch_opencv(params, params_file=reload_file, server_url=reload_server)

    # Run the command in the same process by importing and executing
    # For subprocess commands like lerobot-rollout, we need to set up
    # the environment with the patched cv2
    cmd_str = cmd_args[0]
    # Extract basename for entry point lookup (handles full paths like
    # /opt/miniconda3/envs/lerobot/bin/lerobot-rollout)
    cmd_name = os.path.basename(cmd_str)

    # Check if it's a Python module/script we can run in-process
    if cmd_str.endswith(".py"):
        # Run Python script in-process (patched cv2 is inherited)
        sys.argv = cmd_args
        exec(open(cmd_str).read(), {"__name__": "__main__"})
    else:
        # For installed CLI tools (like lerobot-rollout), we need to
        # find and run them as Python entry points in-process
        # so the monkey-patch takes effect
        import importlib
        from importlib.metadata import entry_points

        # Try to find as console_scripts entry point
        func = None
        try:
            eps = entry_points()
            if hasattr(eps, "select"):
                scripts = eps.select(group="console_scripts")
            else:
                scripts = eps.get("console_scripts", [])
            for ep in scripts:
                if ep.name == cmd_name:
                    func = ep.load()
                    break
        except Exception as e:
            print(f"[FakeCam] Entry point lookup failed: {e}")

        if func:
            sys.argv = cmd_args
            print(f"[FakeCam] Running {cmd_name} in-process with patched cv2")
            func()
        else:
            # Fallback: try importing as module
            module_name = cmd_name.replace("-", "_")
            try:
                mod = importlib.import_module(module_name)
                if hasattr(mod, "main"):
                    sys.argv = cmd_args
                    print(f"[FakeCam] Running {module_name}.main() with patched cv2")
                    mod.main()
                else:
                    print(f"[FakeCam] ERROR: Cannot find entry point for '{cmd_name}'")
                    print(f"[FakeCam] Augmentation requires in-process execution. Aborting.")
                    sys.exit(1)
            except ImportError:
                print(f"[FakeCam] ERROR: Cannot import '{cmd_name}'")
                print(f"[FakeCam] Augmentation requires in-process execution. Aborting.")
                sys.exit(1)


if __name__ == "__main__":
    main()
