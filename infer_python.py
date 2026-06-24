#!/usr/bin/env python3
"""
Standalone Python inference script for SmolVLA / ACT / pi0 policies.
Uses LeRobot's Python API directly instead of lerobot-rollout.

Usage:
    python infer_python.py \
        --model-path ./models/smolvla_V1 \
        --task "pick up the pink bow to the green bowl" \
        --robot-port /dev/tty.usbmodem5B141122411 \
        --robot-id my_awesome_follower_arm \
        --cameras '{"top": 0, "wrist": 1}' \
        --fps 30 \
        --width 320 --height 240
"""

import argparse
import json
import signal
import sys
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Python VLA inference")
    p.add_argument("--model-path", required=True, help="Path to model directory")
    p.add_argument("--task", default="", help="Task instruction text")
    p.add_argument("--robot-port", required=True, help="Robot serial port")
    p.add_argument("--robot-id", default="my_awesome_follower_arm", help="Robot ID")
    p.add_argument("--cameras", default='{"top": 0, "wrist": 1}', help="JSON dict of camera_name: index")
    p.add_argument("--fps", type=int, default=30, help="Control loop FPS")
    p.add_argument("--width", type=int, default=320, help="Camera width")
    p.add_argument("--height", type=int, default=240, help="Camera height")
    p.add_argument("--chunk-size", type=int, default=0, help="Override chunk_size (0=use model default)")
    p.add_argument("--n-action-steps", type=int, default=0, help="Override n_action_steps (0=use model default)")
    p.add_argument("--device", default="", help="Device: cuda, mps, cpu (auto-detect if empty)")
    p.add_argument("--cache-language", action="store_true", help="Cache language tokens & embeddings (same task text every step)")
    return p.parse_args()


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Global flag for graceful shutdown
_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n[infer_python] Caught signal, stopping...")
    _running = False


def main():
    global _running
    args = parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    device = torch.device(args.device) if args.device else detect_device()
    print(f"[infer_python] Device: {device}")
    print(f"[infer_python] Model: {args.model_path}")
    print(f"[infer_python] Task: {args.task}")

    # --- Load model config to determine policy type ---
    import json as _json
    config_path = f"{args.model_path}/config.json"
    with open(config_path) as f:
        model_config = _json.load(f)
    policy_type = model_config.get("type", "unknown")
    print(f"[infer_python] Policy type: {policy_type}")

    # --- Load policy ---
    print("[infer_python] Loading policy...", flush=True)
    t0 = time.time()

    if policy_type == "smolvla":
        print("[infer_python] Importing SmolVLAPolicy...", flush=True)
        from lerobot.policies.smolvla import SmolVLAPolicy
        print("[infer_python] Calling from_pretrained...", flush=True)
        policy = SmolVLAPolicy.from_pretrained(args.model_path)
        print("[infer_python] from_pretrained done", flush=True)
    elif policy_type == "act":
        from lerobot.policies.act import ACTPolicy
        policy = ACTPolicy.from_pretrained(args.model_path)
    elif policy_type in ("pi0", "pi05"):
        from lerobot.policies.pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(args.model_path)
    else:
        print(f"[infer_python] ERROR: Unknown policy type '{policy_type}'")
        sys.exit(1)

    print("[infer_python] Calling policy.eval()...", flush=True)
    policy.eval()
    print(f"[infer_python] Policy loaded in {time.time() - t0:.1f}s", flush=True)

    # --- Build pre/post processors ---
    print("[infer_python] Building pre/post processors...", flush=True)
    from lerobot.policies import make_pre_post_processors
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("[infer_python] Pre/post processors ready", flush=True)

    # --- Connect robot ---
    print("[infer_python] Connecting robot...", flush=True)
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cameras_dict = json.loads(args.cameras)

    # Build camera configs for LeRobot
    from lerobot.cameras.opencv import OpenCVCameraConfig
    cam_configs = {}
    for cam_name, cam_index in cameras_dict.items():
        cam_configs[cam_name] = OpenCVCameraConfig(
            index_or_path=cam_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        print(f"[infer_python] Camera '{cam_name}' index={cam_index}", flush=True)

    robot_config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cam_configs,
    )
    print(f"[infer_python] Creating robot (port={args.robot_port})...", flush=True)
    robot = make_robot_from_config(robot_config)
    print("[infer_python] Calling robot.connect()...", flush=True)
    robot.connect()
    print("[infer_python] Robot connected", flush=True)

    # --- Build dataset features for observation frame ---
    print("[infer_python] Building dataset features...", flush=True)
    from lerobot.utils.feature_utils import hw_to_dataset_features
    print(f"[infer_python] robot.action_features = {robot.action_features}", flush=True)
    print(f"[infer_python] robot.observation_features = {robot.observation_features}", flush=True)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    print(f"[infer_python] dataset_features keys: {list(dataset_features.keys())}", flush=True)

    # --- Override chunk_size / n_action_steps if requested ---
    if args.chunk_size > 0:
        policy.config.chunk_size = args.chunk_size
        print(f"[infer_python] Override chunk_size={args.chunk_size}", flush=True)
    if args.n_action_steps > 0:
        policy.config.n_action_steps = args.n_action_steps
        print(f"[infer_python] Override n_action_steps={args.n_action_steps}", flush=True)

    # --- Determine robot type ---
    robot_type = "so101_follower"

    # --- Monkey-patch sample_actions for detailed timing ---
    if policy_type == "smolvla":
        _orig_sample_actions = policy.model.sample_actions

        def _timed_sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
            import math
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

            model = policy.model
            bsize = state.shape[0]
            dev = state.device

            if noise is None:
                actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
                noise = model.sample_noise(actions_shape, dev)

            t0 = time.time()
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            t_embed = time.time()

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )
            t_kv = time.time()

            num_steps = model.config.num_steps
            dt = -1.0 / num_steps
            x_t = noise
            for step_i in range(num_steps):
                t_val = 1.0 + step_i * dt
                time_tensor = torch.tensor(t_val, dtype=torch.float32, device=dev).expand(bsize)
                v_t = model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )
                x_t = x_t + dt * v_t
            t_denoise = time.time()

            ms = lambda a, b: f"{(b-a)*1000:.1f}ms"
            print(
                f"[infer_python] PREDICT breakdown | "
                f"embed_prefix={ms(t0,t_embed)} "
                f"kv_cache={ms(t_embed,t_kv)} "
                f"denoise_x{num_steps}={ms(t_kv,t_denoise)} "
                f"TOTAL={ms(t0,t_denoise)}",
                flush=True,
            )
            return x_t

        policy.model.sample_actions = _timed_sample_actions
        print("[infer_python] Detailed PREDICT timing ENABLED", flush=True)

    # --- Setup language cache if requested ---
    cached_lang_tokens = None
    cached_lang_mask = None
    cached_lang_embeds = None

    # Exact key names from lerobot constants
    LANG_TOKENS_KEY = "observation.language.tokens"
    LANG_MASK_KEY = "observation.language.attention_mask"

    if args.cache_language and policy_type == "smolvla":
        print("[infer_python] Language caching ENABLED", flush=True)

        # Monkey-patch embed_language_tokens on vlm_with_expert to cache embeddings
        _vlm_expert = policy.model.vlm_with_expert
        _orig_embed_lang = _vlm_expert.embed_language_tokens

        def _cached_embed_language_tokens(tokens):
            nonlocal cached_lang_embeds
            if cached_lang_embeds is not None:
                return cached_lang_embeds
            result = _orig_embed_lang(tokens)
            cached_lang_embeds = result.detach()
            print(f"[infer_python] Cached language embeddings: {result.shape}", flush=True)
            return result

        _vlm_expert.embed_language_tokens = _cached_embed_language_tokens
    elif args.cache_language:
        print(f"[infer_python] Language caching not supported for policy type '{policy_type}', ignoring", flush=True)

    # --- Inference loop ---
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    period = 1.0 / args.fps
    step = 0
    print(f"[infer_python] Starting inference loop at {args.fps} FPS", flush=True)
    print("[infer_python] Press Ctrl+C to stop", flush=True)

    try:
        while _running:
            t0 = time.time()

            # Get observation from robot
            obs = robot.get_observation()
            t_obs = time.time()

            # Build inference frame
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=args.task,
                robot_type=robot_type,
            )
            t_frame = time.time()

            # Preprocess
            obs_processed = preprocess(obs_frame)
            t_preprocess = time.time()

            # Cache language tokens after first preprocess, reuse on subsequent steps
            if args.cache_language and policy_type == "smolvla":
                if cached_lang_tokens is None:
                    if LANG_TOKENS_KEY in obs_processed:
                        cached_lang_tokens = obs_processed[LANG_TOKENS_KEY].clone()
                    if LANG_MASK_KEY in obs_processed:
                        cached_lang_mask = obs_processed[LANG_MASK_KEY].clone()
                else:
                    if LANG_TOKENS_KEY in obs_processed:
                        obs_processed[LANG_TOKENS_KEY] = cached_lang_tokens
                    if LANG_MASK_KEY in obs_processed:
                        obs_processed[LANG_MASK_KEY] = cached_lang_mask

            # Predict action
            queue_empty = len(policy._queues.get("action", [])) == 0
            with torch.inference_mode():
                action = policy.select_action(obs_processed)
            t_action = time.time()

            # Postprocess
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            t_post = time.time()

            # Send to robot
            robot.send_action(action)
            t_send = time.time()

            step += 1

            # Detailed timing every second + on predict steps
            if step <= 3 or step % args.fps == 0 or queue_empty:
                ms = lambda a, b: f"{(b-a)*1000:.1f}ms"
                predict_tag = " ** PREDICT **" if queue_empty else " (reuse)"
                print(
                    f"[infer_python] Step {step}{predict_tag} | "
                    f"obs={ms(t0,t_obs)} frame={ms(t_obs,t_frame)} preprocess={ms(t_frame,t_preprocess)} "
                    f"action={ms(t_preprocess,t_action)} postprocess={ms(t_action,t_post)} send={ms(t_post,t_send)} "
                    f"TOTAL={ms(t0,t_send)}",
                    flush=True,
                )

            # Maintain target FPS
            elapsed = t_send - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"[infer_python] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"[infer_python] Stopping after {step} steps")
        try:
            robot.disconnect()
            print("[infer_python] Robot disconnected")
        except Exception as e:
            print(f"[infer_python] Disconnect error: {e}")


if __name__ == "__main__":
    main()
