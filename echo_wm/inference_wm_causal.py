"""Public 4-step autoregressive pure-UCPE image-to-video inference entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
for package in ("ltx-core/src", "ltx-causal/src", "ltx-pipelines/src"):
    sys.path.insert(0, str(ROOT / package))

DEFAULT_CONFIG = ROOT / "configs" / "inference_wm_causal.yaml"


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _override(value, default):
    return default if value is None else value


def _quantization_policy(name: str, device: torch.device, *, fast_accum: bool = False):
    if fast_accum and name != "rocm-fp8":
        raise ValueError("--rocm-fp8-fast-accum requires --quantization rocm-fp8")
    if name == "none":
        return None
    from ltx_core.quantization import QuantizationPolicy

    if name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if name == "rocm-fp8":
        from ltx_core.quantization.rocm_fp8 import require_rocm_fp8

        require_rocm_fp8(device, probe=True, fast_accum=fast_accum)
        return QuantizationPolicy.rocm_fp8_prequantized(fast_accum=fast_accum)
    if name == "rocm-fp8-reference":
        return QuantizationPolicy.rocm_fp8_prequantized(reference=True)
    raise ValueError(f"unknown quantization policy: {name}")


def _auto_fov(image: Path, model: str, python_bin: str, width: int, height: int) -> float:
    raw = subprocess.run(
        [python_bin, str(ROOT / "helpers" / "moge_fov.py"), "--image", str(image), "--model", model,
         "--target-width", str(width), "--target-height", str(height)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(json.loads(raw)["fov_x_deg"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--action-str", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gemma-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/echo_wm_causal.mp4"))
    parser.add_argument("--auto-fov", action="store_true")
    parser.add_argument("--moge-model", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--moge-python", default=sys.executable)
    parser.add_argument("--fov-deg", type=float)
    parser.add_argument("--translation-speed", type=float)
    parser.add_argument("--rotation-speed-deg", type=float)
    parser.add_argument("--pitch-limit-deg", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--timesteps", type=int, nargs="+")
    parser.add_argument(
        "--video-local-attn-size", "--video_local_attn_size",
        dest="video_local_attn_size", type=int,
    )
    parser.add_argument(
        "--video-sink-size", "--video_sink_size", dest="video_sink_size", type=int,
    )
    parser.add_argument(
        "--video-chunk-size", "--video_chunk_size", dest="video_chunk_size", type=int,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument(
        "--quantization", choices=("none", "fp8-cast", "rocm-fp8", "rocm-fp8-reference"),
        default="none", help="Explicit transformer policy; the original BF16 path remains the default.",
    )
    parser.add_argument(
        "--rocm-fp8-fast-accum", action="store_true",
        help="Opt in to faster, lower-precision accumulation for native ROCm FP8.",
    )
    parser.add_argument(
        "--action-overlay", action=argparse.BooleanOptionalAction, default=True,
        help="Write a second MP4 with a Genie-style action HUD overlay "
             "(default: enabled; disable with --no-action-overlay).",
    )
    return parser.parse_args(argv)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    # Keep --help usable before optional media/CUDA dependencies are installed.
    from ltx_causal import CausalCacheConfig
    from ltx_core.model.video_vae.tiling import TilingConfig
    from ltx_core.model.video_vae.video_vae import get_video_chunks_number
    from ltx_pipelines.causal_ti2vid import CausalTI2VidPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.media_io import encode_video

    from helpers.action_camera import (
        DEFAULT_PITCH_LIMIT_DEG,
        DEFAULT_ROTATION_SPEED_DEG,
        DEFAULT_TRANSLATION_SPEED,
    )
    from helpers.action_condition import action_config, build_action_trajectory, build_causal_action_condition
    from helpers.action_overlay import overlay_genie_on_video

    if not args.image.is_file():
        raise FileNotFoundError(f"I2V first-frame image not found: {args.image}")
    cfg = _load_config(args.config)
    model_cfg, video_cfg, causal_cfg, action_cfg = (
        cfg.get("model", {}), cfg.get("video", {}), cfg.get("causal", {}), cfg.get("action", {})
    )
    checkpoint = args.checkpoint or ROOT / model_cfg.get("checkpoint", "checkpoints/echo-wm-flash.safetensors")
    gemma_path = args.gemma_path or ROOT / model_cfg["gemma_path"]
    width = _override(args.width, video_cfg.get("width", 1280))
    height = _override(args.height, video_cfg.get("height", 704))
    num_frames = _override(args.num_frames, video_cfg.get("num_frames", 241))
    fps = _override(args.fps, video_cfg.get("fps", 24.0))
    seed = _override(args.seed, video_cfg.get("seed", 42))
    fov = _override(args.fov_deg, action_cfg.get("fov_deg", 70.0))
    if args.auto_fov:
        fov = _auto_fov(args.image, args.moge_model, args.moge_python, width, height)
    cache = CausalCacheConfig(
        video_local_attn_size=_override(
            args.video_local_attn_size, causal_cfg.get("video_local_attn_size", 19)
        ),
        video_sink_size=_override(
            args.video_sink_size, causal_cfg.get("video_sink_size", 7)
        ),
        video_chunk_size=_override(
            args.video_chunk_size, causal_cfg.get("video_chunk_size", 3)
        ),
    )
    cache.validate()
    timesteps = tuple(_override(args.timesteps, causal_cfg.get("timesteps", [1000, 750, 500, 250])))
    action_kwargs = dict(
        action=args.action_str, num_frames=num_frames, width=width, height=height,
        translation_speed=_override(args.translation_speed, action_cfg.get("translation_speed", DEFAULT_TRANSLATION_SPEED)),
        rotation_speed_deg=_override(args.rotation_speed_deg, action_cfg.get("rotation_speed_deg", DEFAULT_ROTATION_SPEED_DEG)),
        pitch_limit_deg=_override(args.pitch_limit_deg, action_cfg.get("pitch_limit_deg", DEFAULT_PITCH_LIMIT_DEG)),
        fov_deg=fov, fps=fps,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantization = _quantization_policy(
        args.quantization, device, fast_accum=args.rocm_fp8_fast_accum,
    )
    action = build_causal_action_condition(device=device, **action_kwargs)
    trajectory = build_action_trajectory(
        args.action_str, num_frames=num_frames,
        translation_speed=action_kwargs["translation_speed"],
        rotation_speed_deg=action_kwargs["rotation_speed_deg"],
        pitch_limit_deg=action_kwargs["pitch_limit_deg"], fps=fps,
    ) if args.action_overlay else None

    pipeline = CausalTI2VidPipeline(
        checkpoint_path=str(checkpoint), gemma_root=str(gemma_path), device=device,
        quantization=quantization,
        action_config=action_config(width, height), cache_config=cache,
    )
    video, audio = pipeline(
        prompt=args.prompt, seed=seed, height=height, width=width,
        num_frames=num_frames, frame_rate=fps,
        images=[ImageConditioningInput(str(args.image), 0, 1.0)],
        action_cond=action, timesteps=timesteps,
        video_tiling_config=TilingConfig.default(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video, fps=int(fps), audio=None if args.no_audio else audio,
        output_path=str(args.output),
        video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
    )
    overlay_output = None
    if trajectory is not None:
        overlay_output = args.output.with_name(f"{args.output.stem}_action{args.output.suffix}")
        overlay_genie_on_video(args.output, trajectory, output_path=overlay_output)
    metadata = {
        "mode": "causal_4_step", "prompt": args.prompt, "action": args.action_str,
        "quantization": args.quantization, "rocm_fp8_fast_accum": args.rocm_fp8_fast_accum,
        "checkpoint": str(checkpoint), "timesteps": list(timesteps), "seed": seed,
        "width": width, "height": height, "num_frames": num_frames, "fps": fps,
        "video_local_attn_size": cache.video_local_attn_size,
        "video_sink_size": cache.video_sink_size,
        "video_chunk_size": cache.video_chunk_size,
        "audio_local_attn_size": cache.audio_local_attn_size,
        "audio_sink_size": cache.audio_sink_size, "cfg": False,
        "cache_policy": "bounded sink-plus-FIFO", "camera_policy": "bounded anchor translation",
        "action_overlay": bool(args.action_overlay),
        "overlay_output": overlay_output.name if overlay_output else None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    if overlay_output:
        print(f"Saved {overlay_output}")


if __name__ == "__main__":
    main()
