#!/usr/bin/env python3
"""Run a checked-in WM case with causal 4-step inference."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "examples" / "wm_causal_cases"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "echo-wm-flash.safetensors"


def _case_dirs() -> list[Path]:
    return sorted(path for path in CASES.iterdir() if path.is_dir() and (path / "case.json").is_file())


def _load_case(case_dir: Path) -> dict:
    data = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    missing = [key for key in ("prompt", "action") if key not in data]
    if missing:
        raise ValueError(f"{case_dir}/case.json missing fields: {', '.join(missing)}")
    data["image"] = data.get("image", "input.jpg")
    image = case_dir / data["image"]
    if not image.is_file():
        raise FileNotFoundError(f"Case image not found: {image}")
    return data


def _num_frames_from_action(action: str) -> int:
    """Infer output length from Action DSL durations, including the first frame."""
    segments = "".join(action.replace("，", ",").split()).split(",")
    try:
        durations = [int(segment.rsplit("-", 1)[1]) for segment in segments]
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot infer frame count from action {action!r}") from error
    if not durations or any(duration <= 0 for duration in durations):
        raise ValueError(f"Action durations must be positive: {action!r}")
    return 1 + sum(durations)


def build_command(args: argparse.Namespace) -> list[str]:
    case_dir = (args.case if args.case.is_absolute() else ROOT / args.case).resolve()
    data = _load_case(case_dir)
    image = case_dir / data["image"]
    num_frames = args.num_frames if args.num_frames is not None else _num_frames_from_action(data["action"])
    command = [
        sys.executable, str(ROOT / "inference_wm_causal.py"),
        "--image", str(image), "--prompt", data["prompt"],
        "--action-str", data["action"],
        "--num-frames", str(num_frames),
        "--fov-deg", str(data.get("fov_deg", 70.0)),
        "--output", str(args.output_dir / case_dir.name / "result.mp4"),
        "--checkpoint", str(args.checkpoint),
    ]
    if args.gemma_path:
        command += ["--gemma-path", str(args.gemma_path)]
    for flag, value in (
        ("--width", args.width),
        ("--height", args.height),
        ("--video-local-attn-size", args.video_local_attn_size),
        ("--video-sink-size", args.video_sink_size),
        ("--video-chunk-size", args.video_chunk_size),
        ("--quantization", args.quantization),
    ):
        if value is not None:
            command += [flag, str(value)]
    command += ["--action-overlay" if args.action_overlay else "--no-action-overlay"]
    if args.rocm_fp8_fast_accum:
        command += ["--rocm-fp8-fast-accum"]
    if "seed" in data:
        command += ["--seed", str(data["seed"])]
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gemma-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "wm_cases_causal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--quantization", choices=("none", "fp8-cast", "rocm-fp8", "rocm-fp8-reference"),
    )
    parser.add_argument("--rocm-fp8-fast-accum", action="store_true")
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
    parser.add_argument(
        "--action-overlay", action=argparse.BooleanOptionalAction, default=True,
        help="Also write the HUD copy (default: enabled).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.list:
        for case_dir in _case_dirs():
            print(f"{case_dir.name}\t{_load_case(case_dir).get('description', 'causal I2V case')}")
        return
    if args.case is None:
        raise SystemExit("--case is required unless --list is used")
    command = build_command(args)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
