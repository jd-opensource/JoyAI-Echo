"""Convert an official Echo-WM safetensors checkpoint to scaled ROCm FP8 shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


WM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM_ROOT / "ltx-core" / "src"))

import ltx_core.loader  # noqa: E402, F401
from ltx_core.quantization.rocm_checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    FP8_DTYPE,
    is_fp8_attention_tensor,
    is_scaled_ffn_weight,
    quantize_weight_to_fp8,
    resolve_rocm_checkpoint,
    sha256_file,
    _validate_tensor,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tensor_spec(handle, key: str) -> dict[str, Any]:
    tensor = handle.get_slice(key)
    dtypes = {
        "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
        "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool",
    }
    name = str(tensor.get_dtype())
    if name not in dtypes:
        raise ValueError(f"unsupported source tensor dtype: {name}")
    return {"dtype": dtypes[name], "shape": list(tensor.get_shape())}


def _tensor_bytes(spec: dict[str, Any]) -> int:
    sizes = {
        "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
        "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
    }
    if spec["dtype"] not in sizes:
        raise ValueError(f"unsupported source tensor dtype: {spec['dtype']}")
    result = sizes[spec["dtype"]]
    for dimension in spec["shape"]:
        result *= dimension
    return result


def _output_tensor_bytes(key: str, spec: dict[str, Any]) -> int:
    source_bytes = _tensor_bytes(spec)
    if is_scaled_ffn_weight(key):
        elements = 1
        for dimension in spec["shape"]:
            elements *= dimension
        return elements + 4
    if is_fp8_attention_tensor(key) and spec["dtype"] == "bfloat16":
        return source_bytes // 2
    return source_bytes


def _shard_plan(schema: dict[str, dict[str, Any]], max_shard_bytes: int) -> list[list[str]]:
    if max_shard_bytes < 1:
        raise ValueError("max shard size must be positive")
    plan: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in sorted(schema):
        required = _output_tensor_bytes(key, schema[key])
        if required > max_shard_bytes:
            raise ValueError(f"tensor {key} exceeds --max-shard-bytes")
        if current and current_bytes + required > max_shard_bytes:
            plan.append(current)
            current, current_bytes = [], 0
        current.append(key)
        current_bytes += required
    if current:
        plan.append(current)
    return plan


def _source_description(source: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    with safe_open(source, framework="pt", device="cpu") as handle:
        schema = {key: _tensor_spec(handle, key) for key in handle.keys()}
        metadata = dict(handle.metadata() or {})
    if not schema:
        raise ValueError("source checkpoint has no tensors")
    for key in schema:
        if key.endswith(".weight_scale"):
            raise ValueError(f"source has unsupported preexisting FP8 scale: {key}")
        if is_scaled_ffn_weight(key) and f"{key.removesuffix('.weight')}.weight_scale" in schema:
            raise ValueError(f"source already has a conflicting FP8 scale: {key}")
    return schema, metadata


def _metadata(source_metadata: dict[str, str], source_sha256: str) -> dict[str, str]:
    conflicting = {"quantization", "source_sha256", "rsi_quantization", "rsi_source_sha256"}
    if conflicting & set(source_metadata):
        raise ValueError("source metadata already claims a checkpoint quantization format")
    return {**source_metadata, "quantization": CHECKPOINT_FORMAT, "source_sha256": source_sha256}


def _expected_manifest(
    source: Path,
    source_sha256: str,
    schema: dict[str, dict[str, Any]],
    source_metadata: dict[str, str],
    plan: list[list[str]],
    max_shard_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": CHECKPOINT_FORMAT,
        "strategy": CHECKPOINT_FORMAT,
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "source_schema": schema,
        "source_metadata": source_metadata,
        "max_shard_bytes": max_shard_bytes,
        "shard_plan": plan,
        "source_tensors": len(schema),
    }


def _validate_resume_description(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"resume manifest does not match current {key}")


def _verify_existing_shard(
    target: Path, record: dict[str, Any], keys: list[str],
    schema: dict[str, Any], metadata: dict[str, str],
) -> None:
    if not target.is_file() or target.stat().st_size != record["bytes"]:
        raise ValueError(f"resume shard is missing or has a different size: {target.name}")
    if sha256_file(target) != record["sha256"]:
        raise ValueError(f"resume shard hash mismatch: {target.name}")
    expected_keys = set(keys)
    expected_keys.update(
        key.removesuffix(".weight") + ".weight_scale" for key in keys if is_scaled_ffn_weight(key)
    )
    with safe_open(target, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        if handle.metadata() != metadata or actual_keys != expected_keys:
            raise ValueError(f"resume shard metadata or tensor plan mismatch: {target.name}")
        for key in actual_keys:
            _validate_tensor(key, handle.get_tensor(key), actual_keys, schema)


def convert_checkpoint(
    source: Path | str,
    output_dir: Path | str,
    *,
    device: str = "cpu",
    max_shard_bytes: int = 1024**3,
    expected_source_sha256: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Convert one official safetensors file without materializing a full model."""
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu or cuda")
    source_sha256 = sha256_file(source)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256.lower():
        raise ValueError(f"source SHA-256 mismatch: expected {expected_source_sha256}, got {source_sha256}")
    schema, source_metadata = _source_description(source)
    plan = _shard_plan(schema, max_shard_bytes)
    expected = _expected_manifest(
        source, source_sha256, schema, source_metadata, plan, max_shard_bytes
    )
    output_metadata = _metadata(source_metadata, source_sha256)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    progress_path = output_dir / ".conversion-progress.json"
    if resume and manifest_path.exists():
        completed = _load_completed_manifest(output_dir)
        _validate_resume_description(completed, expected)
        resolve_rocm_checkpoint(output_dir, verify_hashes=True)
        return completed
    if not resume and (
        manifest_path.exists() or progress_path.exists() or list(output_dir.glob("*.safetensors"))
    ):
        raise ValueError("output directory is not empty; use --resume after verifying its manifest")

    weight_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    pending = None
    shard_count = len(plan)
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _validate_resume_description(progress, expected)
        records = progress["shard_records"]
        pending = progress.get("pending_shard")
        if not isinstance(records, list) or len(records) > len(plan):
            raise ValueError("invalid conversion progress")
    elif list(output_dir.glob("*.safetensors")):
        raise ValueError("existing shards have no manifest or progress record and cannot be resumed safely")
    recorded_names = {record["path"] for record in records}
    pending_name = None
    if pending is not None:
        pending_name = f"model-{len(records) + 1:05d}-of-{shard_count:05d}.safetensors"
        if (
            not isinstance(pending, dict) or len(records) >= shard_count
            or pending.get("path") != pending_name
        ):
            raise ValueError("invalid pending shard in conversion progress")
    present_names = {path.name for path in output_dir.glob("*.safetensors")}
    allowed_names = recorded_names | ({pending_name} if pending_name is not None else set())
    if present_names - allowed_names or recorded_names - present_names:
        raise ValueError("unrecorded shards in output; cannot prove their completed conversion")
    if pending is not None:
        target = output_dir / pending_name
        temporary = target.with_name(f".{target.name}.tmp")
        recovering = target if target.exists() else temporary
        _verify_existing_shard(recovering, pending, plan[len(records)], schema, output_metadata)
        if recovering == temporary:
            os.replace(temporary, target)
        records.append(pending)
    _atomic_json(progress_path, {**expected, "shard_records": records})
    for index, keys in enumerate(plan, start=1):
        filename = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        target = output_dir / filename
        if index <= len(records):
            record = records[index - 1]
            if record["path"] != filename:
                raise ValueError("resume shard names do not match the conversion plan")
            _verify_existing_shard(target, record, keys, schema, output_metadata)
            for key in keys:
                weight_map[key] = filename
                if is_scaled_ffn_weight(key):
                    weight_map[key.removesuffix(".weight") + ".weight_scale"] = filename
            continue
        if target.exists():
            raise ValueError(f"refusing to overwrite existing shard: {target}")
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source, framework="pt", device="cpu") as handle:
            for key in keys:
                value = handle.get_tensor(key)
                if is_scaled_ffn_weight(key):
                    quantized, scale = quantize_weight_to_fp8(value, device=target_device)
                    tensors[key] = quantized.contiguous()
                    tensors[f"{key.removesuffix('.weight')}.weight_scale"] = scale
                elif is_fp8_attention_tensor(key) and value.dtype == torch.bfloat16:
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(f"selected FP8 tensor is not finite: {key}")
                    tensors[key] = value.to(target_device).to(FP8_DTYPE).cpu().contiguous()
                else:
                    tensors[key] = value.contiguous()
                del value
        for key, value in tensors.items():
            _validate_tensor(key, value, set(tensors), schema)
        del value
        temporary = target.with_name(f".{target.name}.tmp")
        save_file(tensors, temporary, metadata=output_metadata)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        record = {
            "path": filename, "bytes": temporary.stat().st_size,
            "sha256": sha256_file(temporary),
        }
        # Journal the verified payload before publishing its final shard name.
        _atomic_json(progress_path, {**expected, "shard_records": records, "pending_shard": record})
        os.replace(temporary, target)
        for key in tensors:
            weight_map[key] = filename
        records.append(record)
        _atomic_json(progress_path, {**expected, "shard_records": records})
        del tensors

    _atomic_json(
        output_dir / "model.safetensors.index.json",
        {
            "metadata": {"total_size": sum(_output_tensor_bytes(key, schema[key]) for key in schema)},
            "weight_map": weight_map,
        },
    )
    manifest = {**expected, "shard_records": records}
    _atomic_json(output_dir / "manifest.json", manifest)
    resolve_rocm_checkpoint(output_dir, verify_hashes=True)
    progress_path.unlink()
    return manifest


def _load_completed_manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-shard-bytes", type=int, default=1024**3)
    parser.add_argument("--source-sha256")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = convert_checkpoint(
        args.source, args.output_dir, device=args.device, max_shard_bytes=args.max_shard_bytes,
        expected_source_sha256=args.source_sha256, resume=args.resume,
    )
    print(json.dumps({
        "format": manifest["format"], "source_sha256": manifest["source_sha256"],
        "shards": len(manifest["shard_records"]), "output_dir": str(args.output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
