"""Validation and path resolution for converted ROCm FP8 checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError, safe_open


CHECKPOINT_FORMAT = "amd-fp8-mixed-v1"
FP8_DTYPE = torch.float8_e4m3fn
_FFN_WEIGHT_SUFFIXES = (
    "ff.net.0.proj.weight",
    "ff.net.2.weight",
    "audio_ff.net.0.proj.weight",
    "audio_ff.net.2.weight",
)
_FFN_BIAS_SUFFIXES = (
    "ff.net.0.proj.bias",
    "ff.net.2.bias",
    "audio_ff.net.0.proj.bias",
    "audio_ff.net.2.bias",
)
_ATTENTION_SUFFIXES = (
    "to_q.weight",
    "to_q.bias",
    "to_k.weight",
    "to_k.bias",
    "to_v.weight",
    "to_v.bias",
    "to_out.0.weight",
    "to_out.0.bias",
)
_SHA256_LENGTH = 64

__all__ = [
    "CHECKPOINT_FORMAT",
    "FP8_DTYPE",
    "is_transformer_block",
    "is_scaled_ffn_weight",
    "is_ffn_bias",
    "is_fp8_attention_tensor",
    "is_fp8_converted_tensor",
    "quantize_weight_to_fp8",
    "sha256_file",
    "resolve_rocm_checkpoint",
]


def is_transformer_block(key: str) -> bool:
    return key.startswith("model.diffusion_model.transformer_blocks.")


def is_scaled_ffn_weight(key: str) -> bool:
    return is_transformer_block(key) and key.endswith(_FFN_WEIGHT_SUFFIXES)


def is_ffn_bias(key: str) -> bool:
    return is_transformer_block(key) and key.endswith(_FFN_BIAS_SUFFIXES)


def is_fp8_attention_tensor(key: str) -> bool:
    return is_transformer_block(key) and key.endswith(_ATTENTION_SUFFIXES)


def is_fp8_converted_tensor(key: str) -> bool:
    return is_scaled_ffn_weight(key) or is_fp8_attention_tensor(key)


def quantize_weight_to_fp8(
    weight: torch.Tensor, *, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3FN weight storage plus its positive FP32 dequantization scale."""
    if (
        weight.ndim != 2 or not weight.numel() or not weight.is_floating_point()
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("selected FP8 weight must contain only finite floating-point values")
    target = torch.device(device) if device is not None else weight.device
    value = weight.to(device=target, dtype=torch.float32)
    limit = torch.finfo(FP8_DTYPE).max
    scale = value.abs().amax().clamp_min(1e-12).div(limit)
    quantized = value.div(scale).clamp(-limit, limit).to(FP8_DTYPE)
    return quantized.cpu(), scale.float().cpu()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error
    return value.lower()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _safe_shard(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"unsafe checkpoint shard path: {name!r}")
    candidate = (root / name).resolve()
    if candidate.parent != root:
        raise ValueError(f"checkpoint shard escapes its directory: {name!r}")
    return candidate


def _format_from_metadata(metadata: dict[str, str]) -> str | None:
    values = [
        metadata[key]
        for key in ("quantization", "rsi_quantization")
        if key in metadata
    ]
    if len(set(values)) > 1:
        raise ValueError("conflicting checkpoint quantization metadata")
    return values[0] if values else None


def _source_from_metadata(metadata: dict[str, str]) -> str | None:
    values = [
        metadata[key]
        for key in ("source_sha256", "rsi_source_sha256")
        if key in metadata
    ]
    if len(set(values)) > 1:
        raise ValueError("conflicting checkpoint source hash metadata")
    return values[0] if values else None


def _manifest_format(manifest: dict[str, Any]) -> str:
    values = [manifest[key] for key in ("format", "strategy") if key in manifest]
    if not values or not all(value == CHECKPOINT_FORMAT for value in values):
        raise ValueError(f"unsupported ROCm checkpoint format: {values!r}")
    if "schema_version" in manifest and manifest["schema_version"] != 1:
        raise ValueError(f"unsupported checkpoint schema_version: {manifest['schema_version']!r}")
    return values[0]


def _schema_spec(schema: dict[str, Any], key: str) -> tuple[str, list[int]]:
    spec = schema.get(key)
    if not isinstance(spec, dict):
        raise ValueError(f"manifest schema does not describe {key}")
    dtype, shape = spec.get("dtype"), spec.get("shape")
    if not isinstance(dtype, str) or not (
        isinstance(shape, list) and all(isinstance(item, int) and item >= 0 for item in shape)
    ):
        raise ValueError(f"invalid manifest schema for {key}")
    return dtype, shape


def _validate_tensor(
    key: str,
    value: torch.Tensor,
    keys: set[str],
    schema: dict[str, Any] | None,
) -> None:
    if key.endswith(".weight_scale"):
        weight_key = f"{key.removesuffix('.weight_scale')}.weight"
        if weight_key not in keys or not is_scaled_ffn_weight(weight_key):
            raise ValueError(f"unexpected FP8 scale tensor: {key}")
        if value.dtype != torch.float32 or value.shape != torch.Size([]):
            raise ValueError(f"{key} must be a scalar FP32 tensor")
        if not bool(torch.isfinite(value) & (value > 0)):
            raise ValueError(f"{key} must be finite and positive")
        return

    expected_dtype: str | None = None
    expected_shape: list[int] | None = None
    if schema is not None:
        expected_dtype, expected_shape = _schema_spec(schema, key)
        if list(value.shape) != expected_shape:
            raise ValueError(f"checkpoint tensor shape mismatch for {key}")

    if is_scaled_ffn_weight(key):
        if value.dtype != FP8_DTYPE or value.ndim != 2 or not value.numel():
            raise ValueError(f"{key} must use {FP8_DTYPE}")
        if f"{key.removesuffix('.weight')}.weight_scale" not in keys:
            raise ValueError(f"{key} is missing its weight_scale")
    elif is_fp8_attention_tensor(key):
        required = (
            FP8_DTYPE if schema is None or expected_dtype == "bfloat16"
            else getattr(torch, expected_dtype, None)
        )
        if value.dtype != required:
            raise ValueError(f"checkpoint tensor dtype mismatch for {key}: expected {required}")
    elif schema is not None and value.dtype != getattr(torch, expected_dtype, None):
        # Dtype spellings use torch's public attribute names, but fail closed if unknown.
        raise ValueError(f"checkpoint tensor dtype mismatch for {key}")

    if is_fp8_converted_tensor(key) and not bool(torch.isfinite(value.float()).all()):
        raise ValueError(f"converted checkpoint tensor is not finite: {key}")


def _validate_index(
    root: Path, shard_names: set[str], shard_keys: dict[str, set[str]], required: bool
) -> None:
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        if required:
            raise ValueError("converted checkpoint is missing model.safetensors.index.json")
        return
    index = _load_json(index_path, "checkpoint index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no weight_map")
    seen: set[str] = set()
    for key, name in weight_map.items():
        if not isinstance(key, str) or not isinstance(name, str) or name not in shard_names:
            raise ValueError("checkpoint index references an unknown shard")
        if key in seen or key not in shard_keys[name]:
            raise ValueError(f"checkpoint index has an invalid mapping for {key!r}")
        seen.add(key)
    actual = set().union(*shard_keys.values())
    if seen != actual:
        raise ValueError("checkpoint index does not describe every tensor")


def resolve_rocm_checkpoint(path: Path | str, *, verify_hashes: bool = True) -> tuple[str, ...]:
    """Return validated shard paths for an opt-in converted checkpoint.

    Both current metadata names and the read-only ``rsi_*`` names used by the
    legacy conversion artifact are accepted.  A legacy manifest can omit the
    standard index only when its hashed shard records and tensor headers pass.
    """
    root = Path(path).expanduser()
    if root.name == "manifest.json":
        root = root.parent
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"ROCm checkpoint must be a directory: {root}")
    manifest = _load_json(root / "manifest.json", "checkpoint manifest")
    _manifest_format(manifest)
    source_sha256 = _require_sha256(manifest.get("source_sha256"), "manifest source_sha256")
    records = manifest.get("shard_records")
    if not isinstance(records, list) or not records:
        raise ValueError("checkpoint manifest has no shard_records")
    schema = manifest.get("source_schema")
    if schema is not None and (not isinstance(schema, dict) or not schema):
        raise ValueError("invalid manifest source_schema")
    source_metadata = manifest.get("source_metadata")
    if source_metadata is not None and (
        not isinstance(source_metadata, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in source_metadata.items())
    ):
        raise ValueError("invalid manifest source_metadata")
    shard_names: set[str] = set()
    shards: list[Path] = []
    shard_keys: dict[str, set[str]] = {}
    all_keys: set[str] = set()
    first_metadata: dict[str, str] | None = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid checkpoint shard record")
        name = record.get("path")
        shard = _safe_shard(root, name)
        if name in shard_names:
            raise ValueError(f"duplicate checkpoint shard record: {name}")
        shard_names.add(name)
        if not shard.is_file():
            raise ValueError(f"checkpoint shard is missing: {name}")
        expected_bytes = record.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 1 or shard.stat().st_size != expected_bytes:
            raise ValueError(f"checkpoint shard size mismatch: {name}")
        expected_hash = _require_sha256(record.get("sha256"), f"shard hash for {name}")
        if verify_hashes and sha256_file(shard) != expected_hash:
            raise ValueError(f"checkpoint shard hash mismatch: {name}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if first_metadata is None:
                    first_metadata = metadata
                elif metadata != first_metadata:
                    raise ValueError(f"checkpoint shard metadata differs: {name}")
                if _format_from_metadata(metadata) != CHECKPOINT_FORMAT:
                    raise ValueError(f"checkpoint shard has wrong quantization metadata: {name}")
                if _source_from_metadata(metadata) != source_sha256:
                    raise ValueError(f"checkpoint shard has wrong source hash metadata: {name}")
                if source_metadata is not None and any(
                    metadata.get(key) != value for key, value in source_metadata.items()
                ):
                    raise ValueError(f"checkpoint shard did not preserve source metadata: {name}")
                keys = set(handle.keys())
                if not keys or keys & all_keys:
                    raise ValueError(f"checkpoint shard has duplicate or empty tensor headers: {name}")
                for key in keys:
                    _validate_tensor(key, handle.get_tensor(key), keys, schema)
        except (SafetensorError, OSError) as error:
            raise ValueError(f"invalid checkpoint shard: {name}") from error
        all_keys.update(keys)
        shard_keys[name] = keys
        shards.append(shard)
    if schema is not None:
        if set(schema) != {key for key in all_keys if not key.endswith(".weight_scale")}:
            raise ValueError("manifest source_schema conflicts with checkpoint tensors")
    declared_shards = manifest.get("shards", len(records))
    if declared_shards != len(records):
        raise ValueError("manifest shard count conflicts with checkpoint records")
    source_tensors = manifest.get("source_tensors")
    if source_tensors is not None and (
        not isinstance(source_tensors, int)
        or source_tensors != len({key for key in all_keys if not key.endswith(".weight_scale")})
    ):
        raise ValueError("manifest source_tensors conflicts with checkpoint headers")
    extra_shards = {item.name for item in root.glob("*.safetensors")} - shard_names
    if extra_shards:
        raise ValueError(f"checkpoint directory has unrecorded shards: {sorted(extra_shards)}")
    _validate_index(root, shard_names, shard_keys, required="schema_version" in manifest)
    return tuple(str(shard) for shard in shards)
