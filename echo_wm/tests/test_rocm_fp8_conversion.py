from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


WM_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WM_ROOT / "ltx-core" / "src")]

import ltx_core.loader  # noqa: E402, F401
from ltx_core.quantization.rocm_checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    FP8_DTYPE,
    resolve_rocm_checkpoint,
)


def _converter_module():
    spec = importlib.util.spec_from_file_location(
        "convert_rocm_fp8", WM_ROOT / "tools" / "convert_rocm_fp8.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONVERTER = _converter_module()


def _source(path: Path, *, altered: bool = False) -> Path:
    values = {
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight": torch.tensor(
            [[1.0, -2.0], [3.0, 4.0]], dtype=torch.bfloat16
        ),
        "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.bias": torch.tensor(
            [1.0, 2.0], dtype=torch.bfloat16
        ),
        "model.diffusion_model.transformer_blocks.0.audio_ff.net.2.weight": torch.tensor(
            [[-1.0, 2.0], [3.0, -4.0]], dtype=torch.bfloat16
        ),
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
        ),
        "model.diffusion_model.other.weight": torch.tensor([5.0 if altered else 1.0]),
    }
    save_file(values, path, metadata={"config": "tiny"})
    return path


def _convert(source: Path, target: Path, **kwargs):
    return CONVERTER.convert_checkpoint(
        source, target, max_shard_bytes=12, expected_source_sha256=hashlib.sha256(
            source.read_bytes()
        ).hexdigest(), **kwargs,
    )


def test_tiny_conversion_validates_and_resume_is_byte_identical(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    manifest = _convert(source, target)
    paths = resolve_rocm_checkpoint(target)
    assert len(paths) > 1
    assert (target / "model.safetensors.index.json").is_file()
    before = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    resumed = _convert(source, target, resume=True)
    after = {path.name: path.read_bytes() for path in target.iterdir() if path.is_file()}
    assert resumed == manifest
    assert after == before

    tensors = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensors.update({key: handle.get_tensor(key) for key in handle.keys()})
    weight = "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"
    assert tensors[weight].dtype == FP8_DTYPE
    assert tensors[f"{weight.removesuffix('.weight')}.weight_scale"].dtype == torch.float32
    assert tensors[f"{weight.removesuffix('.weight')}.bias"].dtype == torch.bfloat16


@pytest.mark.parametrize("kind", ("tampered", "missing"))
def test_missing_or_tampered_shard_fails_closed(tmp_path, kind):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    manifest = _convert(source, target)
    shard = target / manifest["shard_records"][0]["path"]
    if kind == "tampered":
        shard.write_bytes(shard.read_bytes() + b"corrupt")
    else:
        shard.unlink()
    with pytest.raises(ValueError, match="shard"):
        resolve_rocm_checkpoint(target)


def test_resume_rejects_mismatched_source_existing_malformed_shard_and_plan(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    _convert(source, target)
    with pytest.raises(ValueError, match="max_shard_bytes"):
        CONVERTER.convert_checkpoint(source, target, max_shard_bytes=300, resume=True)
    _source(source, altered=True)
    with pytest.raises(ValueError, match="source_sha256"):
        _convert(source, target, resume=True)
    (target / "manifest.json").unlink()
    with pytest.raises(ValueError, match="no manifest"):
        _convert(source, target, resume=True)


def test_manifest_hash_cannot_make_a_malformed_existing_shard_resumable(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    manifest = _convert(source, target)
    record = manifest["shard_records"][0]
    shard = target / record["path"]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    save_file(tensors, shard, metadata={"quantization": "wrong", "source_sha256": manifest["source_sha256"]})
    disk_manifest = json.loads((target / "manifest.json").read_text())
    disk_manifest["shard_records"][0]["bytes"] = shard.stat().st_size
    disk_manifest["shard_records"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    (target / "manifest.json").write_text(json.dumps(disk_manifest))
    with pytest.raises(ValueError, match="quantization metadata"):
        _convert(source, target, resume=True)


def test_legacy_manifest_and_metadata_aliases_are_accepted_without_index(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    shard = root / "model-00001-of-00001.safetensors"
    weight = "model.diffusion_model.transformer_blocks.0.ff.net.2.weight"
    scale_key = weight.removesuffix(".weight") + ".weight_scale"
    source_hash = "a" * 64
    save_file(
        {
            weight: torch.ones((2, 2), dtype=FP8_DTYPE),
            scale_key: torch.tensor(0.25, dtype=torch.float32),
        },
        shard,
        metadata={"rsi_quantization": CHECKPOINT_FORMAT, "rsi_source_sha256": source_hash},
    )
    (root / "manifest.json").write_text(json.dumps({
        "strategy": CHECKPOINT_FORMAT,
        "source_sha256": source_hash,
        "source_tensors": 1,
        "shard_records": [{
            "path": shard.name, "bytes": shard.stat().st_size,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        }],
    }))
    assert resolve_rocm_checkpoint(root) == (str(shard.resolve()),)


def test_interrupted_conversion_resumes_only_verified_completed_shards(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    original_save = CONVERTER.save_file
    calls = 0

    def interrupt_second_save(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interruption")
        return original_save(*args, **kwargs)

    with patch.object(CONVERTER, "save_file", interrupt_second_save):
        with pytest.raises(OSError, match="synthetic interruption"):
            _convert(source, target)
    assert not (target / "manifest.json").exists()
    progress = json.loads((target / ".conversion-progress.json").read_text())
    assert len(progress["shard_records"]) == 1
    first = target / progress["shard_records"][0]["path"]
    before = first.read_bytes(), first.stat().st_mtime_ns
    resumed = _convert(source, target, resume=True)
    assert before == (first.read_bytes(), first.stat().st_mtime_ns)
    assert not (target / ".conversion-progress.json").exists()
    assert len(resolve_rocm_checkpoint(target)) == len(resumed["shard_records"])


def test_resume_rejects_corruption_in_an_incomplete_conversion(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    original_save = CONVERTER.save_file
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("interrupted")
        return original_save(*args, **kwargs)

    with patch.object(CONVERTER, "save_file", interrupt), pytest.raises(OSError):
        _convert(source, target)
    progress = json.loads((target / ".conversion-progress.json").read_text())
    shard = target / progress["shard_records"][0]["path"]
    content = shard.read_bytes()
    shard.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        _convert(source, target, resume=True)


def test_conversion_rejects_nonfinite_selected_weights(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({
        "model.diffusion_model.transformer_blocks.0.ff.net.2.weight": torch.full((2, 2), float("nan")),
    }, source)
    with pytest.raises(ValueError, match="finite"):
        _convert(source, tmp_path / "converted")


def test_manifest_rejects_unknown_schema_and_path_traversal(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    manifest = _convert(source, target)
    altered = dict(manifest, schema_version=2)
    (target / "manifest.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="schema_version"):
        resolve_rocm_checkpoint(target)
    manifest["shard_records"][0]["path"] = "../outside.safetensors"
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        resolve_rocm_checkpoint(target)


def test_manifest_rejects_missing_scale_even_with_updated_shard_hash(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    manifest = _convert(source, target)
    record = next(
        record for record in manifest["shard_records"]
        if any(
            key.endswith(".weight_scale")
            for key in safe_open(target / record["path"], framework="pt").keys()
        )
    )
    shard = target / record["path"]
    with safe_open(shard, framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
    del tensors[next(key for key in tensors if key.endswith(".weight_scale"))]
    save_file(tensors, shard, metadata=metadata)
    record["bytes"] = shard.stat().st_size
    record["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing its weight_scale"):
        resolve_rocm_checkpoint(target)


def test_selected_fp32_attention_storage_is_preserved_and_checked(tmp_path):
    source, target = tmp_path / "source.safetensors", tmp_path / "converted"
    key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    save_file({key: torch.ones((2, 2), dtype=torch.float32)}, source)
    manifest = CONVERTER.convert_checkpoint(source, target, max_shard_bytes=20)
    shard = target / manifest["shard_records"][0]["path"]
    with safe_open(shard, framework="pt") as handle:
        assert handle.get_tensor(key).dtype == torch.float32
        metadata = handle.metadata()
    save_file({key: torch.ones((2, 2), dtype=torch.float16)}, shard, metadata=metadata)
    manifest["shard_records"][0].update(
        bytes=shard.stat().st_size, sha256=hashlib.sha256(shard.read_bytes()).hexdigest()
    )
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dtype mismatch"):
        resolve_rocm_checkpoint(target)


@pytest.mark.parametrize("kind", ["duplicate", "extra-shard", "index", "config"])
def test_inconsistent_shard_sets_and_index_fail_closed(tmp_path, kind):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    manifest = _convert(source, target)
    record = manifest["shard_records"][0]
    if kind == "duplicate":
        manifest["shard_records"].append(dict(record))
        (target / "manifest.json").write_text(json.dumps(manifest))
    elif kind == "extra-shard":
        (target / "extra.safetensors").write_bytes((target / record["path"]).read_bytes())
    elif kind == "index":
        index_path = target / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"].pop(next(iter(index["weight_map"])))
        index_path.write_text(json.dumps(index))
    else:
        shard = target / record["path"]
        with safe_open(shard, framework="pt") as handle:
            metadata = handle.metadata()
            tensors = {key: handle.get_tensor(key).clone() for key in handle.keys()}
        metadata["config"] = "different"
        save_file(tensors, shard, metadata=metadata)
        record.update(bytes=shard.stat().st_size, sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
        (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        resolve_rocm_checkpoint(target)


def test_converted_shard_loads_into_the_prequantized_linear_without_requantization(tmp_path):
    from ltx_core.quantization.rocm_fp8 import ROCmScaledFP8Linear

    torch.manual_seed(44)
    prefix = "model.diffusion_model.transformer_blocks.0.ff.net.2."
    source, target = tmp_path / "source.safetensors", tmp_path / "converted"
    original = {
        prefix + "weight": torch.randn(32, 32).mul_(0.1).bfloat16(),
        prefix + "bias": torch.randn(32).mul_(0.01).bfloat16(),
    }
    save_file(original, source, metadata={"config": "{}"})
    CONVERTER.convert_checkpoint(source, target)
    loaded = {}
    for path in resolve_rocm_checkpoint(target):
        with safe_open(path, framework="pt") as handle:
            loaded.update({
                key.removeprefix(prefix): handle.get_tensor(key)
                for key in handle.keys()
            })
    layer = ROCmScaledFP8Linear(32, 32, device="meta", reference=True)
    layer.load_state_dict(loaded, assign=True)
    assert layer.weight.data_ptr() == loaded["weight"].data_ptr()
    assert layer.weight_scale.data_ptr() == loaded["weight_scale"].data_ptr()
    value = torch.randn(2, 3, 32).bfloat16()
    expected = torch.nn.functional.linear(
        value, loaded["weight"].float().mul(loaded["weight_scale"]).bfloat16(),
        loaded["bias"],
    )
    with torch.no_grad():
        assert torch.equal(layer(value), expected)


@pytest.mark.parametrize("after_rename", [False, True])
def test_resume_recovers_interruption_at_the_shard_commit_boundary(tmp_path, after_rename):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    original_replace = CONVERTER.os.replace

    def interrupt_rename(src, dst):
        if Path(dst).suffix == ".safetensors":
            if after_rename:
                original_replace(src, dst)
            raise OSError("interrupted shard commit")
        return original_replace(src, dst)

    with patch.object(CONVERTER.os, "replace", interrupt_rename):
        with pytest.raises(OSError, match="interrupted shard commit"):
            _convert(source, target)
    existing = list(target.glob("*.safetensors"))
    assert len(existing) == int(after_rename)
    if after_rename:
        final = existing[0]
        before = final.read_bytes(), final.stat().st_mtime_ns
    resumed = _convert(source, target, resume=True)
    if after_rename:
        assert before == (final.read_bytes(), final.stat().st_mtime_ns)
    assert len(resolve_rocm_checkpoint(target)) == len(resumed["shard_records"])
    assert not (target / ".conversion-progress.json").exists()


def test_pending_shard_recovery_rejects_changed_payload(tmp_path):
    source, target = _source(tmp_path / "source.safetensors"), tmp_path / "converted"
    original_replace = CONVERTER.os.replace

    def interrupt_after_rename(src, dst):
        original_replace(src, dst)
        if Path(dst).suffix == ".safetensors":
            raise OSError("interrupted shard commit")

    with patch.object(CONVERTER.os, "replace", interrupt_after_rename):
        with pytest.raises(OSError, match="interrupted shard commit"):
            _convert(source, target)
    shard, = target.glob("*.safetensors")
    value = shard.read_bytes()
    shard.write_bytes(value[:-1] + bytes([value[-1] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        _convert(source, target, resume=True)
