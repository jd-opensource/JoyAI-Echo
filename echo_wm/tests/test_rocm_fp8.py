from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F


WM_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(WM_ROOT), str(WM_ROOT / "ltx-core" / "src"),
    str(WM_ROOT / "ltx-causal" / "src"),
    str(WM_ROOT / "ltx-pipelines" / "src"),
]

import ltx_core.loader  # noqa: E402, F401
from ltx_core.quantization import QuantizationPolicy  # noqa: E402
from ltx_core.quantization.rocm_fp8 import (  # noqa: E402
    FP8_DTYPE,
    ROCmScaledFP8Linear,
    require_rocm_fp8,
)
from inference_wm_causal import _quantization_policy, parse_args  # noqa: E402
from scripts.run_wm_case_causal import build_command, parse_args as parse_case_args  # noqa: E402


def state_dict(*, device="cpu"):
    generator = torch.Generator().manual_seed(83)
    weight = torch.randn(32, 32, generator=generator).mul_(0.1)
    limit = torch.finfo(FP8_DTYPE).max
    scale = weight.abs().amax().clamp_min(1e-12).div(limit)
    return {
        "weight": weight.div(scale).clamp(-limit, limit).to(FP8_DTYPE).to(device),
        "weight_scale": scale.float().to(device),
        "bias": torch.randn(32, generator=generator).mul_(0.01).bfloat16().to(device),
    }


def test_default_cli_preserves_unquantized_policy_and_explicit_opt_in():
    args = parse_args(["--image", "input.jpg", "--prompt", "test", "--action-str", "none-24"])
    assert args.quantization == "none"
    assert args.rocm_fp8_fast_accum is False
    assert _quantization_policy("none", torch.device("cpu")) is None
    assert QuantizationPolicy.fp8_cast().checkpoint_format is None
    with pytest.raises(ValueError, match="requires"):
        _quantization_policy("none", torch.device("cpu"), fast_accum=True)
    with pytest.raises(ValueError, match="unknown"):
        _quantization_policy("invalid", torch.device("cpu"))


def test_case_runner_forwards_rocm_policy_without_changing_default_command():
    arguments = ["--case", "examples/wm_causal_cases/0079", "--dry-run"]
    ordinary = build_command(parse_case_args(arguments))
    assert "--quantization" not in ordinary
    assert "--rocm-fp8-fast-accum" not in ordinary
    selected = build_command(parse_case_args(
        arguments + ["--quantization", "rocm-fp8", "--rocm-fp8-fast-accum"]
    ))
    assert selected[selected.index("--quantization") + 1] == "rocm-fp8"
    assert "--rocm-fp8-fast-accum" in selected


def test_rocm_gate_does_not_fall_back_on_cpu_or_cuda_build():
    with pytest.raises(RuntimeError, match="ROCm PyTorch"):
        require_rocm_fp8(torch.device("cpu"))
    with patch.object(torch.version, "hip", None), pytest.raises(RuntimeError, match="ROCm PyTorch"):
        require_rocm_fp8(torch.device("cuda"))
    with (
        patch.object(torch.version, "hip", "test"),
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch, "_scaled_mm", None),
        pytest.raises(RuntimeError, match="_scaled_mm"),
    ):
        require_rocm_fp8(torch.device("cuda"))


def test_reference_uses_the_same_quantized_representation():
    layer = ROCmScaledFP8Linear(32, 32, reference=True).eval()
    source = state_dict()
    layer.load_state_dict(source, assign=True)
    value = torch.randn(2, 3, 32).bfloat16()
    expected = F.linear(value, source["weight"].float().mul(source["weight_scale"]).bfloat16(), source["bias"])
    with torch.inference_mode():
        assert torch.equal(layer(value), expected)
    assert layer.weight.dtype == FP8_DTYPE
    assert layer.weight_scale.dtype == torch.float32


@pytest.mark.parametrize("missing", ["weight", "weight_scale", "bias"])
def test_missing_fp8_storage_or_scale_fails_even_with_nonstrict_load(missing):
    layer = ROCmScaledFP8Linear(32, 32, device="meta", reference=True)
    source = state_dict()
    del source[missing]
    with pytest.raises(RuntimeError, match="prequantized FFN is missing"):
        layer.load_state_dict(source, strict=False, assign=True)


@pytest.mark.parametrize("scale", [
    torch.tensor(0.0), torch.tensor(-1.0), torch.tensor(float("nan")),
    torch.tensor(float("inf")), torch.ones(1), torch.tensor(1.0, dtype=torch.bfloat16),
])
def test_invalid_weight_scale_fails_closed(scale):
    source = state_dict()
    source["weight_scale"] = scale
    layer = ROCmScaledFP8Linear(32, 32, reference=True)
    with pytest.raises(RuntimeError, match="positive FP32 scalar"):
        layer.load_state_dict(source, assign=True)


def test_unconverted_weight_is_not_silently_requantized():
    source = state_dict()
    source["weight"] = source["weight"].bfloat16()
    with pytest.raises(RuntimeError, match="weight must use"):
        ROCmScaledFP8Linear(32, 32).load_state_dict(source, assign=True)


def test_policy_replaces_only_ffns_and_preserves_all_storage_and_scale_keys():
    from ltx_core.model.transformer import LTXModel

    with torch.device("meta"):
        model = LTXModel(num_layers=1)
    policy = QuantizationPolicy.rocm_fp8_prequantized()
    expected = set(model.state_dict())
    ffn_names = [
        name for name, module in model.named_modules()
        if name.startswith("transformer_blocks.")
        and name.endswith(("ff.net.0.proj", "ff.net.2"))
        and isinstance(module, torch.nn.Linear)
    ]
    expected.update(f"{name}.weight_scale" for name in ffn_names)
    for operation in policy.module_ops:
        if operation.matcher(model):
            model = operation.mutator(model)
    assert set(model.state_dict()) == expected
    assert len(ffn_names) == 4
    assert all(isinstance(model.get_submodule(name), ROCmScaledFP8Linear) for name in ffn_names)
    assert isinstance(model.transformer_blocks[0].attn1.to_q, torch.nn.Linear)
    for key, value in model.state_dict().items():
        result, = policy.sd_ops.apply_to_key_value(key, value)
        assert result.new_key == key
        assert result.new_value is value


def test_rocm_dependency_file_does_not_replace_accelerator_wheels():
    import re

    def dependencies(path):
        result = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                result.update(dependencies(path.parent / line[3:]))
            else:
                result.add(line)
        return result

    cuda = dependencies(WM_ROOT / "requirements.txt")
    rocm = dependencies(WM_ROOT / "requirements-rocm.txt")
    assert cuda - rocm == {
        "torch==2.9.1", "torchaudio==2.9.1", "torchvision==0.24.1", "xformers==0.0.33.post2",
    }
    assert not rocm - cuda
    assert not any(re.match(r"^(torch|torchaudio|torchvision|xformers)([<=>]|$)", line) for line in rocm)


def test_ordinary_model_ledger_paths_are_unchanged():
    from ltx_pipelines.utils.model_ledger import ModelLedger

    for policy in (None, QuantizationPolicy.fp8_cast()):
        ledger = ModelLedger(
            dtype=torch.bfloat16, device=torch.device("cpu"),
            checkpoint_path="ordinary.safetensors", quantization=policy,
        )
        assert ledger.transformer_builder.model_path == "ordinary.safetensors"
        assert ledger.vae_decoder_builder.model_path == "ordinary.safetensors"
        assert ledger.audio_decoder_builder.model_path == "ordinary.safetensors"


def test_prequantized_policy_cannot_silently_fuse_loras():
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_pipelines.utils.model_ledger import ModelLedger

    with pytest.raises(ValueError, match="LoRA fusion"):
        ModelLedger(
            dtype=torch.bfloat16, device=torch.device("cpu"),
            checkpoint_path="converted", quantization=QuantizationPolicy.rocm_fp8_prequantized(),
            loras=(LoraPathStrengthAndSDOps("adapter.safetensors", 1.0, None),),
        )


@pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm device required")
@pytest.mark.parametrize("fast_accum", [False, True])
def test_real_rocm_fp8_capability_and_same_checkpoint_oracle(fast_accum):
    device = torch.device("cuda")
    require_rocm_fp8(device, probe=True, fast_accum=fast_accum)
    source = state_dict(device=device)
    native = ROCmScaledFP8Linear(32, 32, device=device, fast_accum=fast_accum).eval()
    reference = ROCmScaledFP8Linear(32, 32, device=device, reference=True).eval()
    native.load_state_dict(source, assign=True)
    reference.load_state_dict(source, assign=True)
    torch.manual_seed(9)
    value = torch.randn(2, 16, 32, device=device, dtype=torch.bfloat16).mul_(0.25)
    with torch.inference_mode():
        actual, expected = native(value), reference(value)
    assert torch.isfinite(actual).all()
    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.999
    with pytest.raises(ValueError, match="BF16 activations"):
        native(value.half())
