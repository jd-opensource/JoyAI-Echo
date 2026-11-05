"""Opt-in scaled E4M3 FFNs for prequantized Echo-WM checkpoints on ROCm."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer import LTXModel
from ltx_core.quantization.fp8_cast import UPCAST_DURING_INFERENCE
from ltx_core.quantization.rocm_checkpoint import CHECKPOINT_FORMAT, FP8_DTYPE


FFN_SUFFIXES = ("ff.net.0.proj", "ff.net.2")


def require_rocm_fp8(device: torch.device, *, probe: bool = False, fast_accum: bool = False) -> None:
    if device.type != "cuda" or torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("rocm-fp8 requires a ROCm PyTorch build and a working HIP device")
    if not callable(getattr(torch, "_scaled_mm", None)):
        raise RuntimeError("this PyTorch build does not provide torch._scaled_mm")
    if not probe:
        return
    with torch.inference_mode(), torch.cuda.device(device):
        value = torch.ones((16, 16), device=device, dtype=FP8_DTYPE)
        scale = torch.ones((), device=device, dtype=torch.float32)
        try:
            result = torch._scaled_mm(
                value, value.t(), scale, scale, out_dtype=torch.bfloat16,
                use_fast_accum=fast_accum,
            )
            torch.cuda.synchronize(device)
        except RuntimeError as error:
            raise RuntimeError(
                f"native E4M3 scaled matmul is unavailable on {torch.cuda.get_device_name(device)}"
            ) from error
        if not torch.equal(result, torch.full((16, 16), 16.0, device=device, dtype=torch.bfloat16)):
            raise RuntimeError("native E4M3 scaled matmul failed the device correctness probe")


class ROCmScaledFP8Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        device: torch.device | str | None = None, *, reference: bool = False,
        fast_accum: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reference = reference
        self.fast_accum = fast_accum
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=FP8_DTYPE, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32, device=device), requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        required = ("weight", "weight_scale", "bias") if self.bias is not None else ("weight", "weight_scale")
        for name in required:
            if prefix + name not in state_dict:
                error_msgs.append(f"prequantized FFN is missing {prefix + name}")
        weight = state_dict.get(prefix + "weight")
        if weight is not None and weight.dtype != FP8_DTYPE:
            error_msgs.append(f"{prefix}weight must use {FP8_DTYPE}, got {weight.dtype}")
        scale = state_dict.get(prefix + "weight_scale")
        if scale is not None and (
            scale.dtype != torch.float32 or scale.shape != torch.Size([])
            or scale.is_meta or not bool(torch.isfinite(scale) & (scale > 0))
        ):
            error_msgs.append(f"{prefix}weight_scale must be a finite positive FP32 scalar")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {value.shape[-1]}")
        bias = None if self.bias is None else self.bias.to(value.dtype)
        if self.reference:
            weight = self.weight.float().mul(self.weight_scale).to(value.dtype)
            return F.linear(value, weight, bias)

        require_rocm_fp8(value.device)
        if value.dtype != torch.bfloat16:
            raise ValueError("the ROCm FP8 path currently supports BF16 activations only")
        if self.in_features % 16 or self.out_features % 16:
            raise ValueError("ROCm FP8 linear dimensions must be multiples of 16")
        flat = value.reshape(-1, self.in_features)
        limit = torch.finfo(FP8_DTYPE).max
        input_scale = flat.float().abs().amax().clamp_min(1e-12).div(limit)
        quantized = flat.float().div(input_scale).clamp(-limit, limit).to(FP8_DTYPE)
        output = torch._scaled_mm(
            quantized, self.weight.t(), input_scale.float(), self.weight_scale,
            bias=bias, out_dtype=value.dtype, use_fast_accum=self.fast_accum,
        )
        return output.reshape(*value.shape[:-1], self.out_features)


def _prepare_ffns(model: nn.Module, *, reference: bool, fast_accum: bool) -> nn.Module:
    replacements = [
        (name, module) for name, module in model.named_modules()
        if name.startswith("transformer_blocks.")
        and name.endswith(FFN_SUFFIXES) and isinstance(module, nn.Linear)
    ]
    for name, linear in replacements:
        parent, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attribute, ROCmScaledFP8Linear(
            linear.in_features, linear.out_features, bias=linear.bias is not None,
            device=linear.weight.device, reference=reference, fast_accum=fast_accum,
        ))
    return model


def _preserve_tensor(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    return [KeyValueOperationResult(key, value)]


def rocm_fp8_operations(*, reference: bool = False, fast_accum: bool = False):
    """Consume the converted representation unchanged; never requantize during load."""
    mode = "reference" if reference else "native"
    return (
        SDOps(f"rocm_fp8_prequantized_{mode}").with_kv_operation(_preserve_tensor),
        (
            ModuleOps(
                name=f"rocm_fp8_{mode}_fast_accum_{fast_accum}",
                matcher=lambda model: isinstance(model, LTXModel),
                mutator=partial(_prepare_ffns, reference=reference, fast_accum=fast_accum),
            ),
            UPCAST_DURING_INFERENCE,
        ),
    )
