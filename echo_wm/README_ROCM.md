# Experimental Echo-WM Flash inference on ROCm

This opt-in path loads an `amd-fp8-mixed-v1` checkpoint without TensorRT-LLM or
xformers. Video and audio feed-forward weights use scaled E4M3 storage with
native PyTorch scaled matmul on ROCm. Other selected linear weights use E4M3
storage and the existing BF16-upcast execution path. VAE, normalization, and
feed-forward biases retain their source precision.

This is a speed/precision trade-off, not lossless compression. It does not add
the interactive Web service, change camera conditioning, reduce inference
steps, or change the default CUDA/BF16 path.

## Environment

Use a separate virtual environment. Install a matching ROCm PyTorch,
torchvision, and torchaudio wheel set from the current
[AMD installation documentation](https://rocm.docs.amd.com/en/latest/), then:

```bash
cd echo_wm
python -m pip install -r requirements-rocm.txt
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0)); assert torch.version.hip and torch.cuda.is_available()"
```

Do not install the CUDA-pinned `requirements.txt` over this environment.
ROCm uses PyTorch's `cuda` device API. The native path requires BF16
activations, E4M3FN support, and `torch._scaled_mm`; a small correctness probe
runs before native CLI inference. Unsupported devices/builds fail explicitly,
rather than silently switching to another numerical path. The private PyTorch
API remains a compatibility boundary that must be revalidated on version
changes.

The initial hardware target is Radeon AI PRO R9700 (`gfx1201`) with ROCm
10.0.0 and PyTorch 2.13.0+rocm10.0.0. This does not establish support or
equivalent performance for other Radeon/Instinct GPUs.

## Convert a local official checkpoint

Obtain the official Flash checkpoint and Gemma weights as described in
[README_CAUSAL.md](README_CAUSAL.md). The original model and Gemma license
terms continue to apply to converted artifacts.

```bash
python tools/convert_rocm_fp8.py \
  --source checkpoints/echo-wm-flash.safetensors \
  --output-dir checkpoints/echo-wm-rocm-fp8 \
  --device cuda
```

Conversion is tensor-by-tensor with bounded output shards; it does not execute
or load the complete BF16 Transformer. CPU conversion is also available through
`--device cpu`. Keep space for both source and converted files until validation
is complete. Re-running with `--resume` verifies existing completed shard
records before continuing; incompatible or damaged output is rejected.
An interrupted temporary shard is replaced, but an unrecorded final shard is
rejected rather than silently trusted.
The progress journal records a completed shard's hash before its final rename,
so `--resume` can recover an interruption on either side of that rename without
accepting an unverified payload.

The output includes a manifest and a safetensors index. FFN weights remain in
`[out_features, in_features]` layout with a scalar FP32 `weight_scale`;
loading does not transpose, rescale, or requantize stored values. The loader
validates the manifest and shard identities. A directory's existence alone
does not select or prove a quantization format.

## Run the official causal entrypoint

Start with a short, reduced-resolution capacity smoke:

```bash
python inference_wm_causal.py \
  --image examples/wm_causal_cases/0079/input.jpg \
  --prompt "$(python -c 'import json; print(json.load(open("examples/wm_causal_cases/0079/case.json"))["prompt"])')" \
  --action-str "j-24" \
  --checkpoint checkpoints/echo-wm-rocm-fp8 \
  --gemma-path checkpoints/gemma-3 \
  --quantization rocm-fp8 \
  --width 576 --height 320 --num-frames 25 \
  --video-local-attn-size 4 --video-sink-size 1 \
  --timesteps 1000 750 500 250 \
  --seed 8181 \
  --no-action-overlay \
  --output outputs/rocm-smoke.mp4
```

For the same-checkpoint reference, repeat with
`--quantization rocm-fp8-reference` and a different output filename. The
independent scene smoke uses case `0081` for both the image and prompt paths.

`--quantization none` remains the default. `fp8-cast` explicitly selects the
existing cast policy; `rocm-fp8-reference` dequantizes the same stored FFN
weights for an oracle without native FP8 activation matmul.
`--rocm-fp8-fast-accum` is a separate optional performance/precision trade-off
and is off by default. It requires `--quantization rocm-fp8`.

The checked-in single-case runner accepts the same explicit quantization
selection. The multi-GPU shell launcher remains the existing CUDA-oriented
batch example; this PR does not claim multi-GPU model partitioning.

The causal pipeline still generates audio. `--no-audio` suppresses audio in
the exported file; it is not a video-only compute switch. LoRA fusion into this
prequantized representation is not supported and fails before loading.

## Validation and claim limits

```bash
python -m pytest -q tests/test_rocm_fp8.py tests/test_rocm_fp8_conversion.py \
  tests/test_echo_wm.py tests/test_echo_wm_causal.py
```

The existing CPU/CUDA policies and their defaults are preserved. ROCm-only
tests skip when a working ROCm device is absent. Native scalar-scale GEMM is
compared with a same-checkpoint BF16-dequantized reference, while converter
tests use small real safetensors files.

Component correctness is not an end-to-end quality claim. Validate real
prompt-conditioned video and generated audio for the intended scene, schedule,
resolution, and cache size before using the path. The short example above
does not establish full 704x1280 rollout capacity on a 32 GiB board, long-run
quality, or real-time throughput.
