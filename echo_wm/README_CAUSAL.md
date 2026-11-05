<div align="center">

<h1>Echo-WM Flash Preview: 4-Step Autoregressive Inference</h1>

[🤗 Model Weights](https://huggingface.co/Echo-Team/Echo-WM) ·
[🌐 Project Page](https://echo-team-joy-future-academy-jd.github.io/Echo-1.5-Page/wm/)

</div>

## 🌍 Overview

**Echo-WM Flash Preview** is a 4-step autoregressive variant of Echo-WM. It
takes a first-frame image, a text prompt, and an Action DSL string, then
generates video and audio jointly. Classifier-free guidance is distilled into
the model through DMD, so inference does not require a separate CFG scale.

| Model | Denoising | Context cache | Status |
| --- | ---: | --- | --- |
| **Echo-WM Flash Preview** | 4 steps per block | Bounded sink-plus-FIFO | Available |

The default cache combines a persistent attention sink with recent FIFO
history. Camera conditioning uses bounded anchor translation to preserve
relative geometry as the active window moves.

The longer-horizon Echo-WM Flash checkpoint will be released alongside a
streaming online demo in a future update. Stay tuned.

An opt-in, prequantized ROCm path is documented in
[README_ROCM.md](README_ROCM.md). The default checkpoint, four-step schedule,
audio-video generation, and CUDA installation remain unchanged.

## 📥 Download Checkpoints

Download the Echo-WM checkpoints from the `echo_wm` directory:

```bash
hf download Echo-Team/Echo-WM --local-dir checkpoints
```

This creates:

```text
checkpoints/echo-wm-base.safetensors
checkpoints/echo-wm-flash.safetensors
```

Echo-WM Flash Preview loads the single merged `echo-wm-flash.safetensors`
checkpoint; no separate action adapter or training checkpoint is required.

Download the Gemma 3 text encoder separately:

```bash
hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir checkpoints/gemma-3
```

Gemma 3 is a gated repository. Accept its license and run `hf auth login`
before downloading. Use the unquantized weights above rather than the quantized
Q4_0 files; the text encoder runs in bfloat16.

## 🧪 Validate the Installation

Check the CLI and list the bundled examples without loading the model:

```bash
python inference_wm_causal.py --help
python scripts/run_wm_case_causal.py --list
```

Run the focused tests:

```bash
python -m pytest -q tests/test_echo_wm_causal.py
```

## 💻 Inference

### Recommended: run a checked-in case

Each case contains an input image, the original WBench caption, an Action DSL
string, FOV, and seed. The runner infers the output length from the action
durations:

```bash
cd /path/to/JoyAI-Echo/echo_wm
python scripts/run_wm_case_causal.py \
  --case examples/wm_causal_cases/0079 \
  --checkpoint checkpoints/echo-wm-flash.safetensors \
  --gemma-path checkpoints/gemma-3 \
  --video_local_attn_size 19 \
  --video_sink_size 7 \
  --video_chunk_size 3 \
  --output-dir outputs/wm_cases_causal
```

For case `0079`, the four 96-frame actions produce 385 output frames, including
the first frame. Use `--dry-run` to print the resolved command without loading
the model. By default, each run writes both the raw video and a second video
with the action HUD overlay, named `<output_stem>_action.mp4`. Pass
`--no-action-overlay` to skip the overlay copy.

### Example: image + prompt + action string

```bash
python inference_wm_causal.py \
  --image examples/wm_causal_cases/0079/input.jpg \
  --prompt "An enchanted crystal cave with massive prismatic crystal formations in purple, teal, and pink. Bioluminescent fungi glow on the cave floor and walls. Floating light motes drift through the air. The crystals refract light into rainbow spectra. To the right, a large crystalline cave monster with glowing purple eyes lurks behind tall crystal clusters. Deep cavern atmosphere with ethereal luminescence. Further to the right beyond the monster, a subterranean crystal pool glows with turquoise light, fed by a thin waterfall dripping from a stalactite cluster. The cave opens into a wider chamber with an ancient stone altar covered in glowing runes. First-person viewer. First-person view with the right hand holding a twisted wooden magic wand topped with a bright blue-white crystal orb that radiates light. The wand rotates together with the viewer's perspective when turning." \
  --action-str "l-96,l-96,l-96,l-96" \
  --checkpoint checkpoints/echo-wm-flash.safetensors \
  --gemma-path checkpoints/gemma-3 \
  --video_local_attn_size 19 \
  --video_sink_size 7 \
  --video_chunk_size 3 \
  --num-frames 385 \
  --fov-deg 70 \
  --output outputs/flash_result.mp4
```

The checked-in causal examples keep the original WBench captions verbatim.
They are passed directly to the Gemma tokenizer/text encoder; causal inference
does not invoke a prompt enhancer or rewrite them into six fields.

### Optional controls

```text
--video_local_attn_size INT   Total video cache window (default: 19)
--video_sink_size INT         Persistent video sink frames (default: 7)
--video_chunk_size INT        Generated video frames per latent block (Flash Preview: 3)
--num-frames INT        Decoded output length (default: 241)
--no-audio              Write a video-only MP4
--action-overlay        Write a separate action HUD copy (default: enabled)
--no-action-overlay     Skip the action HUD copy
--seed INT              Set the random seed
```

Hyphenated aliases such as `--video-local-attn-size` are accepted. The checked-in
16-second examples use 385 frames rather than the direct CLI default of 241.
The corresponding audio local-attention window and sink size are derived
automatically from the video settings to keep audio and video blocks aligned.

## 🎮 Action Control

Each Action DSL segment is `<keys>-<frames>`, and segments are joined by commas:

```text
w/s       forward / backward
a/d       strafe left / right
i/k       pitch up / down
j/l       yaw left / right
none      hold the camera still
```

Keys can be combined, for example `w-96,wj-96,w-96,d-96`.

## 🖼️ Example Gallery

|     Case | Scene                    | Four 4-second actions                    | Action DSL                  |
| -------: | ------------------------ | ---------------------------------------- | --------------------------- |
| `0024` | Ancient Roman ruins | Yaw right/forward alternating | `l-96,w-96,l-96,w-96` |
| `0075` | Cherry blossom garden | Forward four times | `w-96,w-96,w-96,w-96` |
| `0079` | Enchanted crystal cave | Yaw right four times | `l-96,l-96,l-96,l-96` |
| `0081` | Sunlit artist studio     | Pitch down/up, backward/forward          | `k-96,i-96,s-96,w-96`     |
| `0122` | Volcanic crater | Strafe left/right alternating | `a-96,d-96,a-96,d-96` |
| `0170` | Mythological marble hall | Forward/backward, strafe left, yaw right | `w-96,s-96,a-96,l-96`     |

The images and original WBench captions are under
[`examples/wm_causal_cases`](examples/wm_causal_cases). All cases use the
WBench `wbench_4s_turn_rot0.4_trans0.05` camera setup.

### Multi-GPU batch inference

Run all checked-in cases across the selected GPUs:

```bash
GPU_LIST=0,1,2 bash scripts/run_wm_causal_cases_multigpu.sh
```

Cases are assigned round-robin, with at most one inference process active per
GPU. Select a subset with:

```bash
CASES="0081 0170" GPU_LIST=0,1 bash scripts/run_wm_causal_cases_multigpu.sh
```

Supported environment variables:

```text
GPU_LIST        Comma-separated GPU indices (default: 0,1,2)
CASES           Subset of cases (default: every checked-in case)
PYTHON_BIN      Interpreter to use (default: python from the active environment)
ACTION_OVERLAY  Write action HUD copies (default: 1; set to 0 to skip them)
```

Outputs are written under `outputs/wm_causal_cases_multigpu/<case>/`. Each
successful case contains `result.mp4`, `result_action.mp4`, and a per-case
`run_gpu<N>.log`. Set `ACTION_OVERLAY=0` to omit `result_action.mp4`.

## ⚙️ 4-Step Configuration

Public defaults are in
[`configs/inference_wm_causal.yaml`](configs/inference_wm_causal.yaml):

```yaml
video:
  width: 1280
  height: 704
  num_frames: 241
  fps: 24

causal:
  timesteps: [1000, 750, 500, 250]
  video_local_attn_size: 19
  video_sink_size: 7
  video_chunk_size: 3

action:
  translation_speed: 0.05
  rotation_speed_deg: 0.4
  pitch_limit_deg: 40.0
  fov_deg: 70.0
```

The three causal window parameters are measured in video latent frames.
Echo-WM Flash Preview uses a fixed `video_chunk_size` of 3. Each transformer layer
maintains five bounded temporal caches: video self-attention, audio
self-attention, audio-to-video cross-attention, video-to-audio cross-attention,
and UCPE camera attention. The video settings directly configure the
video-side caches; the aligned audio window and sink are derived automatically
(the defaults map `19/7` video frames to `152/52` audio frames). Video and audio
text K/V are static prompt caches initialized once and are not part of these
five rolling caches.

`--num-frames` is measured in decoded frames. Valid output lengths follow
`1 + 24m` (for example, 241 or 385 frames), corresponding to `1 + 3m` video
latent frames after the VAE temporal compression.

## 📄 Citation

```bibtex
@article{echo_wm,
  title   = {Echo-WM: Open and Enterable Omnimodal World Models},
  author  = {Echo Team},
  journal = {arXiv preprint},
  year    = {2026}
}
```
