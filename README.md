# whisper-inference

A from-scratch C++ inference engine for OpenAI's Whisper, built layer by layer against a
Python reference implementation — no ML framework at runtime, no black-box conversion tool.
The weight format, the memory layout, and eventually every kernel are hand-written and
numerically validated against PyTorch at every stage.

This is a personal project to learn how inference engines actually work, and to build
something tuned for the hardware in front of me (RTX 3050, 6GB VRAM) rather than for a
generic deployment target.

## Why build this instead of using whisper.cpp

Because the interesting part isn't running Whisper — it's the fifty small decisions that
make it run *correctly* and *fast*: how you lay out weights in memory so a `mmap` gives you
zero-copy tensors, where op-ordering differences compound into visible transcription errors,
which matmuls actually dominate the profile once you measure instead of guess. whisper.cpp
is used here only as a speed baseline to benchmark against, and as a second implementation
to consult when the math disagrees with Python and it's unclear which side is wrong.

## Architecture recap (what's actually being built)

Whisper is an encoder-decoder transformer with a DSP frontend, not a neural net end to end:

```
raw audio ──▶ resample 16kHz ──▶ STFT ──▶ mel filterbank ──▶ log-mel spectrogram
                                                                  (80 × 3000, 30s window)
                                                                        │
                                                                        ▼
                                      ┌─────────────── ENCODER ───────────────┐
                                      │ conv1d → conv1d (stride 2) → + posemb  │
                                      │ → N × [bidirectional self-attn + MLP]  │
                                      └─────────────────────────────────────────┘
                                                                        │
                                                          audio features (1500 × d_model)
                                                                        │
                                      ┌─────────────── DECODER ───────────────┐
                                      │ token+pos embed                        │
                                      │ → N × [causal self-attn                │
                                      │        + cross-attn(Q=dec, K/V=enc)    │
                                      │        + MLP]                          │
                                      │ → vocab logits (autoregressive)        │
                                      └─────────────────────────────────────────┘
```

## Build discipline

The single most common failure mode in a project like this is optimizing a forward pass
that was never verified correct — and then being unable to tell whether a bug is in the
math or the SIMD. So the plan is strict about ordering:

1. **Reference first.** Every component gets a matching `.npy` dump from real PyTorch
   Whisper before a line of C++ is written against it.
2. **Scalar, obviously-correct C++ before any optimization.** Naive loops, validated,
   *then* fast.
3. **Layer-by-layer numerical validation**, not end-to-end vibes. `np.allclose` at
   `atol=1e-4` per component; a mismatch gets bisected within the layer (post-QK,
   post-softmax, post-attention-output) rather than blamed on "floating point stuff."
4. **Optimization only starts once step 3 passes**, and only where a profile says time is
   actually going — not where it looks like it should.

## Current status

**Working now:**

- **Python reference harness** ([reference_setup.py](reference_setup.py)) — loads Whisper
  in PyTorch, runs a fixed audio sample through it, and hooks every submodule (both conv
  layers, every encoder block, every decoder block, final logits) to dump intermediate
  activations as `.npy`. This is the ground truth the C++ side gets diffed against, layer
  by layer, for the rest of the project.
- **Custom weight format** ([weight_conversion.py](weight_conversion.py)) — walks the
  model's `state_dict` and serializes it to a flat binary designed to be `mmap`'d directly,
  no parsing library required:

  ```
  "WSPR" magic │ version │ n_layers │ n_heads │ d_model │ n_mels │ vocab │ n_tensors │ align
  ── then, per tensor ──
  name_len │ name │ ndim │ dims[ndim] │ padding (to 64B) │ raw fp16 payload
  ```

  Weights are stored **fp16**, matching the released checkpoint's native precision exactly
  — verified bit-for-bit: 0 of 479 tensors in `small` change under an fp32→fp16→fp32
  round-trip. Every payload is **64-byte aligned**, which matters once the reader starts
  handing pointers to SIMD load instructions that require it. Result: a 244M-parameter
  model (`small`) serializes to a 483MB file that a C++ `mmap` can index into with zero
  parsing and zero copying.
- **Zero-copy C++ loader (in progress)** ([weight_loader.cpp](weight_loader.cpp)) — memory-maps
  the weight file and exposes every tensor as a non-owning view (`{ pointer, shape, numel }`)
  directly into the mapped pages. No `state_dict`, no deserialization pass, no duplicate copy
  of a half-gigabyte of weights sitting in two places at once.

**Next up:** mel spectrogram DSP (the one genuinely non-transformer-shaped piece — Hann
window, FFT, mel filterbank, log), then conv1d, then the encoder blocks, validated one at a
time against the reference dumps above.

## Validated correctness, concretely

Not "it compiles" — checked, numerically:

- Every tensor round-tripped through the weight format is **bit-identical** to the live
  PyTorch `state_dict` after dtype upcast.
- **0 of 479 tensors** lose precision under the fp16 conversion.
- **0 bytes of drift** between the writer's framing and an independent reader walking the
  same file — parsing 479 variable-length records lands exactly on EOF.
- Decoder logits dumped from the reference harness produce sane top-k next-token
  predictions (`" The"`, `" the"`, `" Quick"`, ...) against audio that opens
  *"The quick brown fox..."* — confirming the encoder → decoder → logits path is wired
  correctly, not just shape-correct.

## Stack

- **Python** (PyTorch + openai-whisper) — reference implementation and weight export only,
  not part of the runtime path.
- **C++17**, MSVC — the actual inference engine.
- Target hardware: RTX 3050 / 6GB VRAM, tuned for this machine rather than generic
  deployment.

## Roadmap

- [x] Reference tensor dumps (tiny, small)
- [x] Custom fp16, aligned, zero-copy weight format
- [ ] `mmap`-backed C++ weight loader
- [ ] Mel spectrogram (DSP, no ML)
- [ ] Conv1d frontend (stride-2, GELU)
- [ ] Encoder transformer blocks (LayerNorm, MHA, MLP) — validated block by block
- [ ] Decoder transformer blocks (causal self-attn, cross-attn with cached K/V, MLP)
- [ ] Greedy decoding loop + BPE detokenization
- [ ] Profile and optimize: SIMD matmul, multi-threaded attention heads, cache blocking
- [ ] Benchmark against whisper.cpp on identical audio/model
