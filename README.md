# whisper-inference

A (in progress) from-scratch C++ inference engine for OpenAI's Whisper, built against a
Python reference implementation. The weight format, the memory layout, and kernels are written and
validated against PyTorch at every stage.

This is my personal project to learn how inference engines actually work, and to build
something tuned for my PC (RTX 3050, 6GB VRAM) rather than for a
generic deployment target.

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

## Stack

- **Python** (PyTorch + openai-whisper) — reference implementation and weight export only,
  not part of the runtime path.
- **C++17**, MSVC — the actual inference engine.
- Target hardware: RTX 3050 / 6GB VRAM, tuned for this machine rather than generic
  deployment.
