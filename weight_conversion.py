import struct
import whisper

MODEL = "small"
OUT = f"whisper_{MODEL}.bin"

# Every payload starts on a multiple of this. 64 keeps SIMD loads and cache
# lines happy, not just the 4-byte minimum a float needs.
ALIGN = 64

# Weights ship as float16 and we store them as float16. Verified lossless:
# fp32 -> fp16 changes 0 of 479 tensors, because they were fp16 to begin with.
DTYPE = "float16"

# Load the model FIRST so the header can be written from the real dims
# instead of hardcoded numbers that silently go stale if MODEL changes.
model = whisper.load_model(MODEL, device="cpu")
dims = model.dims
state_dict = model.state_dict()  # a dict: name (string) -> tensor

with open(OUT, "wb") as f:
    # Write each config value as a fixed-size integer.
    # '<i' means little-endian, 4-byte signed integer — pick one convention and stick to it.
    f.write(b"WSPR")                                    # magic, so the reader can sanity-check the file
    f.write(struct.pack("<i", 2))                       # format version (2 = fp16 + aligned payloads)
    f.write(struct.pack("<i", dims.n_text_layer))       # n_layers
    f.write(struct.pack("<i", dims.n_text_head))        # n_heads
    f.write(struct.pack("<i", dims.n_text_state))       # d_model
    f.write(struct.pack("<i", dims.n_mels))             # n_mels
    f.write(struct.pack("<i", dims.n_vocab))            # vocab_size
    f.write(struct.pack("<i", len(state_dict)))         # how many tensors follow
    f.write(struct.pack("<i", ALIGN))                   # so the reader need not hardcode it

    padding = 0
    for name, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy().astype(DTYPE)

        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<i", len(name_bytes)))  # how many bytes the name takes
        f.write(name_bytes)                          # the name itself, e.g. "encoder.blocks.0.attn.query.weight"

        f.write(struct.pack("<i", arr.ndim))          # how many dimensions (e.g. 2 for a matrix)
        for dim in arr.shape:
            f.write(struct.pack("<i", dim))           # each dimension's size

        # Names are variable length, so the cursor drifts off the grid. Push it
        # back before the payload, or the reader cannot cast to a typed pointer.
        pad = -f.tell() % ALIGN
        f.write(b"\x00" * pad)
        padding += pad

        f.write(arr.tobytes())

print(f"wrote {OUT}: {len(state_dict)} tensors, {DTYPE}, {padding} bytes of padding")
