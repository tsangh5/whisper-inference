import whisper
import torch
import os
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="path to a fixed audio file (wav/mp3/flac...)")
    ap.add_argument("--model", default="small", help="tiny/base/small/...")
    ap.add_argument("--out", default="reference_tensors", help="output directory for .npy files")
    args = ap.parse_args()
    outdir = os.path.join(args.out, args.model)
    os.makedirs(outdir, exist_ok=True)

    def save(name, tensor):
            """Detach, move to CPU, cast to float32, write one .npy."""
            arr = tensor.detach().to(torch.float32).cpu().numpy()
            path = os.path.join(outdir, name + ".npy")
            np.save(path, arr)
            print(f"  saved {name:28s} shape={tuple(arr.shape)}")

    print(f"loading whisper '{args.model}' on CPU/float32")
    model = whisper.load_model(args.model, device="cpu")
    model.eval()    # switch to inference mode

    # converting audio to log mel spectrogram
    audio = whisper.load_audio(args.audio)
    audio = whisper.pad_or_trim(audio)
    n_mels = model.dims.n_mels  # 80 for tiny/base/small
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)  # (n_mels, 3000)
    save("00_mel", mel)

    # register hooks to capture every submodule output during the pass
    handles = []
    captured = {}

    def hook(tag):
        def fn(_module, _inp, out):
            # some modules return tuples; take the first tensor
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[tag] = t
        return fn

    enc = model.encoder
    dec = model.decoder
    handles.append(enc.conv1.register_forward_hook(hook("01_enc_conv1")))
    handles.append(enc.conv2.register_forward_hook(hook("02_enc_conv2")))
    for i, block in enumerate(enc.blocks):
        handles.append(block.register_forward_hook(hook(f"03_enc_block{i:02d}")))
    handles.append(enc.ln_post.register_forward_hook(hook("04_enc_output")))
    for i, block in enumerate(dec.blocks):
        handles.append(block.register_forward_hook(hook(f"05_dec_block{i:02d}")))
    handles.append(dec.ln.register_forward_hook(hook("06_dec_ln")))

    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=model.is_multilingual,
        num_languages=getattr(model.dims, "n_vocab", None) and model.num_languages
        if hasattr(model, "num_languages") else None,
        language="en",
        task="transcribe",
    )

    # SOT is start of transcript, tokens we use to seed the input
    sot_sequence = list(tokenizer.sot_sequence_including_notimestamps)
    print(f"decoder prompt token ids: {sot_sequence}")
    tokens = torch.tensor([sot_sequence], dtype=torch.long)

    # one encoder then decoder forward pass
    with torch.no_grad():
        mel_batched = mel.unsqueeze(0)               # (1, n_mels, 3000)
        audio_features = model.encoder(mel_batched)  # fires encoder hooks
        logits = model.decoder(tokens, audio_features)  # fires decoder hooks
    save("07_logits", logits)

    # save hook outputs
    for tag in sorted(captured):
        save(tag, captured[tag])
 
    for h in handles:
        h.remove()

    # save seeding tokens, we will use them for c++
    np.save(os.path.join(outdir, "prompt_tokens.npy"),
            np.array(sot_sequence, dtype=np.int64))

    print(f"\ndone. {len(os.listdir(outdir))} files in {outdir}/")
    print("next: run your C++ on the SAME audio, dump matching tensors,")
    print("      then compare with np.allclose(mine, ref, atol=1e-4).")
 
 
if __name__ == "__main__":
    main()