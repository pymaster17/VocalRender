"""GPU smoke test: load the default checkpoint, generate, switch checkpoint, generate again.

Run on a GPU node (not the head node):
    python tests/gpu_smoke.py [out_dir]
"""
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app  # noqa: E402  (loads the default checkpoint)

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/smoke")
out_dir.mkdir(parents=True, exist_ok=True)


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 2**20


def run(variant, preset_index):
    preset = app.SCORE_PRESETS[preset_index]
    value = app.score_value(app._preset_to_rows(preset), preset["bpm"])
    t0 = time.perf_counter()
    wav, status, prompt = app.generate_from_score(value, variant, "Alto-1", None, 2.0, 10, 1.0, 2000)
    elapsed = time.perf_counter() - t0
    assert wav, f"{variant}: no audio returned ({status})"
    audio, sr = sf.read(wav)
    duration = len(audio) / sr
    target = out_dir / f"{variant}_{preset['id']}.wav"
    sf.write(target, audio, sr)
    print(f"[smoke] {variant}: preset {preset['id']} ({len(preset['pitches'])} events, {preset['bpm']} BPM) "
          f"-> {duration:.2f}s @ {sr} Hz in {elapsed:.1f}s, GPU {gpu_mb():.0f} MiB, saved {target}")
    assert duration > 1.0, f"{variant}: audio too short ({duration:.2f}s)"
    assert abs(audio).max() > 0.01, f"{variant}: audio is silent"
    return duration


print(f"[smoke] resident after startup: {app.models.variant}, GPU {gpu_mb():.0f} MiB")
assert app.models.variant == app.DEFAULT_CKPT_VARIANT
first_variant_mb = gpu_mb()
run(app.DEFAULT_CKPT_VARIANT, 0)

others = [v for v in app.CKPT_VARIANTS if v != app.DEFAULT_CKPT_VARIANT]
if others:
    other = others[0]
    msg = app.switch_checkpoint(other).constructor_args["value"]
    print(f"[smoke] switch -> {msg}; resident={app.models.variant}, GPU {gpu_mb():.0f} MiB")
    assert app.models.variant == other
    after_switch_mb = gpu_mb()
    assert after_switch_mb < first_variant_mb * 1.5, (
        f"GPU memory did not drop after switching: {first_variant_mb:.0f} -> {after_switch_mb:.0f} MiB"
    )
    run(other, 1)
    # Switch back to make sure the cache round-trips.
    app.models.get(app.DEFAULT_CKPT_VARIANT)
    assert app.models.variant == app.DEFAULT_CKPT_VARIANT
    print(f"[smoke] switched back to {app.models.variant}, GPU {gpu_mb():.0f} MiB")

print("[smoke] OK")
