# Artifact inventory

Inspection performed on 2026-08-23.

- Current workspace: no pre-existing project files beyond empty output/work directories.
- Mounted Codex attachments: no `Soohyun-main.zip` and no `.pt`, `.pth`, `.onnx`, or `.tflite` artifact was available.
- Referenced conversation: confirms the prior filenames and Raspberry Pi paths but does not expose their file contents as mounted artifacts.
- Actual model binaries included: none.
- Replacement code in this package: newly reconstructed, configurable implementation based on the documented M3C0 protocol and pipeline.

Files to recover directly from the Raspberry Pi if they exist:

```text
/home/pi/joeun/receive_m3c0_wav.py
/home/pi/joeun/live_m3c0_ai.py
/home/pi/tactical_audio_ai/scripts/split_dataset.py
/home/pi/tactical_audio_ai/scripts/train_audio_cnn.py
/home/pi/tactical_audio_ai/models/audio_cnn_best.pt
```

Compare recovered source instead of overwriting it. Treat `audio_cnn_best.pt` as contaminated unless its dataset manifest proves it used gunshot-only MAD labels.
