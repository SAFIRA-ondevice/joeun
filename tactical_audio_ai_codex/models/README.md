# Model artifacts

No model binary was available in the mounted files when this package was built.

Expected Raspberry Pi files:

- `/home/pi/tactical_audio_ai/models/audio_cnn_best.pt`: original contaminated 3-class checkpoint; preserve only for comparison.
- `/home/pi/tactical_audio_ai/models/audio_cnn_4class_best.pt`: recommended rebuilt checkpoint produced by `training/train_audio_cnn.py`.

Do not create an empty `.pt` placeholder: it can be mistaken for a valid checkpoint. `MODEL_MANIFEST.json` records the missing state explicitly.
