# Multi-label + open-set inference design

## 1. Primary tactical detector

The primary network has three independent sigmoid outputs: speech, drone and gunshot. A window can therefore activate zero, one, two or all three outputs. Probabilities do not sum to 100%.

Training uses binary cross entropy and synthetic mixtures of target clips at random -8 to +8 dB SNR. Background clips are negative examples (all-zero target vector) and are also mixed into positives. This is essential: merely changing softmax to sigmoid without mixed training data does not teach simultaneous detection.

## 2. Conditional AudioSet fallback

YAMNet is called only when no tactical target exceeds its class-specific threshold. It produces broad AudioSet guesses such as siren, dog, engine or music. Its top result is emitted as `추정: <label>` only when both conditions pass:

- top score is at least `--yamnet-confidence` (default 0.25)
- top score exceeds the second score by at least 0.05

Otherwise the final result is `UNKNOWN`. YAMNet's label is an estimate, not a verified fact.

## 3. Threshold calibration

The default 0.50 target threshold and 0.25 fallback threshold are starting points only. Record a held-out field set containing isolated events, mixtures, silence and unseen sounds. Select a separate threshold per target using precision/recall requirements, then write the values into the checkpoint's `thresholds` map.

## 4. Raspberry Pi deployment note

TensorFlow Hub can be large for an edge device. First validate correctness. If latency or memory is excessive, convert/freeze YAMNet to TFLite or run the broad fallback less often (for example once per second after several consecutive no-target windows). The tactical CNN remains the fast path.
