# Dataset summary and label policy

- Speech: AI Hub Korean noisy command speech subset; reported 3,398 `-S.wav` speech files. Confirm license before redistribution.
- Drone: DroneAudioDataset; reported 23,408 WAV, including 1,332 `yes_drone`. Remaining 22,076 are candidates for background only after checking labels and duplicates.
- Gunshot: MAD contains seven classes. Use its annotation file and select only the exact `gunshot` label. Never infer label from “military” membership.
- Background: silence plus verified non-target indoor noise, wind, fans, vehicles, motors and non-drone examples. Do not include files containing target events.

Split before augmentation. If clips originate from longer recordings, group by source recording before splitting to prevent leakage. The included simple splitter operates at file level; extend it with group metadata when source relationships are known.
