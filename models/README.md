# Model artifacts

This directory is where ForensIQ looks for trained artifacts. The files are
ignored by git because they are large and dataset-dependent.

| File | Produced by | Purpose |
| --- | --- | --- |
| `detector.joblib` | `python -m aiforensics.training.train_detector` | CLIP features + calibrated logistic detector |
| `attribution.joblib` | `python -m aiforensics.training.train_attribution` | Open-set generator attribution |
| `cnn_detector.pt` | `python -m aiforensics.training.train_cnn` | CNN detector with Grad-CAM |
| `localizer.pt` | `python -m aiforensics.training.train_localizer` | U-Net edit localiser |

Use `python -m aiforensics info` to confirm which artifacts the engine loaded.
Set `FORENSIQ_MODEL_DIR` or the per-artifact environment variables to point
somewhere else. Missing artifacts are not an error: the engine falls back to the
classical pipeline and records that in the report.
