# Model card: ForensIQ

## Overview

ForensIQ is an image-forensics decision-support system. It estimates whether an image is AI-generated, highlights regions inconsistent with the surrounding image, and attributes generation to a known source or an explicit unknown class.

| Item | Value |
| --- | --- |
| Version | 0.1.0 (research prototype) |
| Repository | This repository |
| Runtime | Python 3.10+, FastAPI, OpenCV, scikit-learn; optional PyTorch |
| Default detector | Classical heuristic engine (no trained weights) |
| Optional detectors | CLIP + calibrated logistic head; small CNN or ConvNeXt-Tiny |
| Attribution | Metadata markers plus optional open-set CLIP head |
| Localisation | Residual-noise + JPEG error-level heatmap; optional U-Net |
| Licence | MIT (code) |

Trained artifacts are not distributed. Users train their own heads with `src/aiforensics/training/` and supply the images. Consequently, the model card for any deployed instance must record the actual corpora, generator list, split policy and metrics of that instance.

## Intended use

- Assisting researchers, journalists, moderators and reviewers who need an initial, inspectable lead.
- Teaching and demonstrating fusion of classical forensic signals, calibration and provenance.
- Batch triage where a human reviews every flagged image and the report is retained.
- Benchmarking a trained detector on generator-disjoint data.

## Out-of-scope use

- Sole evidence for accusations, disciplinary action, insurance or legal proceedings.
- Fully automated takedowns or content moderation without human review.
- Authenticating identity, ownership or copyright.
- Detecting every manipulation type. The localiser looks for localised inconsistency, not a semantic forgery and not a full forensic reconstruction.
- Verifying C2PA signatures or watermark validity. Markers and claims are read, not cryptographically validated.
- Analysing video, audio or non-image media.
- Images below roughly 64 px on a side, extreme crops, scans or heavily composited graphics, where the signals are not meaningful.

## How it works

1. Decode and normalise the image; apply EXIF orientation; cap analysis size at 1024 px.
2. Extract EXIF, PNG text, XMP, C2PA/JUMBF markers, JPEG quality estimate and watermark markers.
3. Compute seven classical signals: noise-field heterogeneity, detail distribution, residual noise floor, radial spectral decay, high-frequency balance, colour signature and provenance.
4. Optionally run a trained detector and a trained attribution head on CLIP features.
5. Fuse the trained probability with classical evidence (75% model weight when available), map to confidence and an uncertain band.
6. Build a localisation heatmap from noise and recompression inconsistency, or a U-Net mask when trained.
7. Render heatmap, boxes, spectrum and optional Grad-CAM; attach notes and limitations.

The default heuristic probability is deliberately conservative and uses the image's own median as a reference, so a globally uniform image is not turned into a false hotspot. The probability is an indicator, not a calibrated frequency.

## Training data

No data is bundled. Recommended public sources and licences are listed in [DATASETS.md](DATASETS.md). Any trained artifact must document:

- real-image sources and licences,
- generator names, model ids, versions, seeds and prompts,
- number of images per generator and per split,
- generator-disjoint split assignment,
- post-processing and augmentation,
- deduplication method,
- consent and personal-data handling.

## Evaluation

The repository ships evaluation tooling, not a benchmark claim. `python -m aiforensics.training.evaluate` reports:

- accuracy, precision, recall, specificity, F1, balanced accuracy, ROC-AUC,
- Brier score and expected calibration error,
- confusion matrices,
- per-generator detection rates,
- attribution top-1 accuracy, accepted rate and unknown-generator rejection,
- edit localisation IoU and Dice when a U-Net and masks are available,
- robustness under JPEG q50/q30, 50% resize and blur sigma 2.

The demo figures in the README describe the bundled examples with the heuristic engine. They are not a representative evaluation and must not be quoted as model accuracy.

## Known limitations

- **Distribution shift.** Generators evolve. A head trained on SDXL, SD3 and FLUX may fail on a new model or on images passed through an unseen pipeline.
- **Post-processing.** JPEG, resizing, blur, denoising, screenshots and social platforms erase the cues the detector uses. Provenance metadata is usually the first casualty.
- **Metadata spoofing.** EXIF, PNG text and even C2PA markers can be forged or transplanted. The tool never verifies signatures and labels metadata as spoofable.
- **Classical signal bias.** Noise and frequency statistics depend on camera, ISO, scene texture, denoising and editing software. Low-light or denoised photographs can resemble renders.
- **Localisation.** The heatmap highlights inconsistency, not intent or semantics. A fully synthetic image can show hotspots without any edit, and a subtle edit can be missed. The U-Net, when used, is only as good as its masks.
- **Attribution.** The open-set threshold trades accepted coverage against false names. An "unknown generator" result is informative; a named result is not a certainty.
- **Calibration.** Platt scaling assumes the validation distribution resembles deployment. Without a validation split, thresholds are defaults or test-tuned and metrics become optimistic.
- **Resolution and content.** Small images, text documents, illustrations, heavy filters and composites are outside the validated regime.
- **Privacy.** Images are processed in memory with a bounded in-process cache, but the service cannot control proxy logs, browser caches or the user's own storage.

## Ethical considerations

- Present results with uncertainty and the limitations above. The UI and API return the disclaimer, reliability and notes for this reason.
- Never automate punitive action from a single score.
- Obtain consent and an impact review before processing personal or sensitive images; avoid retaining uploads.
- Document dataset bias. Generator families, subject matter and resolution rarely match deployment traffic.
- Make provenance evidence visible rather than silently trusting it, and distinguish metadata claims from visual evidence.
- Provide a route for appeal and human override wherever the tool informs a decision.

## Maintenance

- Re-evaluate on newly released generators before deployment and after any model update.
- Version every artifact and record its training manifest hash.
- Monitor false-positive rates on real traffic; recalibrate thresholds when the distribution moves.
- Treat reported confidence as a ranking aid, not a probability guarantee.
