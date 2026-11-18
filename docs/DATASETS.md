# Datasets and split policy

Image Detector does not bundle a training corpus. This document lists suitable public sources, the split rules the tooling assumes and the licence questions to settle before publishing a model.

## What the pipeline needs

| Stage | Requirement |
| --- | --- |
| Detection | Real photographs and AI images from several generators |
| Generator-disjoint evaluation | At least three AI generators; each generator kept whole in one split |
| Attribution | At least two AI generators in the training split, at least two images per generator |
| Localisation | Edited images with pixel masks (the `make_edits` helper can synthesise them) |

A useful starting point is 1,000+ images per generator and a comparable number of real photographs, with resolution and content diversity. Small smoke-test corpora are fine for checking the pipeline but say nothing about accuracy.

## Real photographs

| Source | Notes |
| --- | --- |
| [COCO](https://cocodataset.org/) | Scene diversity, annotations CC BY 4.0, individual images carry their original Flickr licences. Check each image's terms before redistribution. |
| [ImageNet](https://www.image-net.org/) | Research use; check the current terms. Strong object diversity, weak scene diversity. |
| [RAISE](https://loki.disi.unitn.it/RAISE/) | Raw camera images, useful for sensor-noise realism. |
| Your own camera photos | Best provenance and easiest consent story. Strip personal metadata before sharing. |

Avoid mixing screenshots, heavily cropped thumbnails and stock composites into the "real" class; they are common shortcut features and inflate accuracy while teaching the detector the wrong cue.

## AI-generated images

| Source | Notes |
| --- | --- |
| [GenImage](https://github.com/GenImage-Dataset/GenImage) | Large corpus spanning several generators; generator labels are suitable for disjoint splits. |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | 32x32 images; quick baseline, but too easy and low-resolution to represent current generators. |
| [DiffusionDB](https://poloclub.github.io/diffusiondb/) | Stable Diffusion prompts and images; useful for SD-family diversity. |
| Locally generated SDXL / SD3 / FLUX | Use `python -m aiforensics.training.generate` or your own pipeline. Document model ids, seeds and prompts. |
| Vendor exports (Midjourney, DALL-E, Imagen, etc.) | Respect each service's terms. Exports may carry C2PA or watermark markers; record that, because the detector must not rely on it. |

Keep the generator name in the manifest. Do not rename several generators to "AI": attribution and generator-disjoint evaluation depend on that column.

## Edited images and masks

| Source | Notes |
| --- | --- |
| [CASIA v2.0](https://github.com/namtpham/casia2groundtruth) | Classical copy-move and splicing masks. Research licence; read and comply with it. |
| [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcgt/) | Small splicing set, useful for sanity checks. |
| Synthetic edits | `python -m aiforensics.training.make_edits` pastes, blurs, shifts colour and adds noise with exact masks. Fast and licence-clean, but distributionally different from real manipulation. |

Masks must align with image pixels. Keep masks in the same split as their source image; never let an edit of a training image appear in test.

## Split policy

1. **Generator-disjoint AI splits.** Whole generators go to train, validation or test. This measures generalisation to unseen generators rather than fingerprints of one model.
2. **Random real splits.** Real photographs are split independently with a fixed seed. Keep enough real images in validation to calibrate the threshold.
3. **No leakage.** Deduplicate exact and near-duplicate images before splitting. Similar frames from one burst can leak across splits if split by image.
4. **Respect existing splits.** If the manifest already has a `split` column, training and evaluation use it unchanged.
5. **Fallback.** With fewer than three generators, `resolve_splits` falls back to a label-stratified random split. Report this explicitly; generator-disjoint claims are not valid in that case.

The splitter is in `src/aiforensics/training/manifest.py`. Splits are written once to the CSV and reused by every stage.

## Augmentation

`src/aiforensics/training/augmentations.py` applies, during training only:

- random crop and resize,
- horizontal flips (classification and mask-aware localisation),
- JPEG recompression at quality 30-100,
- random downscale/upscale,
- Gaussian blur,
- additive Gaussian noise.

Evaluation applies only a deterministic resize. Robustness sweeps in `evaluate.py` apply JPEG q50/q30, 50% resize and blur sigma 2 to the whole engine, including metadata and fusion.

Recompression destroys C2PA/EXIF provenance. Report robustness with and without provenance so the improvement is not mistaken for visual detection.

## Manifest layout

```csv
path,label,generator,split,mask
real/0001.jpg,real,real,train,
sdxl/img_0001.png,ai,SDXL,train,
flux/img_0001.png,ai,FLUX,val,
sd3/img_0001.png,ai,SD3,test,
edits/render-edit00.png,ai,FLUX,train,masks/render-edit00.png
```

Paths are relative to the manifest. `generator` is mandatory for AI rows. The `mask` column is optional and used only by the localiser.

## Licensing and ethics checklist

- Record the source, licence and retrieval date for every corpus in your own dataset card.
- Do not redistribute images whose licences forbid it; share manifests and training scripts instead.
- Do not publish models trained on faces or personal data without consent and an impact review.
- Be aware that public "real" and "AI" corpora contain their own biases: resolution, JPEG history, subject matter and generator version can all become shortcuts.
- If you release an artifact, include the split policy, generator list, metrics and known limitations. Never report a single headline accuracy without the per-generator breakdown.
