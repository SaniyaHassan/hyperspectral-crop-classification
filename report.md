# Hyperspectral Land-Cover Classification: A Pipeline from Spectrum to Map

## Problem

Hyperspectral imaging captures a full reflectance spectrum (here, 200 usable bands
spanning ~400-2500nm) at every pixel, instead of the three broad Red/Green/Blue values a
normal camera captures. That extra spectral resolution lets different land-cover types —
crops, grass, soil, buildings — be told apart by *how* they reflect light across the
spectrum, not just by color or texture. This project builds and evaluates a full pipeline
for classifying land cover pixel-by-pixel from a hyperspectral cube, using the Indian
Pines benchmark scene as a stand-in for the kind of drone/field hyperspectral data a
remote-sensing lab works with directly.

## Data

**Indian Pines**: an AVIRIS airborne hyperspectral scene over agricultural land in
Indiana, USA (June 1992). 145×145 pixels, 200 usable spectral bands after removing 20
water-absorption bands, 16 labeled land-cover classes (mostly row crops — corn, soybean —
plus grass, woods, and built structures), with a large fraction of pixels unlabeled
background. Class sizes are highly imbalanced (from 20 pixels for Oats to 2455 for
Soybean-mintill).

## Method

1. **Baseline (SVM)**: every labeled pixel treated as one 200-dimensional feature vector
   (its spectrum), standardized, classified with an RBF-kernel SVM, default hyperparameters.
2. **Neural network (MLP)**: a hand-built PyTorch MLP (200 → 128 → 64 → 16), trained with a
   manual forward/loss/backward/optimizer loop, compared with and without dropout + weight
   decay regularization.
3. **Vegetation/nitrogen indices**: NDVI, NDRE, and a chlorophyll red-edge index (CIre)
   computed per pixel from a derived band-to-wavelength mapping (see limitations below).
4. **Full evaluation**: the best model run over every pixel to produce a complete
   classification map, with per-class precision/recall rather than a single accuracy figure.
5. **Literature reproduction**: PCA-based dimensionality reduction (20/30 components) +
   SVM, reproducing the protocol from Ustuner (M2GARSS 2024).

## Results

| Model | Test accuracy |
|---|---|
| SVM, 200 bands, default hyperparameters | 82.0% |
| MLP, no regularization (overfit by epoch ~1000) | 91.8% |
| **MLP, dropout + weight decay** | **92.9%** |
| SVM, 20 PCA components | 76.2% |
| SVM, 30 PCA components | 78.1% |

The regularized MLP is the best model and is what's run over the full image
(`phase6_eval/classification_map.png`). Per-class evaluation
(`phase6_eval/classification_report.txt`) shows a macro-average F1 of 0.939, with the
worst-performing class being Buildings-Grass-Trees-Drives (recall 0.805) — a literal
spectral-mixture class, confused mostly with pure Woods, consistent with the spectral
similarity already visible in Phase 2's class-mean spectra and Phase 3's SVM confusion
matrix. The classification map also shows visible pixel-level "speckle" inside otherwise
uniform fields — the expected signature of classifying each pixel independently with no
spatial context.

NDVI/NDRE (`phase5_indices/`) correctly separate established vegetation (Woods,
Grass-pasture) from non-vegetated surfaces (Stone-Steel-Towers), but row crops
(Corn/Soybean) show surprisingly *low* NDVI — consistent with this scene being flown in
June, before canopy closure, so row-crop pixels are still a soil/plant mixture.

## Limitations

- **Train/test split is pixel-random, not spatially block-held-out.** Train and test
  pixels can sit a few pixels apart inside the same field, so the model has effectively
  already seen the spectral signature of every field it's "tested" on. Reported accuracies
  are likely optimistic relative to performance on a genuinely unseen field.
- **No radiometric calibration.** The cube's digital numbers are only proportional to true
  surface reflectance, not equal to it — there's no atmospheric correction or reference
  panel in this benchmark. Absolute index values (NDVI, NDRE, CIre) shouldn't be compared
  across scenes or to literature thresholds; the spatial *pattern* within this one scene is
  trustworthy, the absolute numbers are not.
- **Band-to-wavelength mapping is an approximation.** No calibration file for this specific
  1992 flight is publicly available; the wavelength mapping used for the indices assumes
  uniform spacing across AVIRIS's documented range, cross-checked (not proven) against the
  known false-color band convention for this scene.
- **Hyperparameters are largely untuned** (SVM defaults, one MLP architecture) — the
  comparisons here are "reasonable effort" baselines, not each model's ceiling.

## Bridge: what changes with real MachVIS drone/field data

This benchmark is a convenient stand-in, but real MachVIS data differs in several
concrete ways that would change this pipeline:

- **Radiometric calibration and reference panels**: field cubes need a calibration panel
  of known reflectance in-frame (or a dark/white reference capture) to convert raw digital
  numbers to true reflectance — without it, absolute index values (as seen above) aren't
  trustworthy, only spatial patterns are.
- **Ground sampling distance (GSD) and orthomosaicking**: drone cubes are captured as many
  overlapping frames at a much finer GSD than Indian Pines' 20m pixels, and need
  orthomosaicking (geometric correction + stitching) before per-pixel classification makes
  sense at all — a step this benchmark doesn't require since it's already one rectified
  scene.
- **Narrower wavelength range**: MachVIS's field cubes cover roughly 665-975nm (red through
  NIR), not this benchmark's full 400-2500nm. That rules out anything needing blue, green,
  or SWIR bands directly — NDVI and NDRE both still work (their bands fall inside
  665-975nm), but the CIre-style chlorophyll index built here would need re-deriving with
  bands actually available in that narrower range.
- **Ground-truth collection**: Indian Pines' labels come from a fixed historical survey;
  real field ground truth means physically walking/GPS-tagging plots, which is slower, more
  expensive, and a much stronger practical argument for needing indices/models that
  generalize well from limited labeled data.
- **Time series over a season**: this project treats Indian Pines as one static snapshot.
  Real MachVIS work tracks a field across a growing season, which is exactly where an
  index like NDRE (sensitive even after NDVI saturates) becomes more valuable than a single
  classification map.

## Extensions

Two pieces of this project go beyond the core pipeline:

**Quantum classifier.** `quantum_extension/` contains a real PennyLane variational quantum
classifier trained on PCA-reduced features from this cube, compared against a classical SVM
on the identical features (56.0% quantum vs. 78.0% classical test accuracy, ~56,000x slower
to train on a CPU simulator). The result isn't flattering, and it's reported as-is — a real,
reproducible demonstration of where a small hand-built variational circuit currently stands
relative to a classical baseline, and of the practical simulation-speed wall that motivates
using real quantum hardware or better simulation tooling. Full breakdown in
`quantum_extension/notes.md`.

**Nitrogen-status index.** Phase 5 computes NDVI and NDRE, then extends to CIre (Chlorophyll
Index Red Edge), a standard nitrogen-status proxy in precision-agriculture literature
(`phase5_indices/ci_red_edge_map.png`). The per-class CIre ordering matches NDVI/NDRE's
physically sensible pattern (Woods highest, Stone-Steel-Towers negative). The same
calibration caveat from the limitations section applies here: only the spatial pattern is
trustworthy, not the absolute value. A natural next step would be applying this same index
pipeline to a real multi-flight time series rather than one static scene.
