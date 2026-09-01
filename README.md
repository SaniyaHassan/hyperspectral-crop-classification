# Hyperspectral ML Pipeline

An end-to-end pipeline for classifying land cover pixel-by-pixel from a hyperspectral
image, built up through a series of stages from data loading to a full evaluation, a
literature reproduction, and a quantum machine learning extension.

Full write-up: [`report.md`](report.md). Full phase-by-phase notes and reasoning:
[`LEARNING_LOG.md`](LEARNING_LOG.md).

## Dataset

[Indian Pines](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
(AVIRIS), 145×145 pixels, 200 usable spectral bands, 16 land-cover classes.

## What's here

| Stage | What it does | Key result |
|---|---|---|
| [`phase0_setup/`](phase0_setup) | Load the cube, plot one pixel's spectrum | — |
| [`phase1_fundamentals/`](phase1_fundamentals) | Overfit/underfit refresher on a toy dataset | — |
| [`phase2_data/`](phase2_data) | Per-class mean spectra, ground-truth map | 16 class spectral signatures |
| [`phase3_svm/`](phase3_svm) | Baseline SVM classifier | 82.0% test accuracy |
| [`phase4_mlp/`](phase4_mlp) | Hand-built PyTorch MLP, regularization comparison | **92.9% test accuracy** |
| [`phase5_indices/`](phase5_indices) | NDVI, NDRE, and a nitrogen-status index | see `phase5_indices/notes.md` |
| [`phase6_eval/`](phase6_eval) | Full classification map, per-class precision/recall, + a post-classification majority filter | macro F1 0.939, filter 92.9% → 98.6% |
| [`phase7_paper/`](phase7_paper) | Reproduces a PCA+SVM protocol from published literature | see `phase7_paper/notes.md` |
| [`quantum_extension/`](quantum_extension) | Variational quantum classifier vs. classical SVM | 56.0% vs 78.0% test accuracy |
| [`spatial_holdout/`](spatial_holdout) | Re-evaluates the MLP on a spatially disjoint split | 92.9% → 42.9% test accuracy |

## Running it

```
pip install -r requirements.txt
```

Each stage is a standalone script that loads `data/Indian_pines_corrected.mat` and
`data/Indian_pines_gt.mat` (not committed, see below) and writes its own artifacts:

```
python phase4_mlp/phase4.py
python quantum_extension/quantum_classifier.py   # slow: ~5 min on CPU, see notes.md
```

**Data**: download `Indian_pines_corrected.mat` and `Indian_pines_gt.mat` from the
[EHU/GIC hyperspectral scenes page](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
into `data/` (gitignored, not committed).

## Limitations

The train/test split used in phases 0-7 is pixel-random rather than spatially held-out, so
those accuracies likely overstate performance on a genuinely unseen field --
[`spatial_holdout/`](spatial_holdout) measures the actual size of that gap (92.9% → 42.9%).
The vegetation-index wavelength mapping is derived rather than taken from an exact sensor
calibration file, since none is publicly available for this scene. Full discussion in
`report.md`.
