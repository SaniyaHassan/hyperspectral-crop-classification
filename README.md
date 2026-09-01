# Hyperspectral ML Pipeline

A pipeline for pixel-level land-cover classification from hyperspectral imagery, covering
data loading, classical machine learning, a hand-built neural network, vegetation-index
computation, a literature reproduction, and a quantum machine learning classifier. Hyperspectral
classification and vegetation/nitrogen indices of this kind are used in remote sensing and
precision-agriculture applications (crop monitoring, canopy nitrogen status, land-cover
mapping); this project builds and evaluates that pipeline end to end on a benchmark scene.

Full write-up: [`report.md`](report.md). Full phase-by-phase notes and reasoning:
[`LEARNING_LOG.md`](LEARNING_LOG.md).

## Dataset

[Indian Pines](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
(AVIRIS), 145×145 pixels, 200 usable spectral bands, 16 land-cover classes (mostly row crops).

## What's here

| Stage | What it does | Result |
|---|---|---|
| [`phase0_setup/`](phase0_setup) | Load the cube, plot one pixel's spectrum | — |
| [`phase1_fundamentals/`](phase1_fundamentals) | Overfit/underfit refresher on a toy dataset | — |
| [`phase2_data/`](phase2_data) | Per-class mean spectra, ground-truth map | 16 class spectral signatures |
| [`phase3_svm/`](phase3_svm) | Baseline SVM classifier on the full spectrum | 82.0% test accuracy |
| [`phase4_mlp/`](phase4_mlp) | Hand-built PyTorch MLP, regularization comparison | 92.9% test accuracy |
| [`phase5_indices/`](phase5_indices) | NDVI, NDRE, and a nitrogen-status index (CIre) | see `phase5_indices/notes.md` |
| [`phase6_eval/`](phase6_eval) | Full classification map, per-class precision/recall, post-classification majority filter | macro F1 0.939, filter 92.9% → 98.6% |
| [`phase7_paper/`](phase7_paper) | Reproduces a PCA + SVM classification protocol from published literature | see `phase7_paper/notes.md` |
| [`spatial_holdout/`](spatial_holdout) | Re-evaluates the MLP on a spatially disjoint train/test split | 92.9% → 42.9% test accuracy |
| [`quantum_extension/`](quantum_extension) | Variational quantum classifier vs. classical SVM on PCA-reduced features | 56.0% → 72.5% quantum vs. 78-80% classical |

## Research questions

Beyond the core classification pipeline, a few specific questions were tested directly:

- **Does a pixel-random train/test split overstate accuracy on this dataset?** Yes —
  re-evaluating the MLP on a spatially disjoint split (whole field patches held out, not
  individual pixels) drops test accuracy from 92.9% to 42.9%. `spatial_holdout/`
- **Does post-classification spatial filtering correct the pixel-level noise in the
  classification map?** Yes — a 3×3 majority filter raises held-out test accuracy from 92.9%
  to 98.6%, though part of that gain overlaps with the split-leakage finding above.
  `phase6_eval/notes.md`
- **Can qubit count and ansatz choice close the gap between a variational quantum classifier
  and a classical baseline on the same features?** Partially — going from 4 qubits with a
  basic entangling ansatz to 6 qubits with `StronglyEntanglingLayers` narrows the gap from 22
  points to 7.5 points (56.0% → 72.5% quantum vs. 78-80% classical). `quantum_extension/notes.md`
- **Is a red-edge vegetation index a reasonable nitrogen-status proxy for this kind of data?**
  The index computed here (CIre) is grounded in published field studies reporting NDRE
  correlating with measured crop nitrogen status at R²=0.80 (wheat) and R²=0.67 (cotton).
  `phase5_indices/notes.md`

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
those accuracies overstate performance on a field the model hasn't seen —
[`spatial_holdout/`](spatial_holdout) measures the size of that gap. The vegetation-index
wavelength mapping is derived rather than taken from an exact sensor calibration file, since
none is publicly available for this scene. Full discussion in `report.md`.
