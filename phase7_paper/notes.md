# Phase 7 — Paper reproduction

**Citation:** Ustuner, M. "Randomized Principal Component Analysis for Hyperspectral
Image Classification." *2024 IEEE Mediterranean and Middle-East Geoscience and Remote
Sensing Symposium (M2GARSS)*, Oran, Algeria, 2024.

## What I reproduced

The paper compares standard PCA against randomized PCA as a preprocessing step before
classification (SVM and gradient boosting) on Indian Pines and Pavia University,
reducing the feature space to 20 and 30 principal components. Its stated finding is
that standard PCA outperformed randomized PCA for SVM on both datasets — randomized
PCA is an approximation method that trades some accuracy for speed on very large
feature sets, and the paper is arguing that trade-off isn't worth it here.

I reproduced the piece of the protocol I could fully specify: standard PCA at 20 and
30 components, feeding an RBF-kernel SVM, on Indian Pines, using the same 80/20
stratified train/test split as my own Phase 3 baseline. Results (`pca_svm_comparison.png`):

| Features | Test accuracy |
|---|---|
| 200 bands, no PCA | 0.820 |
| 20 PCA components (97.6% variance explained) | 0.762 |
| 30 PCA components (98.6% variance explained) | 0.781 |

## Critique

The paper's abstract states the qualitative finding (standard PCA beats randomized
PCA for SVM) but doesn't report the actual PCA+SVM accuracy numbers themselves in the
material I could access — only the comparative claim and, separately, that the best
overall Indian Pines result in the paper (0.964) came from LightGBM on the *original*,
non-reduced features, not from any PCA-reduced configuration. That's a useful, honest
gap: I can't directly check whether my 0.762/0.781 numbers match theirs, because they
weren't stated at the level of detail I had access to. My numbers are still meaningful
on their own terms, though: they show that compressing 200 spectral bands down to 20-30
components keeps ~98% of the variance but costs roughly 4-6 points of test accuracy
relative to using all 200 bands directly — a concrete, reproducible illustration of
the tension between dimensionality reduction and information loss that the paper is
built around.

Two likely reasons my numbers would diverge from a full reproduction even if I had the
exact target values: (1) SVM hyperparameters (C, gamma) — I used scikit-learn defaults
for both, matching my own Phase 3 baseline, rather than tuning per feature set, and the
paper doesn't state whether/how it tuned; (2) train/test split — random splits on this
dataset are not fully reproducible pixel-for-pixel across papers unless the exact same
split methodology and random seed are used, so small accuracy differences between two
"honest" reproductions are expected and don't necessarily indicate an error in either.

**What I'd ask the authors:** what SVM hyperparameters (C, gamma, kernel) were used for
each feature configuration, and were they tuned separately per configuration or fixed
across all of them? That single detail would explain a meaningful chunk of any
accuracy gap between my reproduction and theirs.

## Why this choice (efficiency note)

I picked a PCA+SVM protocol specifically because it does double duty for this project:
it's Phase 7's literature reproduction, it's the bootcamp's PCA stretch goal, and the
20-30-component PCA-reduced feature representation it produces is exactly the kind of
compact input a variational quantum classifier needs (see `quantum_extension/` —
quantum circuits are practically limited to a handful of qubits, so PCA-reduced
features are the standard on-ramp in that literature, not just a convenience here).
