# Spatial holdout: fixing the biggest limitation

Every other stage in this project splits pixels into train/test randomly. That means a test
pixel can sit two rows away from a training pixel inside the exact same field, so the model
isn't really being asked to generalize to unseen land -- it's mostly memorizing each field's
spectral signature and recognizing more of the same field. This was flagged as the top
limitation in `report.md` from the start. This script fixes it and reports what actually
happens.

Disjoint/spatial sampling as a specific concern for Indian Pines is discussed in Ahmad,
Mazzara, Distefano, "Importance of Disjoint Sampling in Conventional and Transformer Models
for Hyperspectral Image Classification" (arXiv:2404.14944, 2024).

## Method

For each of the 16 classes, find its separate contiguous field patches (connected components
of that class's mask, 8-connectivity) using `scipy.ndimage.label`. Assign whole patches to
train or test -- never individual pixels -- greedily until roughly 20% of that class's pixels
are in test, matching the 80/20 split used everywhere else in this project.

Classes that only occupy a single contiguous patch (7 of the 16: Alfalfa, Corn,
Grass-pasture-mowed, Hay-windrowed, Oats, Wheat, Stone-Steel-Towers) can't be split this way
without still leaving train and test touching along the cut. These are kept entirely in
train and excluded from the test-set accuracy below -- a real limitation of disjoint sampling
itself for small single-field classes, not something papered over here.

Same model, same hyperparameters as Phase 4's winning MLP (200-128-64-16, dropout 0.3,
weight decay 1e-4, 1000 epochs) -- the only thing that changed is the split.

## Result

| Split | Test accuracy | Classes evaluated |
|---|---|---|
| Pixel-random (Phase 4) | 92.9% | 16 / 16 |
| Spatial holdout (this script) | **42.9%** | 9 / 16 |

Accuracy roughly halves. The two numbers aren't directly comparable (different classes,
different pixel counts), but the direction and size of the drop shows that most of Phase 4's
reported accuracy was coming from the model having already seen each field's spectral
signature, not from learning to distinguish crop types independent of location.

`spatial_split_map.png` shows which patches went where. `split_comparison.png` is the bar
chart above.

Per-class breakdown (`spatial_holdout.py` output) shows the drop isn't uniform: Grass-trees
(88% F1) and Woods (60% F1) hold up reasonably, since they're visually/spectrally distinct
even in the classification map from Phase 6. Grass-pasture (0% F1) and Soybean-notill (3% F1)
collapse -- both are the row-crop and pasture classes that were already the most confused in
Phase 3's confusion matrix, and are exactly the ones most confusable with a neighboring class
once the model can't lean on having seen the specific field.

## What this means for the rest of the project

The 92.9% number in `report.md` and the classification map in `phase6_eval/` are left as-is --
they're accurate descriptions of what a pixel-random evaluation shows, and changing them
after the fact would remove the point of this comparison. 92.9% is the result under an
evaluation protocol with known leakage; 42.9% is a first estimate of performance on fields
the model hasn't seen, on a harder 9-class subset.

## Next steps

- Apply the same spatial-holdout evaluation to Phase 7's PCA+SVM protocol, to see whether
  dimensionality reduction is more or less sensitive to this than the MLP.
- The 7 excluded single-patch classes are exactly the smallest classes in the dataset. A
  larger hyperspectral benchmark with more per-class fields would let all 16 classes get a
  fair spatial-holdout evaluation instead of losing nearly half of them.
