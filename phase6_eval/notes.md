# Post-classification majority filter (`majority_filter.py`)

Phase 6's own write-up already named the problem: the classification map has visible
pixel-level "speckle" inside otherwise-uniform fields, because each pixel is classified
independently with no spatial context. This is a standard fix for that -- reclassify each
pixel as the most common prediction in its 3x3 neighborhood (background excluded from the
vote) -- applied after the fact to the existing prediction map, no retraining needed.

## Result

| | Full image | Held-out test set |
|---|---|---|
| Before filter | 95.24% | 92.93% |
| After 3x3 majority filter | 98.59% | **98.59%** |

Visually (`majority_filter_comparison.png`), the speckle inside solid fields is essentially
gone after filtering, and the map looks close to the ground truth. This matches the standard
finding in the hyperspectral literature that spatial post-processing meaningfully improves
on pixel-independent classification.

## Caveat

This test-set number is connected to what `spatial_holdout/` already flagged. The majority
filter pulls each pixel's label from its *spatial neighbors* -- and because the underlying
split is pixel-random, a test pixel's neighbors are very likely training pixels the model has
already fit closely. So part of this gain is error-correction (isolated bad predictions
getting outvoted by a locally consistent neighborhood), and part of it is the test pixel
picking up its neighbors' already-memorized labels -- the same leakage mechanism as the
pixel-random split itself, working through the filter instead of through training directly.

The filter is a standard technique and the visual cleanup is real. But the 98.6% number
should be read as "spatial smoothing helps, especially with this split," not as a
leakage-free accuracy figure. A standalone test would be applying this same filter to the
spatial-holdout predictions from `spatial_holdout/` instead, where neighbors are much less
likely to already be memorized -- not done here for time.
