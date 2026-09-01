# Grounding the nitrogen-status index

`phase5.py` computes CIre (Chlorophyll Index Red Edge) as a nitrogen-status proxy, on top of
NDVI/NDRE. This note is about why that's a reasonable thing to claim, not just an index that
looks physically plausible.

## The claim

Leaf chlorophyll concentration tracks nitrogen availability closely, and red-edge
reflectance (the steep slope in reflectance around 700-750nm) is strongly affected by
chlorophyll content — more strongly than the red/NIR bands NDVI uses, which saturate over
moderate-to-dense canopy. So red-edge-based indices like NDRE and CIre should track nitrogen
status better than NDVI, especially once a crop's canopy has closed.

## External validation

Castilho, Tedesco, Hernandez, Madari, Ciampitti, "A global dataset for assessing
nitrogen-related plant traits using drone imagery in major field crop species," *Scientific
Data* 11, 585 (2024) — a systematic review aggregating 41 studies and >11,000 field
observations across 11 crop species and 13 countries. Their technical validation reports
NDRE correlating with measured plant nitrogen status at R²=0.80 for wheat and R²=0.67 for
cotton, and finds NDRE outperforms NDVI for this specifically (NDVI's correlation with
nitrogen-related traits ranged R²=0.11-0.65 across crops in the studies they aggregated) —
because NDVI saturates and NDRE doesn't.

## What this does and doesn't establish for this project

This is real field validation that the underlying physical mechanism (red-edge reflectance
tracking nitrogen status) holds up across many independent studies and crop types — it's not
just a textbook formula. It does **not** validate the actual CIre numbers computed in this
repo: Indian Pines has no ground-truth nitrogen measurements to check against, and (per
`report.md`'s limitations) this scene has no radiometric calibration, so the absolute index
values here aren't comparable to literature thresholds — only the spatial pattern within the
scene is trustworthy. The citation grounds *why this is a sensible index to compute at all*,
not *that these specific pixel values are correct nitrogen estimates*.
