from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Standard Indian Pines class names (16 classes), for reference when
# interpreting which classes line up with high-NDVI regions.
CLASS_NAMES = [
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
    "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
    "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
    "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
]

# ======================================
# Load data
# ======================================

cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

cube = cube_data["indian_pines_corrected"].astype(np.float64)
ground_truth = gt_data["indian_pines_gt"]

print("Cube shape:", cube.shape)

# ======================================
# Band -> wavelength mapping
# ======================================
# The .mat files ship with no wavelength metadata, only band indices. AVIRIS
# (the sensor that captured this scene) is documented as covering ~400-2500nm
# across its original 220 bands for this dataset; the precise per-band
# calibration file for this specific 1992 flight isn't publicly available, so
# uniform spacing across that range is the standard approximation used when
# the exact file is missing. Sanity check: under this model, raw band 27
# (1-indexed) lands at ~649nm, which matches the well-established convention
# of using bands (50, 27, 17) as a (NIR, Red, Green) false-color composite for
# this exact dataset -- band 27 being "red" under that convention lines up
# with this approximation, which is reassuring but not a substitute for the
# real calibration file.

raw_wavelengths = np.linspace(400, 2500, 220)

# Water-absorption bands removed to go from 220 -> 200 corrected bands
# (1-indexed ranges, per the dataset's standard documentation).
removed_1idx = list(range(104, 109)) + list(range(150, 164)) + [220]
removed_0idx = [b - 1 for b in removed_1idx]

keep_mask = np.ones(220, dtype=bool)
keep_mask[removed_0idx] = False
wavelengths = raw_wavelengths[keep_mask]  # aligned with the 200 corrected bands
assert len(wavelengths) == cube.shape[2] == 200


def band_for(target_nm):
    return int(np.argmin(np.abs(wavelengths - target_nm)))


red_band = band_for(660)
red_edge_band = band_for(720)
nir_band = band_for(800)

print(f"\nRed band: index {red_band}, {wavelengths[red_band]:.1f}nm")
print(f"Red-edge band: index {red_edge_band}, {wavelengths[red_edge_band]:.1f}nm")
print(f"NIR band: index {nir_band}, {wavelengths[nir_band]:.1f}nm")

red = cube[:, :, red_band]
red_edge = cube[:, :, red_edge_band]
nir = cube[:, :, nir_band]

# ======================================
# NDVI / NDRE
# ======================================

eps = 1e-8
ndvi = (nir - red) / (nir + red + eps)
ndre = (nir - red_edge) / (nir + red_edge + eps)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("NDVI = (NIR - Red) / (NIR + Red)")
plt.colorbar(im, ax=ax, label="NDVI")
plt.tight_layout()
plt.savefig("phase5_indices/ndvi_map.png")
plt.close(fig)
print("\nNDVI map saved to phase5_indices/ndvi_map.png")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(ndre, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("NDRE = (NIR - RedEdge) / (NIR + RedEdge)")
plt.colorbar(im, ax=ax, label="NDRE")
plt.tight_layout()
plt.savefig("phase5_indices/ndre_map.png")
plt.close(fig)
print("NDRE map saved to phase5_indices/ndre_map.png")

# ======================================
# Compare index values against ground-truth classes
# ======================================

print("\nMean NDVI / NDRE per class:")
for class_id in range(1, 17):
    class_mask = ground_truth == class_id
    if class_mask.sum() == 0:
        continue
    print(
        f"  {class_id:2d} {CLASS_NAMES[class_id - 1]:<30s}"
        f" NDVI={ndvi[class_mask].mean():.3f}  NDRE={ndre[class_mask].mean():.3f}"
        f"  (n={class_mask.sum()})"
    )

# ======================================
# Extension: nitrogen-relevant index
# ======================================
# Chlorophyll Index Red Edge (CIre) = (NIR / RedEdge) - 1. This is a
# standard nitrogen-status proxy in precision-agriculture literature: leaf
# chlorophyll concentration (which tracks nitrogen availability) strongly
# affects red-edge reflectance, so NIR/RedEdge ratios are more sensitive to
# chlorophyll/nitrogen status than NDVI, which saturates over dense canopy.

ci_red_edge = (nir / (red_edge + eps)) - 1

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(ci_red_edge, cmap="viridis")
ax.set_title("CIre = (NIR / RedEdge) - 1  [nitrogen-status proxy]")
plt.colorbar(im, ax=ax, label="CIre")
plt.tight_layout()
plt.savefig("phase5_indices/ci_red_edge_map.png")
plt.close(fig)
print("\nCIre (nitrogen-proxy) map saved to phase5_indices/ci_red_edge_map.png")

print("\nMean CIre per class:")
for class_id in range(1, 17):
    class_mask = ground_truth == class_id
    if class_mask.sum() == 0:
        continue
    print(
        f"  {class_id:2d} {CLASS_NAMES[class_id - 1]:<30s}"
        f" CIre={ci_red_edge[class_mask].mean():.3f}"
    )
