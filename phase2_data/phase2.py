from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# Load both .mat files
cube_data = loadmat("data/Indian_pines_corrected.mat")
gt_data = loadmat("data/Indian_pines_gt.mat")

# Check what keys are inside
print("Cube file keys:")
print(cube_data.keys())

print("\nGround truth file keys:")
print(gt_data.keys())

# Extract the actual arrays
cube = cube_data["indian_pines_corrected"]
ground_truth = gt_data["indian_pines_gt"]

# Verify what we loaded
print("\nCube")
print("Type:", type(cube))
print("Shape:", cube.shape)

print("\nGround Truth")
print("Type:", type(ground_truth))
print("Shape:", ground_truth.shape)

# Print a small part of the label matrix
print("\nFirst 10 x 10 labels:")
print(ground_truth[:10, :10])

plt.figure(figsize=(8, 8))

plt.imshow(ground_truth)

plt.title("Indian Pines Ground Truth")

plt.colorbar(label="Class Label")

plt.savefig("phase2_data/gt_map.png")

plt.show()

# Select all pixels belonging to class 3
class3_pixels = cube[ground_truth == 3]

print("\nClass 3 pixels shape:")
print(class3_pixels.shape)

# Compute mean spectrum
mean_spectrum = class3_pixels.mean(axis=0)

print("\nMean spectrum shape:")
print(mean_spectrum.shape)

# Plot the mean spectrum
plt.figure(figsize=(8, 5))
plt.plot(mean_spectrum)

plt.title("Mean Spectrum of Class 3")
plt.xlabel("Band Index")
plt.ylabel("Mean Reflectance")

plt.grid(True)

plt.savefig("phase2_data/class3_mean_spectrum.png")
plt.show()

plt.figure(figsize=(10, 6))

#Now plotting mean spectrum of all the classes

for class_id in range(1, 17):

    class_pixels = cube[ground_truth == class_id]

    mean_spectrum = class_pixels.mean(axis=0)

    plt.plot(mean_spectrum, label=f"Class {class_id}")

plt.title("Mean Spectrum of All Classes")
plt.xlabel("Band Index")
plt.ylabel("Mean Reflectance")

plt.grid(True)

plt.legend(fontsize=7, ncol=2)

plt.savefig("phase2_data/class_spectra.png")

plt.show()