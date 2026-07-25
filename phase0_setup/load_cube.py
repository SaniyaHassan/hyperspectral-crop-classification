from scipy.io import loadmat
import matplotlib.pyplot as plt

# Loading hyperspectral image
data = loadmat("data/Indian_pines_corrected.mat")

# Showing that what variables are stored inside the .mat file
print("Keys in file:")
print(data.keys())

# Getting the hyperspectral cube
cube = data["indian_pines_corrected"]

# Printing information about it
print("\nType:", type(cube))
print("Shape:", cube.shape)

# Select one pixel (row 50, column 50)
pixel = cube[50, 50, :]

# Plot the spectral signature
plt.figure(figsize=(8, 5))
plt.plot(pixel)

plt.title("Spectral Signature of Pixel (50, 50)")
plt.xlabel("Band Index")
plt.ylabel("Reflectance")

# Save the figure
plt.savefig("phase0_setup/spectrum.png")

# Display the figure
plt.show()