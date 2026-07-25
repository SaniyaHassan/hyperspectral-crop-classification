# Phase 0

## Why it works

A hyperspectral pixel is different from a normal RGB pixel because instead of storing only three values (Red, Green, and Blue), it stores reflectance values measured across many different spectral bands. In the Indian Pines dataset, each pixel contains 200 reflectance values. Since every value corresponds to the amount of light reflected at a particular spectral band, plotting these values creates a curve called the spectral signature. Different materials such as vegetation, soil, and water reflect light differently across the spectrum, which is why each material has its own unique spectral signature. This information allows machine learning models to distinguish between different land-cover types.

---

### Prove-it Question 1

**What are the two spatial dimensions and the one spectral dimension of your cube, with numbers?**

The hyperspectral cube has a shape of **(145, 145, 200)**.

- Height (spatial dimension): **145 pixels**
- Width (spatial dimension): **145 pixels**
- Spectral dimension: **200 spectral bands**

This means the image contains **145 × 145 = 21,025 pixels**, and each pixel has measurements across **200 spectral bands**.

---

### Prove-it Question 2

**A normal photo has 3 values per pixel (R, G, B). How many does this cube have, and what does each represent physically?**

A normal RGB image stores three values per pixel representing the intensity of red, green, and blue light. In contrast, each pixel in this hyperspectral cube stores **200 reflectance values**. Each value represents how much incoming light was reflected by the surface at a particular spectral band. Together, these measurements form the spectral signature of that pixel.

---

### Prove-it Question 3

**Why did you .gitignore the data instead of committing it?**

The dataset was added to `.gitignore` because the `.mat` files are large binary files that unnecessarily increase the repository size and are not suitable for version control. Since the Indian Pines dataset is publicly available, anyone can download it separately. Git is best used for tracking source code and documentation rather than large datasets.
