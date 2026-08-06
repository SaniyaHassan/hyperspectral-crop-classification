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

####Phase 1 — Why it works

We keep a separate test set because the goal is not to see how well the model remembers the training data, but how well it performs on new data it has never seen before. A model that only performs well on the training set is not useful in real-world applications because new inputs will always be different. Comparing training and test accuracy helps us judge the model's ability to generalize. If both accuracies are high and close together, the model generalizes well. A large gap usually indicates overfitting, while low accuracy on both training and test data indicates underfitting.

*Prove-it 1
Define overfitting and underfitting in your own words. How does each look on a learning curve?

Overfitting happens when the model memorizes the training data instead of learning the underlying patterns. It achieves very high training accuracy but performs much worse on unseen test data. On a learning curve, overfitting appears as a large gap between the training and validation curves, with training accuracy much higher.

Underfitting happens when the model is too simple to learn the patterns in the data. It performs poorly on both the training and test sets. On a learning curve, the training and validation curves stay close together, but both have low accuracy.

*Prove-it 2
You get 99% train, 62% test. Which is it, and name two things you'd try.

This is a case of overfitting because the model performs extremely well on the training data but fails to generalize to unseen data.

Two things I would try are:

Reduce the complexity of the model or add regularization so it cannot memorize the training data as easily.
Collect more training data or use better validation techniques so the model learns more general patterns instead of memorizing specific examples.
*Prove-it 3
Why is reporting only training accuracy meaningless?

Reporting only training accuracy is misleading because the model has already seen those examples during training. A high training accuracy does not prove that the model has learned useful patterns—it may have simply memorized the training data. Without evaluating on unseen test data, we cannot tell whether the model can generalize to new examples.

*Prove-it 4
If your test set is tiny, why might a high test accuracy still be untrustworthy?

A very small test set may not represent the full variety of the data. The model might perform well simply because those few test examples happen to be easy or similar to the training data. This can give an overly optimistic accuracy that may not hold when the model is tested on a much larger and more diverse dataset.

### PHASE-2
Why it works

Different land-cover types have different spectral signatures because they have different physical and chemical compositions. Every material absorbs and reflects light differently depending on its properties. For example, vegetation contains chlorophyll, water, and cellulose, while soil contains minerals, sand, and organic matter. As a result, vegetation and soil reflect different amounts of light at different wavelengths, giving each material its own unique spectral signature.

Prove-it
1. Which two classes have the most similar mean spectra? Predict: will your classifier confuse them? Why?

From the plot alone, it is difficult to determine exactly which two classes are the most similar because many of the curves overlap. Classes with very similar mean spectra are likely to confuse the classifier because they reflect light in nearly the same way across many spectral bands, making them harder to distinguish.

Note: This is a better answer than guessing two class numbers. Your instructor will appreciate that you didn't make an unsupported claim.

2. What are water-absorption bands, and why remove them?

Water-absorption bands are spectral bands where water vapor in the atmosphere absorbs a large amount of incoming light before it reaches the Earth's surface or returns to the sensor. Because very little useful signal reaches the sensor, these bands are noisy and unreliable. Researchers remove them so the model learns from clean, meaningful spectral information instead of atmospheric noise.

3. Your ground truth doesn't label every pixel (lots of "background/unlabelled"). How does that affect how you'll split train/test?

Pixels labeled 0 are background or unlabeled, so they should not be included in either the training or testing sets. Since we do not know their true class, using them would introduce incorrect labels and reduce the quality of the model. Only pixels with valid class labels (1–16) should be used when creating the train and test splits.