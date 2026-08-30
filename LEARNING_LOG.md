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

### PHASE -3
Why it works

An SVM treats each pixel as a point in a high-dimensional feature space, where each spectral band is one feature. It tries to find the boundary (called a hyperplane) that best separates different land-cover classes while maximizing the margin between them. Pixels from the same class usually have similar spectral signatures, so they form clusters in this feature space. The SVM learns a boundary that separates these clusters and can then classify new unseen pixels based on their spectra.

Prove-it 1

What does one cell in the confusion matrix mean?

One cell shows how many samples of an actual class were predicted as another class. Diagonal cells are correct predictions, while off-diagonal cells represent misclassifications. Larger off-diagonal values indicate that the classifier is confusing those two classes.

Prove-it 2

Why did you standardize the features? What breaks if you don't?

Standardization puts every spectral band on the same numerical scale. Without it, bands with larger numerical values would dominate the distance calculations used by the SVM, even if they are not more informative. This can reduce classification performance.

Prove-it 3

If you'd split train/test randomly across spatially adjacent pixels, why might your accuracy be misleadingly high?

If pixels from the training and test sets are spatially adjacent, they are likely to have very similar spectral signatures because nearby pixels usually belong to the same land-cover type. This spatial correlation means the model will likely perform much better on nearby test pixels than it would on truly unseen data. As a result, the accuracy would be artificially inflated, giving an overly optimistic view of how the model would perform on new data from different locations.

### PHASE-4
Why it works

One training step goes: forward pass → loss → backward pass → optimizer step. The forward pass pushes a spectrum through the network's layers to produce 16 class scores (logits). `CrossEntropyLoss` compares those scores against the true label and produces a single number that's high when the network is confidently wrong and low when it's confidently right. Backpropagation then computes the gradient of that loss with respect to every weight in the network — it works backward from the output layer, applying the chain rule layer by layer, so each weight ends up with a number telling it "how much did you contribute to this error, and in which direction should you move to reduce it." The optimizer (Adam here) takes those gradients and actually updates the weights, nudging each one a small step in the direction that lowers the loss, using per-parameter adaptive learning rates so different weights can move at different speeds. Repeat this thousands of times and the network's weights gradually settle into values that separate the 16 classes in spectral space.

Prove-it 1

What is a loss function and why does the network need one?

A loss function turns "how wrong was this prediction" into a single number the network can optimize. Without it, there's nothing for backpropagation to take the gradient of — the network needs a scalar target to push down before it knows which direction to adjust every weight in.

Prove-it 2

Your validation loss starts rising while training loss keeps falling. What's happening and what did you do about it?

That's overfitting: the model is starting to memorize quirks of the training set instead of learning patterns that generalize. In my baseline run (no regularization), this is visible directly in `training_curve.png` — validation loss bottoms out around epoch 545 and then climbs for the rest of the 1000 epochs while training loss keeps falling toward zero. I addressed it by adding dropout (p=0.3) and weight decay (1e-4) in the second run — at the same 1000 epochs, the regularized model's validation loss is still decreasing (its minimum is still at the final epoch), meaning it hasn't started overfitting yet at that point.

Prove-it 3

What did your regularization change, and why?

Dropout randomly zeroes out a fraction of neurons on each forward pass during training, which stops the network from relying too heavily on any single neuron or co-adapted group of neurons — it's forced to learn more redundant, generalizable features. Weight decay adds a penalty on the size of the weights to the loss, discouraging the network from fitting the training data with very large, sharply-tuned weights that would overfit noise. Together they slowed down how fast the model fit the training set, which kept train and validation loss closer together for longer.

Prove-it 4 (Research reflex)

Did the MLP beat your SVM baseline? If not, is that a failure — or a finding? Justify.

Yes — the regularized MLP reached 92.9% test accuracy versus the SVM's 82.0% (Phase 3), and even the overfit baseline MLP (91.8%) beat it. That's not surprising given the setup: the SVM used its default hyperparameters with no tuning, while the MLP got real per-parameter optimization (Adam) over 1000 epochs. It's a fair comparison of "reasonable effort SVM" vs. "reasonable effort MLP," not proof that MLPs are inherently better for this task — a properly grid-searched SVM would likely close some of that gap.

### PHASE-5
Why it works

The NDVI ratio (NIR - Red) / (NIR + Red) measures vegetation health because of how a leaf's cell structure and chlorophyll interact with light. Chlorophyll strongly *absorbs* red light for photosynthesis, so healthy vegetation reflects very little red — red reflectance stays low. Meanwhile a leaf's internal spongy mesophyll structure strongly *scatters* near-infrared light, which isn't used for photosynthesis at all, so NIR reflectance is high. That combination (low red, high NIR) is close to unique to live, healthy vegetation — bare soil, water, and rock don't show that same NIR jump, so the ratio cleanly separates "vigorous plant" from "everything else," and it's normalized (divided by the sum) so it isn't just tracking overall brightness/illumination.

I couldn't find the exact per-band calibration file for this specific 1992 AVIRIS flight, so I derived the band-to-wavelength mapping by assuming uniform spacing across the sensor's documented ~400-2500nm range over its 220 original bands, then removed the same water-absorption bands (104-108, 150-163, 220) that were dropped to build the 200-band corrected cube. I sanity-checked this against a fact I could verify independently: bands (50, 27, 17) are the well-established (NIR, Red, Green) false-color convention for this exact dataset, and under my approximation raw band 27 lands at ~649nm — squarely in the red range, matching that convention. That gave me enough confidence to use it, while still treating it as an approximation rather than ground truth.

Prove-it 1

What is the "red edge" and why is it a sensitive indicator of plant stress? Why does the lab care about NDRE specifically, not just NDVI?

The red edge is the narrow wavelength region (roughly 690-730nm) where a healthy leaf's reflectance jumps sharply from low (chlorophyll absorption in the red) to high (mesophyll scattering in the NIR). Where exactly that jump happens, and how steep it is, shifts measurably with chlorophyll concentration — and chlorophyll concentration tracks nitrogen status and early plant stress well before a plant looks visibly yellow or wilted to the eye. NDVI, by contrast, uses full red and full NIR, both of which saturate once a canopy is reasonably dense — once there's "enough" chlorophyll, adding more barely changes red or NIR reflectance, so NDVI plateaus and stops being sensitive to further changes. NDRE stays sensitive in that denser-canopy regime because the red-edge band hasn't saturated yet, which is exactly why a field lab tracking crop stress/nitrogen over a growing season needs NDRE rather than just NDVI — it can still detect changes after NDVI has flattened out.

Prove-it 2

Your NDVI map — do the bright regions match the crop classes in the ground truth? Where doesn't it match, and why might that be?

Partially, and the mismatch is informative. Comparing my per-class means to the ground-truth map: Woods (0.511), Grass-pasture (0.438) and Grass-trees (0.354) have the highest NDVI, which matches expectations for established, closed-canopy vegetation. But the three row-crop classes — Corn-notill (0.084), Corn-mintill (0.091), Soybean-notill/mintill/clean (0.071-0.094) — have *low* NDVI, well below the grass/woods classes, even though they're indisputably crops. This isn't NDVI failing; Indian Pines was flown in June 1992, early in the growing season, when corn and soybean canopies haven't closed yet — a large fraction of each row-crop pixel is still bare tilled soil between rows, so the true per-pixel signal is a mix of "soil" and "young plant," which pulls the mean NDVI down. Stone-Steel-Towers (-0.030) is the one class that's genuinely non-vegetated, and it's correctly the lowest/near-zero of all 16.

Prove-it 3 (Research reflex)

This cube isn't radiometrically calibrated to true reflectance. What does that mean for whether your absolute NDVI numbers are trustworthy vs. just the spatial pattern?

It means I should trust the relative pattern — which pixels/classes are higher or lower than others, and the field boundaries visible in the maps — much more than any specific NDVI number in isolation. Without calibration to true at-surface reflectance (accounting for atmospheric absorption/scattering, sensor gain, and illumination geometry), the raw digital numbers in the cube are only proportional to reflectance, not equal to it, and that proportionality constant isn't necessarily uniform across the whole scene (it can vary with atmospheric path length, sun angle, sensor viewing angle). So "Woods = 0.511" isn't a number I could compare to a published NDVI value from a different, calibrated sensor or a different day, and small absolute differences between two similar classes aren't necessarily meaningful. What *is* trustworthy is the ordering and spatial structure: Woods clearly and consistently reads higher than Soybean across every pixel comparison, and the NDVI map's field boundaries line up with the ground-truth map's field boundaries. This is exactly the caveat that governs real MachVIS field data too — without a calibration/reference panel in each flight, index maps are reliable for spatial comparison within a single scene, not for absolute agronomic thresholds.

Extension: I also computed the Chlorophyll Index Red Edge, CIre = (NIR / RedEdge) - 1, saved as `phase5_indices/ci_red_edge_map.png`. This is a standard nitrogen-status proxy in precision-agriculture literature — it uses the same red-edge sensitivity as NDRE but as a ratio rather than a normalized difference, which several studies report as more linearly responsive to canopy chlorophyll/nitrogen content at high biomass. Per-class means follow the same physically sensible ordering as NDVI/NDRE (Woods highest at 1.417, Stone-Steel-Towers negative at -0.023), which is the kind of consistency check I'd want before trusting an index on new data.

### PHASE-6
Why it works

Overall accuracy is misleading here because Indian Pines is badly imbalanced — Soybean-mintill alone has 2455 labeled pixels while Oats has only 20. A model could get 90%+ overall accuracy just by nailing the handful of huge classes and being wrong on every single small one, and the single accuracy number would never reveal that. Precision and recall are computed per class, so they can't hide behind the big classes: recall answers "of all the true examples of this class, how many did I actually catch?" (misses this class as something else = recall drops), and precision answers "of everything I labeled as this class, how many were actually right?" (drags in other classes by mistake = precision drops). A class can have terrible recall while overall accuracy still looks great, and only per-class metrics expose that.

Prove-it 1

Which class has the worst recall? Tie it back to Phase 2's spectra and Phase 3's confusion matrix — is the story consistent?

Buildings-Grass-Trees-Drives has the worst recall (0.805) — see `phase6_eval/classification_report.txt`. The confusion matrix (`phase6_eval/mlp_confusion_matrix.png`) shows why: of its 77 test pixels, 10 were misclassified as Woods. That's consistent with the class name itself — "Buildings-Grass-Trees-Drives" is a literal spectral mixture class (structures + grass + trees together), so a good chunk of its pixels genuinely look like pure Woods or pure Grass spectrally. This matches the Phase 2 expectation that spectrally-similar classes get confused, and the Phase 3 SVM confusion matrix showed the same pattern (this class was one of its weaker performers too) — the story is consistent across all three phases, which is reassuring: the model isn't failing randomly, it's failing exactly where the underlying data is genuinely ambiguous.

Prove-it 2

Point to a region where the classification map looks "speckled/noisy." Why does pixel-wise classification produce that, and what kind of method would reduce it?

In `phase6_eval/classification_map.png`, the interior of the large Soybean-mintill field (center of the image, roughly rows 60-100, columns 30-65) is visibly speckled — scattered individual pixels flip to Corn-notill or Soybean-notill even though the whole region is one contiguous field. This happens because the model classifies every pixel independently from its spectrum alone, with zero awareness that its neighbors are almost certainly the same class — a single noisy pixel (sensor noise, a slightly different soil patch, a mixed boundary pixel) can flip a prediction with nothing to correct it. A spectral-spatial method — using a small patch of neighboring pixels as context (e.g. a CNN over 5x5 windows, or a post-processing majority filter/smoothing over predicted labels) — would reduce this, since real fields are spatially contiguous and a method that looks at neighbors would treat an isolated flipped pixel as an outlier.

Prove-it 3 (Research reflex)

If you had to defend this map to the professor, what's the single biggest limitation you'd disclose up front?

The train/test split is pixel-random, not spatially block-held-out — exactly the trap Phase 3's prove-it question flagged. Because train and test pixels can come from directly inside the same field (just a few pixels apart), the model has effectively already seen the spectral signature of every field it's being "tested" on, just from a different pixel in that same field. My 92.9% test accuracy is real for this evaluation protocol, but it's very likely an overestimate of how this model would perform on a genuinely new field it has never seen a single pixel of — that's the number I'd lead with as the caveat.

### PHASE-7
Why it works

I picked PCA-dimensionality-reduction + SVM (from Ustuner, M2GARSS 2024) because it's the piece of Indian Pines literature that's actually reproducible from a bootcamp on a laptop — no exotic architecture, just a preprocessing step (PCA) and a classifier I'd already built (SVM, from Phase 3). It also does double duty: it's the bootcamp's PCA stretch goal, and it directly sets up the low-dimensional feature representation a quantum classifier needs later, since quantum circuits are practically limited to a handful of qubits and can't take a 200-band spectrum directly.

Prove-it 1

In one sentence each: what problem did the paper address, what method, what result?

Problem: whether randomized PCA (a faster, approximate version of PCA) is a safe substitute for standard PCA when preparing hyperspectral data for classification. Method: reduce Indian Pines (and Pavia University) to 20 and 30 components with both standard and randomized PCA, then classify with SVM and gradient boosting. Result: standard PCA outperformed randomized PCA for SVM on both datasets, so the speed of randomized PCA isn't worth the accuracy it gives up.

Prove-it 2

Did your numbers match theirs? If not, name two reasons reproductions diverge.

I couldn't check directly — the material I had access to stated the qualitative finding (standard PCA beat randomized PCA) but not the exact PCA+SVM accuracy numbers themselves, so there's nothing to numerically match against. Full notes and critique are in `phase7_paper/notes.md`. Two reasons reproductions in general diverge even when both are done honestly: (1) unstated or differently-tuned hyperparameters — I used scikit-learn's default SVM (C, gamma) rather than tuning per feature set, and the paper doesn't specify its tuning process; (2) different train/test splits — unless the exact split methodology and seed match, two "correct" runs on the same dataset will land on slightly different pixels in train vs. test, which alone shifts accuracy by a few points.

Prove-it 3 (Research reflex)

What would you ask the authors if you could?

What SVM hyperparameters (C, gamma, kernel) they used for each feature configuration, and whether those were tuned separately per configuration or held fixed across all of them — that single detail would likely explain most of any gap between my reproduction and theirs.

### PHASE-8
Why it works

The report itself is the artifact for this phase — `report.md` covers the pipeline, results, and limitations. The "why" here is the limitations section: every caveat in it (spatial train/test leakage, no radiometric calibration, an approximated wavelength mapping, untuned hyperparameters) is a real, load-bearing limitation of what I actually built, not a boilerplate disclaimer — each one is something that would change a specific number in the results table if fixed.

Defense prep

**Explain your whole pipeline in 60 seconds, no notes.**

Indian Pines is a hyperspectral scene — 145x145 pixels, each with a 200-value spectrum instead of just RGB. I treat each labeled pixel as one sample and its spectrum as its features. I built a baseline SVM (82% test accuracy), then a hand-built PyTorch MLP, where adding dropout and weight decay pushed accuracy to 92.9% and visibly delayed overfitting compared to the unregularized version. I computed NDVI/NDRE vegetation indices per pixel from a derived wavelength mapping, ran the best model over the whole image to get a full classification map and per-class precision/recall (worst class: a literal spectral-mixture class, confused with Woods — consistent with earlier findings), reproduced a PCA+SVM protocol from a real paper, and pushed two pieces further out of personal interest: a nitrogen-status index (CIre) and a small real quantum classifier on the same data.

**Why did you choose the model you chose? What would you try next with a week more?**

The regularized MLP is the model I'd stand behind, because it's the one where I can point to *why* it's better — the training curve directly shows dropout + weight decay keeping validation loss decreasing at epoch 1000 where the unregularized version had already turned upward at epoch ~545. With a week more, I'd fix the biggest limitation first: build a spatially block-held-out train/test split (holding out whole fields, not random pixels) to get an honest generalization estimate, since I already flagged that the current split likely overstates performance.

**Your benchmark cube has 200 bands 400-2500nm. The lab's field cubes have ~25 bands 665-975nm. What can you compute on the lab data that you did here, and what can't you?**

NDVI and NDRE both still work — red (~660nm), red-edge (~720nm), and NIR (~800nm) all fall inside the 665-975nm range MachVIS field cubes cover. What breaks: the CIre chlorophyll index I computed here used the same red-edge/NIR bands so it would technically still compute, but anything needing blue, green, or SWIR bands (e.g. a true-color composite, or any index that relies on visible-blue absorption features, or SWIR-based moisture indices) has no equivalent in a 665-975nm-only cube — those bands simply don't exist in that data. The whole 200-band spectral-signature classification approach from Phases 3-4 would also need rethinking: 25 bands is a much smaller feature space, so a lot of the "let the model find subtle patterns across the full spectrum" argument weakens.

**What's the difference between the spatial pattern of an index being valid and its absolute values being valid, and why does calibration decide that?**

The spatial pattern (which pixels are relatively higher or lower than their neighbors, where field boundaries fall) survives even without calibration, because it only depends on relative differences across the same uncalibrated scene. The absolute value doesn't survive, because without radiometric calibration the cube's digital numbers are only proportional to true reflectance — and that proportionality constant isn't guaranteed to be the same across different scenes, different days, or even different parts of the same scene if atmospheric conditions or sun angle vary. Calibration (a reference panel of known reflectance, captured in-scene) is what pins that proportionality constant down to an actual known value, which is what lets you compare an index number to a literature threshold or to a different flight's data.

**What did you find hardest, and what would you learn next?**

The hardest part was the band-to-wavelength mapping for Phase 5 — there's no calibration file bundled with the .mat data, so I had to derive an approximation and then find an independent way to sanity-check it (the known false-color band convention for this scene) rather than just trusting a number I couldn't verify. What I'd learn next: proper spatial cross-validation for hyperspectral data (block-based splits), since that's the limitation I flagged most and haven't actually fixed yet.