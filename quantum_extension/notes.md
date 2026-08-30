# Quantum classifier extension

## What was built

A PennyLane variational quantum classifier (`quantum_classifier.py`), binary classification
between the two largest Indian Pines classes (Corn-notill, Soybean-mintill):

- Standardize the 200-band spectrum, reduce to 4 principal components (88.8% variance
  retained), scale to `[-pi, pi]`.
- Angle-embed the 4 features onto 4 qubits (`qml.AngleEmbedding`, Y-rotation).
- A 3-layer basic entangling variational ansatz (`qml.BasicEntanglerLayers`) with trainable
  weights.
- Readout: expectation value of Pauli-Z on qubit 0, plus a trainable bias; sign of the
  output is the predicted class.
- Trained with Adam via PennyLane's `backprop` differentiation on a classical simulator
  (`default.qubit`) — this is simulated quantum computing, not real quantum hardware.
- Model selection: best-validation-accuracy checkpoint (a held-out validation split,
  distinct from the test set), same discipline as Phase 4's MLP early-stopping approach —
  avoids picking a flattering epoch by peeking at test performance.

A classical SVM was trained on the *identical* 4 PCA features and split, as the fair
baseline comparison.

## Results (`quantum_vs_classical.png`)

| Model | Test accuracy | Train time |
|---|---|---|
| Classical SVM (4 PCA features) | 78.0% | 0.005s |
| Quantum VQC (4 qubits, 3 layers, best-val checkpoint) | 56.0% | 284.6s |

## Interpretation

The quantum classifier underperformed the classical baseline on both accuracy and speed,
and training was visibly noisy even with a validation-based checkpoint. Three factors
explain most of the gap:

1. **Simulation, not hardware, and slow because of it.** `default.qubit` simulates 4 qubits
   exactly in software; every gradient step evaluates the circuit once per training sample
   with no batching in this implementation, which is why training took roughly 56,000x
   longer wall-clock than the SVM for a problem this small. Real quantum hardware wouldn't
   fix the accuracy gap, but a batched/vectorized simulation would fix most of the speed gap.
2. **Very shallow ansatz.** 3 layers of basic entangling gates on 4 qubits is a small
   hypothesis space compared to an RBF-kernel SVM implicitly operating in a much
   higher-dimensional feature space. More expressive published architectures (hybrid
   quantum-classical transformers) get much closer to classical performance, but with
   substantially more architectural sophistication than this from-scratch demo.
3. **4 features is a hard compression.** Going from 200 bands to 4 PCA components for the
   classical baseline itself costs several accuracy points (Phase 7 showed 20-30 components
   costs ~4-6 points versus the full spectrum) — 4 components is a much more aggressive cut,
   made necessary by qubit count, not by what the data actually needs.

**Next steps:** more qubits (more PCA components without losing as much signal), a deeper
or more expressive ansatz (e.g. `StronglyEntanglingLayers`, or a quantum kernel method
rather than a variational circuit), and enough compute to do that without a 5-minute
training run becoming a multi-hour one.
