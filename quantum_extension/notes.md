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

## Follow-up: does a bigger circuit close the gap? (`improved_ansatz.py`)

Tried both next steps from above at once: 6 qubits instead of 4 (91.2% PCA variance
retained instead of 88.8%), and `qml.StronglyEntanglingLayers` instead of
`qml.BasicEntanglerLayers` (54 trainable weights instead of 18). Sample count and epochs
were both reduced (120 train / 40 test, 30 epochs vs. the original's 240 train / 100 test,
40 epochs) purely because a 6-qubit statevector with 3x the parameters per layer was no
longer tractable at the original scale on a laptop CPU in reasonable time.

| Model | Test accuracy | Train time |
|---|---|---|
| Classical SVM (6 PCA features) | 80.0% | 0.004s |
| Quantum VQC, 4 qubits, BasicEntanglerLayers (original) | 56.0% | 284.6s |
| Quantum VQC, 6 qubits, StronglyEntanglingLayers (this run) | **72.5%** | 125.6s |

The gap to classical narrowed a lot — from 22 points down to 7.5. Training accuracy was
still climbing at epoch 30 (`improved_ansatz_comparison.png`, left panel) when the run was
cut off for time, so this number is likely a slight underestimate of what the same setup
would reach with more epochs, not a converged ceiling. The 125.6s vs. 284.6s training time
isn't a fair speed comparison either way — this run trained on half as many samples for
fewer epochs, so it doesn't mean a bigger circuit trains faster; it means less data made it
feasible to run at all.

**Takeaway:** ansatz expressiveness and qubit count (aggressive PCA compression) were real
bottlenecks, not an inherent limit of quantum computing on this task — improving both closed
roughly two-thirds of the gap to classical. The remaining gap and the sample-size compromise
this run needed both point the same direction: a more capable simulator (GPU-backed, or fewer
per-sample circuit evaluations via batching) is the next lever, not further architecture
tweaks on this same laptop-CPU setup.
