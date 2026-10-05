# Leave No-One-Out: Context Modifications in Tabular Foundation Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreaKarlova/interpret_tfm/blob/main/leave_no_one_out_tfm.ipynb)
[![CI Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)
[![Paper Status](https://img.shields.io/badge/Paper-AISTATS%202027%20Under%20Review-orange.svg)](https://github.com/AndreaKarlova/interpret_tfm)

Official codebase and interactive Colab demonstration supporting:  
**"Leave No-One-Out: Context Modifications in Tabular Foundation Models"** (Anonymous Authors, AISTATS 2027 Under Review).

---

## Interactive Google Colab Notebook

Launch the complete interactive benchmark with one click:  
👉 **[Open in Google Colab](https://colab.research.google.com/github/AndreaKarlova/interpret_tfm/blob/main/leave_no_one_out_tfm.ipynb)**

### What the Notebook Covers:
1. **The Three Evaluation Targets (Section 2.2):** Disentangles **Difficulty** ($L_i^{\text{rec}}$ / $L_i$), **Influence** ($I_i(x; \mathcal{D})$), and **Intervention Value** ($V(\mathcal{A}; \mathcal{D})$).
2. **Mathematical Exactness of Hat-Matrix LOO:** Closed-form $O(n^2)$ leave-one-out scores via the PRESS identity ($p^{\text{fr},-i}(y_i) = \frac{q_i - S_{ii}}{1 - S_{ii}}$, $L_i = 1 - p^{\text{fr},-i}$) and Gaussian Process LOO (Rasmussen & Williams, Eq. A.20), matching brute-force deletion down to $< 10^{-12}$ error.
3. **Auditing Context Explanations (Section 4):** Decomposes deletion into frozen-head effect and signed representation error (Eq. 8), verifies exactness on fixed-feature negative controls ($E_i \equiv 0$), audits context admission (Eq. 12), and computes Table 1 audit metrics (Bias, RMS error, Sign agreement, Top-$k$ overlap).
4. **Difficulty $\ne$ Intervention Value (Section 5, Table 2 & Table D.3):** Demonstrates the **Sign-Reversal Phenomenon**: despite minority detection AUC of $0.991$, setting $\tau=0.80$ (23.1% tail precision) upweights hard majority points and collapses worst-group accuracy ($0.158 \to 0.045$), while setting $\tau=0.95$ (83.5% tail precision) surges worst-group accuracy to $0.602$ (pseudo-group balancing: $0.607$).
5. **Multiple Tabular Foundation Models (Section 5.4, Table 4 & Table B.1):** Evaluates **TabPFN** (with In-Context Prompt Resampling decomposition, Eq. D.7), **TabICLv2** (with symmetric embedding extraction and KernelICL head), and **KernelICLClassifier**.
6. **Representation Geometry Audit (Section 4.5):** Tests the neighborhood label agreement hypothesis in foundation model embedding space (comparing majority vs minority $k$-NN label agreement).
7. **Label Noise & The Prediction Rule Dependency (Section 5.3, Table 3 & Table D.5):** Shows that kernel smoothers average away label noise (pruning gives no gain), whereas noise-sensitive consumers (1-NN) recover from $0.594 \to 0.729$.
8. **Censored Responses as an Analytical Stress Test (Section 6 & Appendix E):** Implements closed-form cavity LOO for Tobit / censored GP likelihoods and demonstrates the **Surprisal Units Dilemma** (Eq. E.8), where raw mixed surprisal rankings flip solely based on time unit scaling.

### MNIST-C rare-corruption notebook

[`notebooks/kernel_icl_leave_one_out_mnist_corrupted.ipynb`](notebooks/kernel_icl_leave_one_out_mnist_corrupted.ipynb) adapts the Section 5 experiments (Tables 2–4, D.1–D.5) to MNIST-C. Each context has a few **rare but correctly labelled** corrupted digits, and the corruption is independent of the class. The notebook asks whether difficulty scores detect those digits, and whether upweighting, pruning or resampling them helps. It covers the kernel head, logistic regression, 1-NN (label noise) and TabICLv2.

It is built for Colab: the first cell clones this repository and imports its helpers from `kernel_louis` (`mnist_c`, `cnn`, `predictors`, plus the multiclass score/head/metric functions). Push any changes before running it there.

### Waterbirds notebook

[`notebooks/kernel_icl_leave_one_out_waterbirds.ipynb`](notebooks/kernel_icl_leave_one_out_waterbirds.ipynb) runs the same sections and tables as the MNIST-C notebook on Waterbirds, where the rare groups (landbird on water, waterbird on land) fail because they break a **background shortcut** rather than because they are scarce. Features come from an ImageNet ResNet-50 (cached in Drive), the test set is the official test split, and the official validation split supplies the τ/λ selection set and a group-balanced reference context. Data helpers are in `kernel_louis/waterbirds.py`. The earlier experiment is kept as `notebooks/kernel_icl_leave_one_out_waterbirds_old.ipynb`.

### KernelICL audit notebooks

[`notebooks/kernelicl_waterbirds.ipynb`](notebooks/kernelicl_waterbirds.ipynb) and [`notebooks/kernelicl_mnist_corrupted.ipynb`](notebooks/kernelicl_mnist_corrupted.ipynb) run the paper's Section 4 audits on the fine-tuned KernelICL checkpoint (`paper.pt`). For every context edit the paper defines (deletion, label replacement, admission, head reweighting), they compare the frozen head explanation with rerunning the model on the edited context, plus a GP head on KernelICL embeddings. Specs: [`specs/`](specs/). They use the matched edited passes in `kernelicl/kernelicl_diagnostics.py` and the calibration in `kernelicl/kernelicl_clinical.py`.

---

## Installation

```bash
git clone https://github.com/AndreaKarlova/interpret_tfm.git
cd interpret_tfm
pip install -e .
```

To install tabular foundation model backends:
```bash
pip install tabpfn tabicl
```

## Running Tests

```bash
PYTHONPATH=. pytest -v
```

