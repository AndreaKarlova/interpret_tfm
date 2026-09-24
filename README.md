# Kernel-LOUIS: Exact In-Context LOO for Tabular Foundation Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreaKarlova/interpret_tfm/blob/main/leave_no_one_out_tfm.ipynb)
[![CI Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](tests/)

Official codebase and Colab demonstration supporting:  
**"Leave No One Out in Context: Exact Leave-One-Out Scores for Interpretable Tabular Foundation Models"** (Anonymous Authors).

---

## Interactive Google Colab Notebook

Launch the complete interactive benchmark with one click:  
👉 **[Open in Google Colab](https://colab.research.google.com/github/AndreaKarlova/interpret_tfm/blob/main/leave_no_one_out_tfm.ipynb)**

### What the Notebook Covers:
1. **Mathematical Exactness of Hat-Matrix LOO:** Closed-form $O(1)$ leave-one-out scores via the PRESS identity ($p^{-i}(y_i) = \frac{q_i - S_{ii}}{1 - S_{ii}}$) and Gaussian Process LOO (Rasmussen & Williams), matching brute-force deletion down to $< 10^{-12}$ error.
2. **Auditing Context Influence (Eq. 6):** Decomposes actual context deletion into a head-level term (closed-form) and a representation-mediated discrepancy (upstream transformer shift).
3. **The Core Finding: Tail Precision Governs Downstream Accuracy:** Reproduces Table 1, Table 2, and Table 3. Demonstrates the **Sign-Reversal Phenomenon**: mis-setting the tail quantile ($\tau=0.80$ for a $4.7\%$ minority) upweights hard majority points and collapses worst-group accuracy ($0.158 \to 0.092 \to 0.045$), whereas setting $\tau=0.95$ recovers worst-group accuracy up to $0.602$ and pseudo-group balancing to $0.607$.
4. **Multiple Tabular Foundation Models:** Loads and benchmarks **TabPFN** (with In-Context Prompt Resampling, Eq. 11), **TabICLv2** (with symmetric embedding extraction and KernelICL head), and **KernelICLClassifier**.
5. **Representation Geometry Audit:** Evaluates top-$k$ nearest neighbor label agreement in the foundation model embedding space, testing Section 4.5's hypothesis regarding minority placement in same-label vs. opposite-label neighborhoods.
6. **Zero-Shot Label Noise Detection (Table 5):** Validates the closed-form score as an exact, zero-shot mislabel detector ($\text{AUC} > 0.94$), and shows that while kernel heads average away noise, noise-sensitive consumers (1-NN) recover from $0.594 \to 0.729$.
7. **Censored Likelihood Extension:** Explores Tobit (censored-normal) likelihoods for survival and thresholded outcomes.

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

