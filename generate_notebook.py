"""Build the complete, publication-grade Google Colab notebook for Kernel-LOUIS."""

import json
import os

def create_notebook():
    cells = []

    def md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def code(text):
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # =========================================================================
    # HEADER CELL
    # =========================================================================
    md("""# Leave No One Out in Context: Exact Leave-One-Out Scores for Interpretable Tabular Foundation Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreaKarlova/interpret_tfm/blob/main/leave_no_one_out_tfm.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-interpret__tfm-blue.svg)](https://github.com/AndreaKarlova/interpret_tfm)

---

### Executive Summary & Theoretical Framework

Tabular Foundation Models (TFMs) such as **TabPFN** and **TabICL** have emerged as state-of-the-art architectures for In-Context Learning (ICL) on structured data. However, their real-world adoption in mission-critical applications is hindered by:
1. **Opaque Prediction Heads:** The standard Multi-Layer Perceptron (MLP) head obscures how context rows are aggregated.
2. **Subpopulation Failure:** Standard ICL prioritizes majority groups under empirical risk minimization, failing on under-represented sub-groups where spurious correlations dominate.

**KernelICL** addresses interpretability by replacing the opaque head with an explicit, self-normalized kernel regression smoother:
$$\\hat{y}(x) = \\sum_{i=1}^n w_i(x) y_i, \\quad w_i(x) = \\frac{K(h(x), h(x_i))}{\\sum_j K(h(x), h(x_j))}, \\quad S_{ij} = \\frac{K_{ij}}{\\sum_k K_{ik}}$$

This paper (**"Leave No One Out in Context"**) makes three pivotal theoretical and empirical advances:

1. **Exact Closed-Form In-Context LOO Scores (The Instruments):**
   Because the explicit head is a linear smoother $\\hat{y} = Sy$, leave-one-out predictions follow in closed form from the diagonal leverage $S_{ii}$ via the classical **PRESS identity** without retraining, Taylor expansions, or Hessian solves:
   $$p^{-i}(y_i) = \\frac{q_i - S_{ii}}{1 - S_{ii}}, \\quad L_i = 1 - p^{-i}(y_i) \\in [0, 1]$$
   and its regularized Gaussian Process counterpart (Rasmussen & Williams):
   $$\\mu_i^{-i} = y_i - \\frac{\\alpha_i}{[\\tilde{K}^{-1}]_{ii}}, \\quad (\\sigma_i^{-i})^2 = \\frac{1}{[\\tilde{K}^{-1}]_{ii}}, \\quad L_i^{\\text{GP}} = -\\log \\mathcal{N}\\left(y_i \\,\\middle|\\, \\mu_i^{-i}, (\\sigma_i^{-i})^2\\right)$$

2. **Auditing Context Influence & Representation-Mediated Discrepancy:**
   In an in-context transformer, context point $i$ not only contributes its label to the head, but also shapes the representations $h(x_j)$ of all other points. We decompose actual context deletion into:
   $$\\underbrace{f_D(x) - f_{D\\setminus i}(x)}_{\\text{Actual Deletion Effect}} = \\underbrace{\\frac{w_i(x; D)}{1 - w_i(x; D)} (y_i - f_D(x))}_{\\text{Frozen-Head Effect (Closed Form)}} + \\underbrace{\\left[f_{D\\setminus i}^{\\text{frozen}}(x) - f_{D\\setminus i}(x)\\right]}_{\\text{Representation-Mediated Discrepancy}}$$

3. **Difficulty $\\neq$ Intervention Value: Tail Precision Governs Downstream Accuracy:**
   The score is an exceptional anomaly and minority detector (mislabel AUC 0.92–0.99, separation 1.00). **However, detection does not automatically transfer to accuracy.** 
   The success of reweighting is strictly governed by the **precision of the upweighted tail set**. When the quantile threshold $\\tau$ is misaligned (e.g. $\\tau=0.80$ for a $4.7\\%$ minority), $77\\%$ of the upweighted examples are hard majority points, **inverting the sign of the intervention** (worst-group accuracy drops from $0.158 \\to 0.092 \\to 0.045$). When $\\tau$ matches the minority rate ($\\tau=0.95$), worst-group accuracy jumps to $0.602$!

---

### Notebook Contents
- **Module 1:** Colab Setup & Package Verification
- **Module 2:** Exact LOO Mathematics & Machine-Precision Validation
- **Module 3:** Context Influence Audit (Fixed-Feature Control vs. Transformers)
- **Module 4:** Tabular Waterbirds Benchmark: Synthetic Spurious Shift
- **Module 5:** The Sign-Reversal Phase Transition: Tail Precision vs. Downstream Accuracy
- **Module 6:** Foundation Models in Action: TabPFN & TabICL Integration
- **Module 7:** Representation Geometry: Top-$k$ Embedding Neighborhood Audit
- **Module 8:** Zero-Shot Label-Noise Detector & Structural Averaging Paradox
- **Module 9:** Censored Likelihoods & Tobit Stress Test
- **Module 10:** Theoretical Conclusions & Practitioner's Cheat Sheet""")

    # =========================================================================
    # MODULE 1: SETUP
    # =========================================================================
    md("""## Module 1: Colab Setup & Environment Initialization

This cell automatically configures Google Colab or local execution by cloning the repository, installing `kernel_louis`, and loading `TabPFN` and `TabICL`.""")

    code("""# Check runtime environment and install dependencies
import sys
import os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Detected Google Colab environment. Installing kernel_louis and dependencies...")
    !git clone https://github.com/AndreaKarlova/interpret_tfm.git
    %cd interpret_tfm
    !pip install -q -e .
    !pip install -q tabpfn tabicl
    print("Environment setup complete.")
else:
    print("Running in local environment.")
    if '.' not in sys.path:
        sys.path.insert(0, '.')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Core mathematical & data libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import torch
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Compute Device: {DEVICE}")

# Kernel-LOUIS Package Modules
from kernel_louis.kernels import default_gamma, gaussian_kernel, median_heuristic_gamma
from kernel_louis.heads import self_normalized_predict, gp_predict, tobit_log_likelihood
from kernel_louis.loo import incontext_loo_nll, gp_loo_logdensity, press_residuals, crossfit_difficulty
from kernel_louis.modulation import (
    tail_multiplier, smooth_multiplier, exponential_multiplier,
    pseudo_group_balance, resample_context
)
from kernel_louis.data import (
    make_tabular_waterbirds, make_spurious_moons, make_spurious_split, make_noisy_label_split
)
from kernel_louis.evaluation import compute_group_metrics, compute_tail_precision_recall
from kernel_louis.audit import audit_kernel_head_influence
from kernel_louis.models import KernelICLClassifier
from kernel_louis.embeddings import extract_symmetric_embeddings

# Tabular Foundation Models
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
    print("✓ TabPFN Foundation Model successfully loaded.")
except Exception as e:
    HAS_TABPFN = False
    print(f"TabPFN not available ({e}).")

try:
    from tabicl import TabICLClassifier
    HAS_TABICL = True
    print("✓ TabICLv2 Foundation Model successfully loaded.")
except Exception as e:
    HAS_TABICL = False
    print(f"TabICL not available ({e}).")

# Aesthetic Matplotlib Configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
print("System ready.")""")

    # =========================================================================
    # MODULE 2: MATHEMATICAL FOUNDATION
    # =========================================================================
    md("""## Module 2: Mathematical Foundation: Exact LOO from the Hat Matrix

### 2.1 The Hat Matrix and the PRESS Identity
For any linear smoother $\\hat{y} = S y$, the leave-one-out prediction error follows directly from the diagonal entries $S_{ii}$ (leverage) without refitting the model:
$$y_i - \\hat{y}_i^{-i} = \\frac{y_i - \\hat{y}_i}{1 - S_{ii}}$$

For the soft class-vote head predicting probabilities:
$$q_i = \\sum_{j=1}^n S_{ij} \\mathbb{I}[y_j = y_i], \\quad p^{-i}(y_i) = \\frac{q_i - S_{ii}}{1 - S_{ii}}$$
The bounded leave-one-out error is:
$$L_i = 1 - p^{-i}(y_i) \\in [0, 1]$$
or surprisal:
$$L_i^{\\text{surprisal}} = -\\log p^{-i}(y_i)$$

### 2.2 Deep Gaussian Process Regression Head
For regularized kernel regression (Gaussian Process with $y \\sim \\mathcal{N}(f, \\sigma^2)$):
From a single matrix inverse $\\tilde{K}^{-1} = (K + \\sigma^2 I)^{-1}$ and $\\alpha = \\tilde{K}^{-1} y$:
$$\\mu_i^{-i} = y_i - \\frac{\\alpha_i}{[\\tilde{K}^{-1}]_{ii}}, \\quad (\\sigma_i^{-i})^2 = \\frac{1}{[\\tilde{K}^{-1}]_{ii}}$$
$$L_i^{\\text{GP}} = -\\log \\mathcal{N}\\left(y_i \\,\\middle|\\, \\mu_i^{-i}, (\\sigma_i^{-i})^2\\right)$$

Below we empirically verify that the closed-form score matches a brute-force leave-one-out procedure down to **machine precision** ($< 10^{-12}$) while being orders of magnitude faster.""")

    code("""# Experiment 1: Numerical Validation of Exactness & Runtime Comparison
import time

rng = np.random.default_rng(42)
n_samples = 60
d_features = 4

X_synth = rng.normal(0, 1, (n_samples, d_features))
y_synth = rng.integers(0, 2, size=n_samples)

gamma = default_gamma(d_features)
K = gaussian_kernel(X_synth, X_synth, gamma)

# 1. Closed-Form Leave-One-Out (Single Matrix Operation)
t0 = time.perf_counter()
L_closed, p_closed = incontext_loo_nll(K, y_synth, return_probs=True)
t_closed = time.perf_counter() - t0

# 2. Brute-Force Leave-One-Out (N Separate Recomputations)
t0 = time.perf_counter()
p_brute = np.zeros(n_samples)
for i in range(n_samples):
    mask = np.ones(n_samples, dtype=bool)
    mask[i] = False
    X_sub, y_sub = X_synth[mask], y_synth[mask]
    K_sub = gaussian_kernel(X_synth[i:i+1], X_sub, gamma)
    p1, _ = self_normalized_predict(K_sub, y_sub)
    p_brute[i] = p1[0] if y_synth[i] == 1 else (1.0 - p1[0])
t_brute = time.perf_counter() - t0

# Compare exactness
max_diff = np.max(np.abs(p_closed - p_brute))
print(f"=== LOO EXACTNESS AUDIT ===")
print(f"Max Absolute Discrepancy: {max_diff:.2e}")
print(f"Closed-form Runtime:      {t_closed*1000:.3f} ms")
print(f"Speedup Factor:           {t_brute / max(t_closed, 1e-9):.1f}x")
assert max_diff < 1e-10, "Exactness failed!"

# 3. Gaussian Process LOO vs Soft-Vote LOO
L_gp = gp_loo_logdensity(K, y_synth, sigma2=0.1)

# Visual Comparison Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Exact Match Validation
axes[0].scatter(p_brute, p_closed, color='#1f77b4', edgecolors='k', s=50, alpha=0.8, label='Context Points')
min_val = min(p_brute.min(), p_closed.min())
max_val = max(p_brute.max(), p_closed.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity y = x')
axes[0].set_title(f"Hat-Matrix LOO Exactness (Max Diff = {max_diff:.1e})", fontweight='bold')
axes[0].set_xlabel("Brute-Force Retrained p^{-i}(y_i)")
axes[0].set_ylabel("Closed-Form Hat-Matrix p^{-i}(y_i)")
axes[0].legend()

# Plot 2: Soft-Vote LOO vs GP Variance-Aware LOO
axes[1].scatter(L_closed, L_gp, color='#ff7f0e', edgecolors='k', s=50, alpha=0.8)
axes[1].set_title("Soft-Vote LOO vs. GP Variance-Aware LOO", fontweight='bold')
axes[1].set_xlabel("Soft-Vote LOO Error L_i")
axes[1].set_ylabel("GP Leave-One-Out Score L_i^{GP}")

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 3: CONTEXT INFLUENCE AUDIT
    # =========================================================================
    md("""## Module 3: Context Influence Audit (Section 3 of Paper)

### The Central Question: Is the Transparent Head Faithful?
An explicit kernel head is advertised as "transparent" because predictions are linear weighted sums of context points:
$$f_D(x) = \\sum_{j=1}^n w_j(x; D) y_j$$
However, in modern in-context learners, **a context row also shapes the representations $h(x_j)$ of other rows** — an upstream channel the head cannot see.

To quantify this, Section 3 of the paper establishes the exact context intervention decomposition:
$$\\underbrace{f_D(x) - f_{D\\setminus i}(x)}_{\\text{actual deletion effect}} = \\underbrace{\\frac{w_i(x; D)}{1 - w_i(x; D)} (y_i - f_D(x))}_{\\text{frozen-head effect (closed form)}} + \\underbrace{\\left[f_{D\\setminus i}^{\\text{frozen}}(x) - f_{D\\setminus i}(x)\\right]}_{\\text{representation-mediated discrepancy}}$$

- On **fixed features** (or frozen embeddings), the second term **identically vanishes** (the negative control).
- On **transformer foundation models**, deleting a context row alters the self-attention keys and queries of remaining points.""")

    code(r"""# Experiment 2: Auditing Context Influence Decomposition (Eq. 6)
n_ctx = 40
n_query = 15
d = 3

X_ctx = rng.normal(0, 1, (n_ctx, d))
y_ctx = rng.integers(0, 2, size=n_ctx)
X_qry = rng.normal(0, 1, (n_query, d))

gamma = default_gamma(d)
K_tr = gaussian_kernel(X_ctx, X_ctx, gamma)
K_qt = gaussian_kernel(X_qry, X_ctx, gamma)

# Select a row to delete
del_idx = 7

# Run the Audit
audit = audit_kernel_head_influence(K_qt, K_tr, y_ctx, del_idx=del_idx)

print("=== CONTEXT INFLUENCE AUDIT (FIXED KERNEL NEGATIVE CONTROL) ===")
print(f"Max Absolute Discrepancy: {audit['max_abs_discrepancy']:.2e}")
print(f"Mean Absolute Discrepancy: {np.mean(np.abs(audit['discrepancy'])):.2e}")

# Visualization of the Influence Decomposition
fig, ax = plt.subplots(figsize=(10, 4.5))
x_idx = np.arange(n_query)
width = 0.35

ax.bar(x_idx - width/2, audit['actual_deletion_effect'], width, label=r'Actual Deletion Effect $f_D(x) - f_{D\setminus i}(x)$', color='#2ca02c')
ax.bar(x_idx + width/2, audit['closed_form_effect'], width, label=r'Frozen-Head Closed-Form $\frac{w_i}{1-w_i}(y_i - f_D)$', color='#1f77b4', alpha=0.7)

ax.set_title(f"Context Influence Audit: Exact Deletion Match (Discrepancy ≡ 0 on Fixed Kernel)", fontweight='bold')
ax.set_xlabel("Query Test Instance Index")
ax.set_ylabel("Prediction Movement Upon Row Deletion")
ax.legend()
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 4: TABULAR WATERBIRDS BENCHMARK
    # =========================================================================
    md("""## Module 4: Tabular Waterbirds Benchmark & Subpopulation Shift

To study subpopulation shift under spurious correlation without visual confounders, Section 4 introduces the **Tabular Waterbirds** construction:
1. **Core Feature:** $x_{\\text{core}} = (2y - 1) + \\mathcal{N}(0, \\sigma_{\\text{core}}^2)$. Tracks true label $y \\in \\{0, 1\\}$ for $100\\%$ of points, but with high noise (weak signal).
2. **Spurious Feature:** $x_{\\text{spurious}} = (2a - 1) \\cdot s + \\mathcal{N}(0, 0.1^2)$. Strong attribute aligned with the label $P(a = y) = 95.3\\%$ in majority groups, but **flipped** for the $4.7\\%$ minority.

This creates 4 distinct groups:
- **Group 0 (Majority):** $y=0, a=0$ (~$47.65\\%$)
- **Group 1 (Minority):** $y=0, a=1$ (~$2.35\\%$)
- **Group 2 (Minority):** $y=1, a=0$ (~$2.35\\%$)
- **Group 3 (Majority):** $y=1, a=1$ (~$47.65\\%$)""")

    code("""# Generate Tabular Waterbirds Dataset (Section 4.3 & 4.4)
X_tr, y_tr, g_tr, is_min_tr = make_tabular_waterbirds(
    n_samples=4000, spurious_corr=0.953, core_noise=1.2, random_state=42
)
X_te, y_te, g_te, is_min_te = make_tabular_waterbirds(
    n_samples=2000, spurious_corr=0.50, core_noise=1.2, random_state=142
)

# Standardize
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
X_tr = (X_tr - mu) / sd
X_te = (X_te - mu) / sd

# Compute exact LOO difficulty scores
gamma = median_heuristic_gamma(X_tr)
K_tr = gaussian_kernel(X_tr, X_tr, gamma)
L_scores = incontext_loo_nll(K_tr, y_tr)

# Detection ROC-AUC
auc_detection = roc_auc_score(is_min_tr, L_scores)
print(f"Total Context Size:    {len(y_tr)}")
print(f"Minority Context Size: {is_min_tr.sum()} ({is_min_tr.mean()*100:.2f}%)")
print(f"Difficulty Score AUC:  {auc_detection:.4f}")

# Visualization: Feature Space & Score Separation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: 2D Feature Space
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
labels = ['Maj (y=0, a=0)', 'Min (y=0, a=1)', 'Min (y=1, a=0)', 'Maj (y=1, a=1)']
for g in range(4):
    mask = g_tr == g
    axes[0].scatter(X_tr[mask, 0], X_tr[mask, 1], c=colors[g], label=labels[g], 
                    alpha=0.6, s=20 if (g==0 or g==3) else 45, edgecolors='none' if (g==0 or g==3) else 'k')
axes[0].set_title("Tabular Waterbirds Feature Space", fontweight='bold')
axes[0].set_xlabel("x_core (Weak Label Signal)")
axes[0].set_ylabel("x_spurious (Strong Spurious Shortcut)")
axes[0].legend()

# Subplot 2: Score Distribution Separation
bins = np.linspace(0, max(L_scores), 40)
axes[1].hist(L_scores[~is_min_tr], bins=bins, alpha=0.6, density=True, label=f'Majority Points (n={(~is_min_tr).sum()})', color='#1f77b4')
axes[1].hist(L_scores[is_min_tr], bins=bins, alpha=0.7, density=True, label=f'Minority Points (n={is_min_tr.sum()})', color='#d62728')
axes[1].set_title(f"LOO Difficulty Separation (ROC-AUC = {auc_detection:.3f})", fontweight='bold')
axes[1].set_xlabel("In-Context LOO Difficulty Score L_i")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 5: TAIL PRECISION & SIGN REVERSAL
    # =========================================================================
    md("""## Module 5: Tail Precision Governs Downstream Accuracy (The Sign Inversion)

### 5.1 Difficulty is Not Intervention Value
Prior work assumes that identifying difficult examples automatically enables improving accuracy via upweighting.
**This paper disproves that assumption.**

In Tabular Waterbirds, the minority comprises $4.7\\%$ of the dataset.
- At $\\tau = 0.80$, the upweighted tail contains the top $20\\%$ ($800$ points).
- Of those $800$ points, only $185$ are true minority! Over $600$ are **hard majority points** (majority points lying near the decision boundary with noisy core features).
- Upweighting hard majority points amplifies the majority bias against the minority, causing worst-group accuracy to **collapse**!
- At $\\tau = 0.95$, the tail size is $200$ points, and precision jumps to $83.5\\%$, driving a massive surge in worst-group accuracy.

Let us reproduce **Table 2** and **Table 3** from the paper.""")

    code("""# Table 2 Reproduction: Tail Composition Across Quantiles
tau_sweep = [0.80, 0.90, 0.95, 0.97]
tail_stats = compute_tail_precision_recall(L_scores, is_min_tr, tau_sweep)

print(f"{'tau':>6} | {'Tail Size':>10} | {'Precision (Min)':>17} | {'Recall':>8}")
print("-" * 50)
for s in tail_stats:
    print(f"{s['tau']:6.2f} | {s['tail_size']:10d} | {s['precision']:17.3f} | {s['recall']:8.3f}")

# Table 3 Reproduction: Modulation Strategies and Downstream Accuracy
K_te = gaussian_kernel(X_te, X_tr, gamma)

conditions = {
    "Baseline (KernelICL, λ=0)": np.ones(len(y_tr)),
    "Tail-set τ=0.80, λ=8":      tail_multiplier(L_scores, lam=8.0, tau=0.80),
    "Tail-set τ=0.80, λ=20":     tail_multiplier(L_scores, lam=20.0, tau=0.80),
    "Tail-set τ=0.95, λ=8":      tail_multiplier(L_scores, lam=8.0, tau=0.95),
    "Tail-set τ=0.95, λ=20":     tail_multiplier(L_scores, lam=20.0, tau=0.95),
    "Pseudo-group balance τ=0.95": pseudo_group_balance(y_tr, L_scores, tau=0.95),
    "Random-tail control":       1.0 + 8.0 * (rng.random(len(y_tr)) > 0.95).astype(float),
    "Oracle (True Minority)":    1.0 + 20.0 * is_min_tr.astype(float),
}

table3_results = []
print(f"\\n{'Condition':<30} | {'Mean Acc':>10} | {'Worst-Group Acc':>16}")
print("-" * 62)
for name, m in conditions.items():
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr, sample_weight=m)
    y_pred = clf.predict(X_te)
    metrics = compute_group_metrics(y_pred, y_te, is_min_te)
    table3_results.append((name, metrics['mean'], metrics['worst']))
    print(f"{name:<30} | {metrics['mean']:10.3f} | {metrics['worst']:16.3f}")

# Visualizing the Phase Transition / Sign Inversion
taus = np.linspace(0.70, 0.98, 25)
worst_l8 = []
worst_l20 = []
for t in taus:
    m8 = tail_multiplier(L_scores, lam=8.0, tau=t)
    m20 = tail_multiplier(L_scores, lam=20.0, tau=t)
    
    clf8 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr, sample_weight=m8)
    worst_l8.append(compute_group_metrics(clf8.predict(X_te), y_te, is_min_te)['worst'])
    
    clf20 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr, sample_weight=m20)
    worst_l20.append(compute_group_metrics(clf20.predict(X_te), y_te, is_min_te)['worst'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(taus, worst_l8, 'o-', color='#1f77b4', lw=2, label='Tail-set λ=8')
ax.plot(taus, worst_l20, 's-', color='#ff7f0e', lw=2, label='Tail-set λ=20')
ax.axhline(table3_results[0][2], color='gray', linestyle='--', label=f'Baseline Worst-Group ({table3_results[0][2]:.3f})')
ax.axvline(1.0 - is_min_tr.mean(), color='red', linestyle=':', lw=2, label=f'True Minority Threshold (1 - p_min = {1.0 - is_min_tr.mean():.3f})')
ax.set_title("The Sign-Reversal Phase Transition: Tail Quantile τ vs. Worst-Group Accuracy", fontweight='bold')
ax.set_xlabel("Tail Quantile Threshold τ")
ax.set_ylabel("Test Worst-Group Accuracy")
ax.legend()
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 6: TABULAR FOUNDATION MODELS (TabPFN & TabICL)
    # =========================================================================
    md("""## Module 6: Loading & Testing Tabular Foundation Models

Here we load multiple Tabular Foundation Models:
1. **KernelICLClassifier:** Explicit in-context kernel classifier with closed-form LOO modulation.
2. **TabPFN (`TabPFNClassifier`):** The canonical tabular foundation model executing pure in-context inference. We evaluate both standard TabPFN and TabPFN with **In-Context Prompt Resampling** (Eq. 11):
   $$\\mathbb{E}[\\hat{y}_c(x)] = \\hat{y}_m(x) + \\mathcal{O}(1/M), \\quad c \\sim \\text{Multinomial}\\left(M, \\frac{m}{\\sum_j m_j}\\right)$$
3. **TabICL (`TabICLClassifier`):** TabICLv2 foundation model, extracting symmetric embeddings $h(x)$ to evaluate Kernel-LOUIS heads directly on foundation model latent spaces.""")

    code("""# Benchmark on Tabular Foundation Models
tfm_results = {}

# 1. KernelICLClassifier (Interpretable Kernel TFM Head)
kicl_base = KernelICLClassifier(lam=0.0)
kicl_base.fit(X_tr, y_tr)
kicl_base_worst = compute_group_metrics(kicl_base.predict(X_te), y_te, is_min_te)['worst']

kicl_mod = KernelICLClassifier(lam=8.0, tau=0.95, modulation='tail')
kicl_mod.fit(X_tr, y_tr)
kicl_mod_worst = compute_group_metrics(kicl_mod.predict(X_te), y_te, is_min_te)['worst']

tfm_results['KernelICL Baseline'] = kicl_base_worst
tfm_results['KernelICL Tail-Mod (τ=0.95, λ=8)'] = kicl_mod_worst

# 2. TabPFN Foundation Model
if HAS_TABPFN:
    print("Running TabPFN inference...")
    # Subsample for responsive in-context inference
    sub_size = min(200, len(X_tr))
    sub_idx = rng.choice(len(X_tr), size=sub_size, replace=False)
    te_sub = rng.choice(len(X_te), size=min(200, len(X_te)), replace=False)
    
    X_sub = np.ascontiguousarray(X_tr[sub_idx], dtype=np.float32)
    y_sub = np.ascontiguousarray(y_tr[sub_idx], dtype=np.int32)
    X_te_sub = np.ascontiguousarray(X_te[te_sub], dtype=np.float32)
    y_te_sub = y_te[te_sub]
    is_min_te_sub = is_min_te[te_sub]
    
    # Baseline TabPFN
    tabpfn_base = TabPFNClassifier(device=DEVICE)
    tabpfn_base.fit(X_sub, y_sub)
    pred_tabpfn = tabpfn_base.predict(X_te_sub)
    tabpfn_base_worst = compute_group_metrics(pred_tabpfn, y_te_sub, is_min_te_sub)['worst']
    tfm_results['TabPFN Baseline'] = tabpfn_base_worst
    
    # TabPFN with LOO Prompt Resampling (Eq. 11)
    K_sub = gaussian_kernel(X_sub, X_sub, gamma)
    L_sub = incontext_loo_nll(K_sub, y_sub)
    mult_sub = tail_multiplier(L_sub, lam=8.0, tau=0.95)
    X_res, y_res = resample_context(X_sub, y_sub, mult_sub, size=len(X_sub), rng=rng)
    X_res = np.ascontiguousarray(X_res, dtype=np.float32)
    y_res = np.ascontiguousarray(y_res, dtype=np.int32)
    
    tabpfn_res = TabPFNClassifier(device=DEVICE)
    tabpfn_res.fit(X_res, y_res)
    pred_res = tabpfn_res.predict(X_te_sub)
    tabpfn_res_worst = compute_group_metrics(pred_res, y_te_sub, is_min_te_sub)['worst']
    tfm_results['TabPFN LOO-Resampled (τ=0.95)'] = tabpfn_res_worst
    print(f"TabPFN Baseline Worst-Group:   {tabpfn_base_worst:.3f}")
    print(f"TabPFN Resampled Worst-Group:  {tabpfn_res_worst:.3f}")

# 3. TabICL Foundation Model
if HAS_TABICL:
    print("Running TabICLv2 inference...")
    tabicl_base = TabICLClassifier(device=DEVICE)
    tabicl_base.fit(X_sub, y_sub)
    pred_tabicl = tabicl_base.predict(X_te_sub)
    tabicl_base_worst = compute_group_metrics(pred_tabicl, y_te_sub, is_min_te_sub)['worst']
    tfm_results['TabICL Baseline'] = tabicl_base_worst
    print(f"TabICL Baseline Worst-Group:   {tabicl_base_worst:.3f}")

# Visual Comparison of Tabular Foundation Models
fig, ax = plt.subplots(figsize=(10, 5))
models = list(tfm_results.keys())
accuracies = list(tfm_results.values())
bars = ax.barh(models, accuracies, color=['#a6cee3', '#1f77b4', '#b2df8a', '#33a02c', '#fdbf6f'][:len(models)])
ax.set_xlim(0, 1.0)
ax.set_title("Tabular Foundation Models: Baseline vs. LOO-Modulation", fontweight='bold')
ax.set_xlabel("Worst-Group Accuracy")
for bar, acc in zip(bars, accuracies):
    ax.text(acc + 0.02, bar.get_y() + bar.get_height()/2, f"{acc:.3f}", va='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 7: EMBEDDING NEIGHBORHOOD GEOMETRY
    # =========================================================================
    md("""## Module 7: Foundation Model Representation Geometry Analysis

Section 4.5 of the paper presents a fascinating conjecture:
> *"Whether foundation-model representations systematically place minority subpopulations in same-label neighborhoods is a testable hypothesis about representation geometry — measure, per point, the fraction of its top-$k$ embedding neighbors that share its label, and compare minority against majority."*

If minority points are embedded among same-label majority neighbors, their neighbors mostly agree with them, causing LOO difficulty $L_i$ to remain deceptively low! Let us test this hypothesis.""")

    code("""# Representation Geometry: Top-k Embedding Neighborhood Label Agreement
if HAS_TABICL:
    print("Extracting symmetric foundation model embeddings...")
    sub_ctx = 150
    X_sub_geom = np.ascontiguousarray(X_tr[:sub_ctx], dtype=np.float32)
    y_sub_geom = np.ascontiguousarray(y_tr[:sub_ctx], dtype=np.int32)
    is_min_sub = is_min_tr[:sub_ctx]
    
    clf_tabicl = TabICLClassifier(device=DEVICE)
    E_train, _ = extract_symmetric_embeddings(clf_tabicl, X_sub_geom, y_sub_geom, X_sub_geom[:2], device=DEVICE)
    
    k_neighbors = 5
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(E_train)
    distances, indices = nbrs.kneighbors(E_train)
    
    # Exclude self at index 0
    neighbor_labels = y_sub_geom[indices[:, 1:]]
    same_label_ratio = (neighbor_labels == y_sub_geom[:, None]).mean(axis=1)
    
    maj_agreement = same_label_ratio[~is_min_sub].mean()
    min_agreement = same_label_ratio[is_min_sub].mean()
    
    print(f"Top-{k_neighbors} Neighbor Label Agreement (Majority Points): {maj_agreement:.3f}")
    print(f"Top-{k_neighbors} Neighbor Label Agreement (Minority Points): {min_agreement:.3f}")
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(same_label_ratio[~is_min_sub], bins=10, alpha=0.6, density=True, label='Majority Points', color='#1f77b4')
    ax.hist(same_label_ratio[is_min_sub], bins=10, alpha=0.7, density=True, label='Minority Points', color='#d62728')
    ax.set_title(f"TabICL Representation Geometry: Neighbor Label Agreement (k={k_neighbors})", fontweight='bold')
    ax.set_xlabel("Fraction of k-NN Sharing True Label")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("TabICL not available for embedding geometry test.")""")

    # =========================================================================
    # MODULE 8: ZERO-SHOT LABEL NOISE DETECTOR
    # =========================================================================
    md("""## Module 8: Zero-Shot Label-Noise Detector & Structural Averaging Paradox

Section 4.6 introduces the exact LOO score as a **zero-shot, training-free label-noise detector**:
- Injected label noise (20–30%) on 2-D and 20-D datasets is detected with AUC **0.92–0.99** without any clean validation split!
- **The Structural Averaging Paradox:** Pruning mislabeled points on the self-normalized kernel head fails to improve accuracy, because the head already averages out label noise!
- However, feeding the LOO-cleaned prompt to a noise-sensitive model (**1-Nearest Neighbor**) recovers the full noise gap (recovering from $0.594 \\to 0.729$)!

Let us reproduce **Table 5** from the paper.""")

    code("""# Table 5 Reproduction: Zero-Shot Label Noise Detection and Intervention
X_tr_clean, y_tr_clean, y_noisy, is_flipped, X_te_noise, y_te_noise = make_noisy_label_split(
    seed=42, n_train=600, n_test=1200, n_noise_dims=0, eps_noise=0.20
)

# Bandwidth and Gram matrix
gamma_noise = default_gamma(2)
K_noisy = gaussian_kernel(X_tr_clean, X_tr_clean, gamma_noise)
K_te_noisy = gaussian_kernel(X_te_noise, X_tr_clean, gamma_noise)

# Detection Scores
L_soft = incontext_loo_nll(K_noisy, y_noisy)
L_gp_noise = gp_loo_logdensity(K_noisy, y_noisy, sigma2=0.1)

auc_soft = roc_auc_score(is_flipped, L_soft)
auc_gp_n = roc_auc_score(is_flipped, L_gp_noise)

print("=== TABLE 5: LABEL NOISE DETECTION AUC ===")
print(f"Soft-Vote Score Detection AUC: {auc_soft:.3f}")
print(f"GP Score Detection AUC:        {auc_gp_n:.3f}")

# 1-NN Noise-Sensitive Consumer Experiment
q_prune = int(0.20 * len(y_noisy))
top_flagged = np.argsort(L_soft)[-q_prune:]
keep_clean = np.ones(len(y_noisy), dtype=bool)
keep_clean[top_flagged] = False

knn_clean = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean, y_tr_clean).score(X_te_noise, y_te_noise)
knn_noisy = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean, y_noisy).score(X_te_noise, y_te_noise)
knn_loo   = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean[keep_clean], y_noisy[keep_clean]).score(X_te_noise, y_te_noise)
knn_oracle= KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean[~is_flipped], y_noisy[~is_flipped]).score(X_te_noise, y_te_noise)

print("\\n=== NOISE-SENSITIVE CONSUMER (1-NN, 20% NOISE) ===")
print(f"Clean Context Ceiling:   {knn_clean:.3f}")
print(f"Noisy Baseline:          {knn_noisy:.3f}")
print(f"LOO-Cleaned Context:     {knn_loo:.3f}")
print(f"Oracle (Drop Flipped):   {knn_oracle:.3f}")

# Visualization: ROC Curve and 1-NN Recovery
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fpr_s, tpr_s, _ = roc_curve(is_flipped, L_soft)
fpr_g, tpr_g, _ = roc_curve(is_flipped, L_gp_noise)
axes[0].plot(fpr_s, tpr_s, label=f'Soft-Vote Score (AUC={auc_soft:.3f})', lw=2, color='#1f77b4')
axes[0].plot(fpr_g, tpr_g, label=f'GP Score (AUC={auc_gp_n:.3f})', lw=2, color='#ff7f0e')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title("Label Noise Detection ROC Curves", fontweight='bold')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Bar chart of 1-NN Accuracy Recovery
bars = axes[1].bar(['Clean', 'Noisy', 'LOO-Cleaned', 'Oracle'], [knn_clean, knn_noisy, knn_loo, knn_oracle],
            color=['#2ca02c', '#d62728', '#1f77b4', '#9467bd'])
axes[1].set_ylim(0.5, 1.0)
axes[1].set_title("1-NN Accuracy Recovery via Zero-Shot LOO Data Cleaning", fontweight='bold')
axes[1].set_ylabel("Test Accuracy")
for b in bars:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 9: CENSORED LIKELIHOODS & TOBIT
    # =========================================================================
    md("""## Module 9: Censored Likelihoods & Tobit Stress Test (Section 5)

Tabular Foundation Models are pretrained on fully-observed targets. When applied to survival analysis or censored physical outcomes ($y^* \\ge u$), standard models treat $u$ as an exact coordinate, creating severe bias.

Section 5 attaches a **Tobit (censored-normal) likelihood** to the explicit kernel head:
$$p(y_i \\mid f_i, c_i) = \\begin{cases} 
\\Phi\\left(\\frac{\\ell_i - f_i}{\\sigma}\\right) & c_i = -1 \\text{ (left-censored)} \\\\[6pt]
\\frac{1}{\\sigma} \\phi\\left(\\frac{y_i - f_i}{\\sigma}\\right) & c_i = 0 \\text{ (uncensored)} \\\\[6pt]
1 - \\Phi\\left(\\frac{u_i - f_i}{\\sigma}\\right) & c_i = 1 \\text{ (right-censored)}
\\end{cases}$$

Under a Gaussian approximation, leave-one-out predictive densities and scores are computable in closed form.""")

    code("""# Tobit Censored Likelihood Implementation & Diagnostic
f_latent = np.linspace(-3, 3, 200)
y_obs = 1.0
sigma = 1.0

# Evaluate log-likelihoods across latent values
ll_uncensored = tobit_log_likelihood(np.full_like(f_latent, y_obs), f_latent, np.zeros_like(f_latent))
ll_right_cens = tobit_log_likelihood(np.full_like(f_latent, y_obs), f_latent, np.ones_like(f_latent))
ll_left_cens  = tobit_log_likelihood(np.full_like(f_latent, y_obs), f_latent, -np.ones_like(f_latent))

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(f_latent, ll_uncensored, label='Uncensored Normal Likelihood (c=0)', lw=2, color='#1f77b4')
ax.plot(f_latent, ll_right_cens, label='Right-Censored: y* ≥ 1.0 (c=+1)', lw=2, color='#2ca02c')
ax.plot(f_latent, ll_left_cens, label='Left-Censored: y* ≤ 1.0 (c=-1)', lw=2, color='#d62728')
ax.set_title("Tobit Head: Censored Likelihood Profiles for Observed Threshold y = 1.0", fontweight='bold')
ax.set_xlabel("Latent Head Function Value f(x)")
ax.set_ylabel("Log Likelihood")
ax.legend()
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 10: PRACTITIONER CHEAT SHEET & CONCLUSION
    # =========================================================================
    md("""## Module 10: Theoretical Summary & Practitioner's Cheat Sheet

### Summary of Theoretical Principles & Guidelines

| Theoretical Principle | Finding & Diagnostic Rule |
| :--- | :--- |
| **Exactness of Linear Smoother LOO** | For self-normalized kernel heads $\\hat{y} = Sy$, leave-one-out scores are mathematically exact via the PRESS identity in $\\mathcal{O}(n^2)$ without retraining or approximations. |
| **Difficulty $\\neq$ Influence $\\neq$ Intervention Value** | Never conflate these three objects. A point can have high difficulty ($L_i$) while having zero or harmful intervention value. |
| **Tail Precision Governs Accuracy** | Reweighting's sign depends on tail precision: set $\\tau$ to match the estimated minority rate ($1 - p_{\\text{min}}$). Mis-setting $\\tau$ inverts the sign of the effect! |
| **Modulation Strategy** | Use bounded tail-set modulation ($m_i = 1 + \\lambda \\mathbb{I}[L_i \\ge Q_\\tau(L)]$) or pseudo-group balancing. **Avoid exponential weighting**, which collapses mass and predictions to chance. |
| **Auditing Foundation Models** | Decompose context deletion into frozen-head effect + representation-mediated discrepancy (Eq. 6). Frozen heads cannot see upstream representation shifts. |
| **Label Noise Data Cleaning** | Closed-form LOO is a zero-shot label-noise detector with $\\text{AUC} > 0.95$. Data pruning does not benefit kernel smoothers (which average noise away), but completely restores noise-sensitive consumers like 1-NN. |

---
*Created as part of the Kernel-LOUIS package supporting **"Leave No One Out in Context: Exact Leave-One-Out Scores for Interpretable Tabular Foundation Models"**.*""")

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "leave_no_one_out_tfm.ipynb",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Save to both root and notebooks directory
    paths = [
        "/Users/bromia/projects/interpret_tfm/leave_no_one_out_tfm.ipynb",
        "/Users/bromia/projects/interpret_tfm/notebooks/leave_no_one_out_tfm.ipynb"
    ]
    for p in paths:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)
        print(f"Saved notebook to {p}")

if __name__ == "__main__":
    create_notebook()
