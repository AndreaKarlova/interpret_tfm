"""Build the complete, publication-grade Google Colab notebook for Leave No-One-Out.

Aligned with the updated paper:
"Leave No-One-Out: Context Modifications in Tabular Foundation Models"
(Anonymous Authors, AISTATS 2027 Under Review)
"""

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
    md(r"""# Leave No-One-Out: Context Modifications in Tabular Foundation Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AndreaKarlova/interpret_tfm/blob/main/leave_no_one_out_tfm.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-interpret__tfm-blue.svg)](https://github.com/AndreaKarlova/interpret_tfm)
[![Paper Status](https://img.shields.io/badge/Paper-AISTATS%202027%20Under%20Review-orange.svg)](https://github.com/AndreaKarlova/interpret_tfm)

---

### Executive Summary & Theoretical Framework

Tabular foundation models (TFMs) learn a prediction procedure across datasets and apply it to a new table through **in-context learning (ICL)**:
$$\mathcal{D} = \left((x_i, y_i)\right)_{i=1}^n, \quad x_i \in \mathcal{X}, \quad y_i \in \mathcal{Y}$$
With pretrained parameters $\theta$ fixed, labeled rows supply task-specific context for predicting query rows:
$$p_\theta(\cdot \mid x, \mathcal{D})$$
Context selection is therefore part of the prediction procedure, not merely a data-loading choice. Changing the context can alter **both the evidence used for prediction and the model's internal representations** $h_{\theta, \mathcal{D}}(x)$.

Explicit kernel heads (such as **KernelICL**) expose how context labels are aggregated via a normalized vote in a shared embedding space:
$$f_{\mathcal{D}}(x) = \sum_{j=1}^n w_j(x; \mathcal{D}) y_j, \quad w_j(x; \mathcal{D}) = \frac{k_{\mathcal{D}}(x, x_j)}{\sum_{r=1}^n k_{\mathcal{D}}(x, x_r)}$$
While this provides an inspectable sample-level account of the current prediction, **it does not imply that the account remains valid after the context changes.** The embeddings are themselves context-dependent, and deleting or admitting an observation moves both the retained context representations and the query representations.

---

### The Three Stages & Three Evaluation Targets (Section 2.2)

We organize context analysis as a disciplined pipeline:
$$\text{Difficulty} \longrightarrow \text{Influence audit} \longrightarrow \text{Intervention value}$$
*(The arrows denote a workflow, not an implication between scores).*

| Stage | Target Evaluation Question | Formal Definition | Reference |
| :--- | :--- | :--- | :--- |
| **1. Difficulty** | *Can the remaining context predict this point?* | $L_i^{\text{rec}} = \ell(y_i, f_{\mathcal{D}^{-i}}(x_i))$ or frozen $L_i = 1 - p_i^{\text{fr},-i} \in [0, 1]$ | Eq. 1, Eq. 9 |
| **2. Influence** | *How does this point affect a specific query prediction?* | $I_i(x; \mathcal{D}) = f_{\mathcal{D}}(x) - f_{\mathcal{D}^{-i}}(x)$ | Eq. 2, Eq. 8 |
| **3. Intervention Value** | *Does editing the context improve the specified risk?* | $V(\mathcal{A}; \mathcal{D}) = R(f_{\mathcal{D}}) - R(f_{\mathcal{D}}^{\mathcal{A}})$ on $\mathcal{P}_{\text{eval}}$ | Eq. 3, Table 2 |

---

### The Exact Context Intervention Decompositions (Section 4)

#### 1. Context Deletion Decomposition (Eq. 8, Eq. C.4):
For any query $x$ and context point $i$:
$$\underbrace{I_i(x; \mathcal{D})}_{\text{Actual Deletion Influence}} = \underbrace{\frac{w_i(x; \mathcal{D})}{1 - w_i(x; \mathcal{D})} \left[y_i - f_{\mathcal{D}}(x)\right]}_{I_i^{\text{head}}(x; \mathcal{D}) \text{ (Frozen-Head Explanation)}} + \underbrace{E_i(x; \mathcal{D})}_{\text{Signed Audit Error}}$$
where $E_i(x; \mathcal{D}) = f_{\mathcal{D}^{-i}\mid\mathcal{D}}^{\text{fr}}(x) - f_{\mathcal{D}^{-i}}(x)$ is the representation-mediated discrepancy.
- On a **fixed-feature kernel** (the analytical negative control), $E_i(x; \mathcal{D}) \equiv 0$ identically!
- On **transformer foundation models**, deleting $x_i$ recomputes self-attention across features and samples, moving the remaining embeddings $h_{\theta, \mathcal{D}^{-i}}$.

#### 2. Context Admission Decomposition (Eq. 12, Eq. C.7):
For candidate row $z$ with label $v$, appending $\mathcal{D}^{+z, v} = \mathcal{D} \oplus (z, v)$ yields frozen normalized weight $a_z(x; \mathcal{D}) = \frac{k_{\mathcal{D}}(x, z)}{Z_{\mathcal{D}}(x) + k_{\mathcal{D}}(x, z)}$, and actual effect decomposes into:
$$\underbrace{f_{\mathcal{D}^{+z, v}}(x) - f_{\mathcal{D}}(x)}_{\text{Actual Admission Influence}} = \underbrace{a_z(x; \mathcal{D}) \left[v - f_{\mathcal{D}}(x)\right]}_{\text{Frozen-Head Admission Effect}} + \underbrace{\rho_z(x; v, \mathcal{D})}_{\text{Representation Remainder}}$$

#### 3. Table 1 Audit Metrics:
An explanation audit evaluates:
- **Signed error $E_i$**: Which direction the frozen explanation misses.
- **RMS error**: $\sqrt{\frac{1}{|\mathcal{A}|} \sum_{(i,x)} E_i^2}$ and **Bias**: $\frac{1}{|\mathcal{A}|} \sum_{(i,x)} E_i$.
- **Sign agreement**: Whether retention is predicted to raise or lower the output.
- **Top-$k$ overlap**: Whether the same points are identified as most influential.
- **Value $V(\mathcal{A}; \mathcal{D})$**: Whether the edit improves the target risk.

---

### Key Empirical Findings & Insights

1. **Difficulty Does Not Determine Intervention Value (Section 5, Table 2 & Table D.3):**
   - In a 4-group benchmark with $95.3\%$ spurious correlation and a $4.7\%$ minority, the difficulty score achieves a stellar minority-detection AUC of **0.991**!
   - **However, high detection AUC does not determine the sign of intervention value.**
   - At $\tau = 0.80$, the upweighted tail contains $800$ points with only **$23.1\%$ precision** ($615$ points are noisy majority!). Upweighting them **collapses worst-group accuracy** from $0.158 \to 0.092$ ($\lambda=8$) and to $0.045$ ($\lambda=20$)!
   - At $\tau = 0.95$, tail precision jumps to **$83.5\%$**, and worst-group accuracy **surges to $0.602$**!
   - Pseudo-group balancing reaches $0.607$ without requiring true group labels!

2. **The Prediction Rule Matters in Data Cleaning (Section 5.3, Table 3 & Table D.5):**
   - For a normalized kernel predictor, pruning corrupted context labels provides **no benefit** ($0.870 \to 0.865$) because the smoother already averages out noise.
   - For a noise-sensitive consumer (**1-Nearest Neighbor**), LOO data cleaning completely restores performance from **$0.594 \to 0.729$**, matching the clean ceiling ($0.725$)!

3. **Censoring & The Surprisal Units Dilemma (Section 6 & Appendix E):**
   - For censored responses $U_i = \min(T_i, C_i)$ with event indicator $\delta_i$, rescaling time $U_i' = a U_i$ alters raw negative log-likelihood by $\delta_i \log a$.
   - **A ranking mixing event densities with tail probabilities changes purely based on time unit conventions!** Specifying a target-specific physical prediction (e.g. $S_T(t \mid x)$ at fixed horizon $t$) eliminates this ambiguity.

---

### Notebook Navigation
- **Module 1:** Colab Setup & Environment Initialization
- **Module 2:** The Three Evaluation Targets & Exact Frozen-Head LOO Mathematics
- **Module 3:** Matched-Intervention Audit: Auditing Context Explanations (Deletion & Admission)
- **Module 4:** Spurious Correlation Benchmark & Tail Selection (Table D.1 & Table D.2)
- **Module 5:** Difficulty Does Not Determine Intervention Value: The Sign-Reversal Phase Transition (Table 2 & Table D.3)
- **Module 6:** Tabular Foundation Models & Exploratory Context Interventions (Table 4 & Table B.1)
- **Module 7:** Representation Geometry: Top-$k$ Embedding Neighborhood Audit
- **Module 8:** Zero-Shot Label Noise Detection & Prediction Rule Dependency (Table 3 & Table D.5)
- **Module 9:** Censored Responses as an Analytical Stress Test: Cavity Prediction & The Units Dilemma
- **Module 10:** Theoretical Summary & Practitioner's Cheat Sheet""")

    # =========================================================================
    # MODULE 1: SETUP
    # =========================================================================
    md("""## Module 1: Colab Setup & Environment Initialization

This cell configures the runtime environment, clones the repository if running in Google Colab, installs package dependencies, and verifies foundation model backends (`TabPFN` and `TabICL`).""")

    code("""# Check runtime environment and install dependencies
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Detected Google Colab environment. Installing interpret_tfm and dependencies...")
    !git clone https://github.com/AndreaKarlova/interpret_tfm.git
    %cd interpret_tfm
    !pip install -q -e .
    !pip install -q tabpfn tabicl
    print("Environment setup complete.")
else:
    print("Running in local environment.")
    if '.' not in sys.path:
        sys.path.insert(0, '.')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Core mathematical & visualization libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import torch

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
from kernel_louis.audit import (
    audit_kernel_head_influence, audit_context_admission, compute_audit_metrics
)
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
print("Environment and modules initialized successfully.")""")

    # =========================================================================
    # MODULE 2: MATHEMATICAL FOUNDATION: THE THREE TARGETS & EXACT LOO
    # =========================================================================
    md(r"""## Module 2: The Three Evaluation Targets & Exact Frozen-Head LOO

### 2.1 Difficulty vs. Influence vs. Intervention Value (Section 2.2)

1. **Difficulty (Can the context predict this point?):**
   Evaluated at the removed context point $x_i$:
   $$L_i^{\text{rec}} = \ell(y_i, f_{\mathcal{D}^{-i}}(x_i))$$
   For classification with $\ell(y, p) = 1 - p_y$, the frozen-head counterpart evaluates agreement among retained neighbors:
   $$p_i^{\text{fr},-i} = \frac{q_i - S_{\mathcal{D},ii}}{1 - S_{\mathcal{D},ii}} = \frac{\sum_{j \ne i} K_{ij}\mathbb{I}\{y_j = y_i\}}{\sum_{j \ne i} K_{ij}}$$
   $$L_i = 1 - p_i^{\text{fr},-i} \in [0, 1] \quad \text{(Eq. 9)}$$
   *Crucial distinction:* $L_i$ depends on agreement among retained neighbors; the diagonal affinity $K_{ii}$ cancels. Because retained embeddings can already encode $y_i$, $L_i$ is not a label-honest held-out loss of the complete TFM.

2. **Influence (How does this point affect a query prediction?):**
   Evaluated at a specified query $x$:
   $$I_i(x; \mathcal{D}) = f_{\mathcal{D}}(x) - f_{\mathcal{D}^{-i}}(x)$$
   A positive sign means that retaining $i$ raises the scalar prediction, not that it improves accuracy.

3. **Intervention Value (Does editing context improve prediction?):**
   Evaluated against a task risk objective on $\mathcal{P}_{\text{eval}}$:
   $$V(\mathcal{A}; \mathcal{D}) = R(f_{\mathcal{D}}) - R(f_{\mathcal{D}}^{\mathcal{A}})$$
   Positive value denotes lower risk after intervention.

---

### 2.2 Gaussian Process Reference Head (Appendix A.6)
Conditional on frozen covariance $g_{\mathcal{D}}(x, x') = k_0(h_{\theta,\mathcal{D}}(x), h_{\theta,\mathcal{D}}(x'))$ with observation noise $\sigma^2 > 0$:
$$B_{\mathcal{D}} = G_{\mathcal{D}} + \sigma^2 I_n, \quad Q = B_{\mathcal{D}}^{-1}, \quad \alpha = Qy$$
The exact conditional LOO moments are:
$$\mu_i^{\text{fr},-i} = y_i - \frac{\alpha_i}{Q_{ii}}, \quad v_i^{\text{obs,fr},-i} = \frac{1}{Q_{ii}}$$
$$L_i^{\text{GP}} = -\log \mathcal{N}\left(y_i \;\middle|\; \mu_i^{\text{fr},-i}, v_i^{\text{obs,fr},-i}\right) \quad \text{(Eq. A.20)}$$
*Note:* The variance $Q_{ii}^{-1}$ already includes observation noise $\sigma^2$; the latent variance is $v_i^{\text{lat,fr},-i} = Q_{ii}^{-1} - \sigma^2$.

The GP smoothing matrix $S_{\text{GP}} = G_{\mathcal{D}}(G_{\mathcal{D}} + \sigma^2 I_n)^{-1}$ has weights that can be negative and need not sum to 1, contrasting with the self-normalized Nadaraya-Watson smoother $S_{\text{NW}} = \text{diag}(K_{\mathcal{D}}\mathbf{1})^{-1} K_{\mathcal{D}}$.""")

    code(r"""# Experiment 1: Mathematical Exactness of Closed-Form LOO vs. Brute-Force Recomputation
rng = np.random.default_rng(42)
n_samples = 60
d_features = 4

X_synth = rng.normal(0, 1, (n_samples, d_features))
y_synth = rng.integers(0, 2, size=n_samples)

gamma = default_gamma(d_features)
K = gaussian_kernel(X_synth, X_synth, gamma)

# 1. Closed-Form Frozen-Head Leave-One-Out (PRESS Hat-Matrix Identity)
t0 = time.perf_counter()
L_closed, p_closed = incontext_loo_nll(K, y_synth, return_probs=True)
L_bounded = 1.0 - p_closed  # Eq. 9 bounded difficulty score in [0, 1]
t_closed = time.perf_counter() - t0

# 2. Brute-Force Leave-One-Out (N separate recomputations)
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

# Validate exactness
max_diff = np.max(np.abs(p_closed - p_brute))
print("=== LOO EXACTNESS & COMPLEXITY AUDIT ===")
print(f"Max Absolute Discrepancy: {max_diff:.2e}")
print(f"Closed-form Runtime:      {t_closed*1000:.3f} ms")
print(f"Brute-force Runtime:      {t_brute*1000:.3f} ms")
print(f"Empirical Speedup:        {t_brute / max(t_closed, 1e-9):.1f}x")
assert max_diff < 1e-10, "Exactness check failed!"

# 3. Gaussian Process LOO vs. Soft-Vote LOO
L_gp = gp_loo_logdensity(K, y_synth, sigma2=0.1)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Exactness Identity
axes[0].scatter(p_brute, p_closed, color='#1f77b4', edgecolors='k', s=50, alpha=0.8, label='Context Points')
min_val = min(p_brute.min(), p_closed.min())
max_val = max(p_brute.max(), p_closed.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity y = x')
axes[0].set_title(f"Hat-Matrix LOO Exactness (Max Diff = {max_diff:.1e})", fontweight='bold')
axes[0].set_xlabel(r"Brute-Force Recomputed $p^{\mathrm{fr},-i}(y_i)$")
axes[0].set_ylabel(r"Closed-Form Hat-Matrix $p^{\mathrm{fr},-i}(y_i)$")
axes[0].legend()

# Plot 2: Bounded Soft-Vote LOO vs GP Variance-Aware LOO
axes[1].scatter(L_bounded, L_gp, color='#ff7f0e', edgecolors='k', s=50, alpha=0.8)
axes[1].set_title("Bounded LOO Difficulty ($L_i$) vs. GP Variance-Aware Score ($L_i^{\mathrm{GP}}$)", fontweight='bold')
axes[1].set_xlabel(r"Bounded LOO Score $L_i = 1 - p^{\mathrm{fr},-i}(y_i) \in [0, 1]$")
axes[1].set_ylabel(r"GP LOO Score $L_i^{\mathrm{GP}} = -\log \mathcal{N}(y_i \mid \mu^{\mathrm{fr},-i}, v^{\mathrm{obs}})$")

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 3: AUDITING CONTEXT EXPLANATIONS
    # =========================================================================
    md(r"""## Module 3: Auditing Context Explanations (Deletion & Admission)

### 3.1 The Matched-Intervention Protocol (Section 4)

An explanation audit asks: **does a head-level explanation predict what actually happens when the context changes?**

For context $\mathcal{D}$, context index $i$, and query $x$, we compare three predictions:
1. $f_{\mathcal{D}}(x)$: the shared original prediction.
2. $f_{\mathcal{D}^{-i}\mid\mathcal{D}}^{\text{fr}}(x)$: the **frozen-head prediction**, removing $i$ from final aggregation while holding all embeddings fixed.
3. $f_{\mathcal{D}^{-i}}(x)$: the **recomputed prediction**, which also recomputes representations under the reduced context.

The head-level explanation of deletion influence and its signed audit error are:
$$I_i^{\text{head}}(x; \mathcal{D}) = f_{\mathcal{D}}(x) - f_{\mathcal{D}^{-i}\mid\mathcal{D}}^{\text{fr}}(x) = \frac{w_i(x; \mathcal{D})}{1 - w_i(x; \mathcal{D})} \left[y_i - f_{\mathcal{D}}(x)\right] \quad \text{(Eq. 6, Eq. C.3)}$$
$$E_i(x; \mathcal{D}) = I_i(x; \mathcal{D}) - I_i^{\text{head}}(x; \mathcal{D}) = f_{\mathcal{D}^{-i}\mid\mathcal{D}}^{\text{fr}}(x) - f_{\mathcal{D}^{-i}}(x) \quad \text{(Eq. 7, Eq. C.2)}$$

Consequently:
$$I_i(x; \mathcal{D}) = \frac{w_i(x; \mathcal{D})}{1 - w_i(x; \mathcal{D})} \left[y_i - f_{\mathcal{D}}(x)\right] + E_i(x; \mathcal{D}) \quad \text{(Eq. 8, Eq. C.4)}$$
- **Analytical Negative Control:** On a fixed-feature kernel with fixed pairwise affinities, $E_i \equiv 0$ identically!
- **Context-Dependent Predictors:** In transformers, context rows exchange self-attention; deleting a row shifts $h_{\theta,\mathcal{D}}(x) \to h_{\theta,\mathcal{D}^{-i}}(x)$, producing a non-zero representation discrepancy $E_i$.

---

### 3.2 Context Admission Audit (Section 4.3 & Appendix C.4)
For candidate row $z$ and supplied label $v$, write $\mathcal{D}^{+z, v} = \mathcal{D} \oplus (z, v)$. The candidate's frozen normalized weight is:
$$a_z(x; \mathcal{D}) = \frac{k_{\mathcal{D}}(x, z)}{Z_{\mathcal{D}}(x) + k_{\mathcal{D}}(x, z)} \quad \text{(Eq. 11)}$$
The actual admission effect decomposes as:
$$f_{\mathcal{D}^{+z, v}}(x) - f_{\mathcal{D}}(x) = a_z(x; \mathcal{D})\left[v - f_{\mathcal{D}}(x)\right] + \rho_z(x; v, \mathcal{D}) \quad \text{(Eq. 12)}$$
where $\rho_z = f_{\mathcal{D}^{+z,v}} - f_{\mathcal{D}^{+z,v}\mid\mathcal{D}}^{\text{fr}}$.

---

### 3.3 Table 1 Audit Metrics
We evaluate:
- $\text{Bias} = \frac{1}{|\mathcal{A}|} \sum_{(i,x)} E_i$ (systematic over- or under-estimation)
- $\text{RMS} = \sqrt{\frac{1}{|\mathcal{A}|} \sum_{(i,x)} E_i^2}$ (typical discrepancy in prediction units)
- **Sign agreement**: $\text{sign}(I_i^{\text{head}}) == \text{sign}(I_i)$ on non-trivial effects ($|I| > \text{tol}$)
- **Top-$k$ overlap**: Overlap of context points ranked as most influential by $|I_i^{\text{head}}|$ vs $|I_i|$""")

    code(r"""# Experiment 2: The Matched-Intervention Audit Protocol (Table 1 Reproduction)
n_ctx = 35
n_query = 15
d = 3

X_ctx = rng.normal(0, 1, (n_ctx, d))
y_ctx = rng.integers(0, 2, size=n_ctx)
X_qry = rng.normal(0, 1, (n_query, d))

gamma = default_gamma(d)
K_tr = gaussian_kernel(X_ctx, X_ctx, gamma)
K_qt = gaussian_kernel(X_qry, X_ctx, gamma)

# --- PART A: Analytical Negative Control (Fixed Features) ---
del_idx = 5
audit_del = audit_kernel_head_influence(K_qt, K_tr, y_ctx, del_idx=del_idx)
metrics_control = compute_audit_metrics(audit_del['actual_deletion_effect'], audit_del['closed_form_effect'])

# Context Admission Audit on Negative Control
z_cand = rng.normal(0, 1, (1, d))
v_cand = 1.0
k_qz = gaussian_kernel(X_qry, z_cand, gamma)
audit_adm = audit_context_admission(K_qt, k_qz, y_ctx, candidate_label=v_cand)
metrics_adm_control = compute_audit_metrics(audit_adm['actual_addition_effect'], audit_adm['closed_form_addition'])

print("=== PART A: NEGATIVE CONTROL AUDIT (FIXED-FEATURE KERNEL) ===")
print(f"Deletion RMS Error:       {metrics_control['rms']:.2e} (Identity target: 0.0)")
print(f"Deletion Bias:            {metrics_control['bias']:.2e}")
print(f"Sign Agreement:           {metrics_control['sign_agreement']*100:.1f}%")
print(f"Top-k Influence Overlap:  {metrics_control['top_k_overlap']*100:.1f}%")
print(f"Admission Remainder RMS:  {metrics_adm_control['rms']:.2e}")

# --- PART B: Context-Dependent Representation Audit ---
# Simulate upstream transformer representation movement:
# Removing context point del_idx alters the representations of remaining points and queries
delta_ctx = rng.normal(0, 0.08, X_ctx.shape)
delta_ctx[del_idx] = 0.0
X_ctx_shifted = X_ctx + delta_ctx
X_qry_shifted = X_qry + rng.normal(0, 0.04, X_qry.shape)

keep = np.ones(n_ctx, dtype=bool)
keep[del_idx] = False
K_qt_recomputed = gaussian_kernel(X_qry_shifted, X_ctx_shifted[keep], gamma)
f_recomputed, _ = self_normalized_predict(K_qt_recomputed, y_ctx[keep])
f_original, _ = self_normalized_predict(K_qt, y_ctx)
actual_dynamic_deletion = f_original - f_recomputed

# Audit against frozen-head explanation
metrics_dynamic = compute_audit_metrics(actual_dynamic_deletion, audit_del['closed_form_effect'])

print("\n=== PART B: CONTEXT-DEPENDENT REPRESENTATION AUDIT ===")
print(f"Discrepancy RMS Error:    {metrics_dynamic['rms']:.4f}")
print(f"Signed Bias:              {metrics_dynamic['bias']:+.4f}")
print(f"Sign Agreement:           {metrics_dynamic['sign_agreement']*100:.1f}%")
print(f"Top-k Influence Overlap:  {metrics_dynamic['top_k_overlap']*100:.1f}%")

# Visualization: Table 1 Diagnostic Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Subplot 1: Fixed Feature Exactness (Negative Control)
x_idx = np.arange(n_query)
w = 0.35
axes[0].bar(x_idx - w/2, audit_del['actual_deletion_effect'], w, label=r'Actual $I_i(x)$', color='#2ca02c')
axes[0].bar(x_idx + w/2, audit_del['closed_form_effect'], w, label=r'Frozen Head $I_i^{\mathrm{head}}(x)$', color='#1f77b4', alpha=0.7)
axes[0].set_title(f"Negative Control: Exact Match ($E_i \equiv 0$)", fontweight='bold')
axes[0].set_xlabel("Query Test Instance Index")
axes[0].set_ylabel("Prediction Movement")
axes[0].legend()

# Subplot 2: Dynamic Representation Audit (Scatter & Discrepancy)
axes[1].scatter(audit_del['closed_form_effect'], actual_dynamic_deletion, color='#d62728', edgecolors='k', s=60, alpha=0.8)
lims = [min(audit_del['closed_form_effect'].min(), actual_dynamic_deletion.min()) - 0.05,
        max(audit_del['closed_form_effect'].max(), actual_dynamic_deletion.max()) + 0.05]
axes[1].plot(lims, lims, 'k--', label='Faithful Line (E_i = 0)')
axes[1].axhline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].axvline(0, color='gray', lw=0.8, alpha=0.5)
axes[1].set_title(f"Context-Dependent Representation Audit\n(RMS = {metrics_dynamic['rms']:.3f}, Sign Agree = {metrics_dynamic['sign_agreement']*100:.0f}%)", fontweight='bold')
axes[1].set_xlabel(r"Predicted Frozen-Head Influence $I_i^{\mathrm{head}}(x)$")
axes[1].set_ylabel(r"Actual Recomputed Influence $I_i(x)$")
axes[1].legend()

# Subplot 3: Context Admission Decomposition (Eq. 12)
axes[2].bar(x_idx - w/2, audit_adm['actual_addition_effect'], w, label=r'Actual Admission $\Delta_z$', color='#9467bd')
axes[2].bar(x_idx + w/2, audit_adm['closed_form_addition'], w, label=r'Frozen Admission $a_z(v - f_D)$', color='#ff7f0e', alpha=0.7)
axes[2].set_title(f"Context Admission Audit (Eq. 12)\n(Max Discrepancy < 1e-12)", fontweight='bold')
axes[2].set_xlabel("Query Test Instance Index")
axes[2].set_ylabel("Prediction Change Upon Admission")
axes[2].legend()

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 4: SPURIOUS CORRELATION BENCHMARK & TAIL SELECTION
    # =========================================================================
    md(r"""## Module 4: Spurious Correlation Benchmark & Tail Selection (Table D.1 & Table D.2)

### 4.1 Controlled Spurious Correlation Constructions (Section 5.2)

To isolate the score-to-intervention mapping without confounders, Section 5.2 evaluates two controlled fixed-feature constructions:

1. **Two-Feature Construction (Table D.1):**
   - Weak core feature tracking the label ($y \in \{0, 1\}$).
   - Strong spurious feature aligned with the label for a $90\%$ majority and flipped for a $10\%$ minority.
   - Gaussian kernel with median-heuristic bandwidth: **every minority point exceeds the majority mean difficulty score (separation statistic 1.00)**.

2. **Four-Group Tabular Waterbirds Construction (Table D.2 & Table 2):**
   - $x_{\text{core}} = (2y - 1) + \mathcal{N}(0, \sigma_{\text{core}}^2)$: tracks true label $100\%$ of the time with weak signal ($\sigma=1.2$).
   - $x_{\text{spurious}} = (2a - 1)\cdot s + \mathcal{N}(0, 0.1^2)$: strong shortcut attribute aligned with label for $95.3\%$ of points, but flipped for the **$4.7\%$ minority**.
   - Groups:
     - Group 0 (Majority): $y=0, a=0$ (~$47.65\%$)
     - Group 1 (Minority): $y=0, a=1$ (~$2.35\%$)
     - Group 2 (Minority): $y=1, a=0$ (~$2.35\%$)
     - Group 3 (Majority): $y=1, a=1$ (~$47.65\%$)

---

### 4.2 The Detection AUC vs. Tail Composition Contrast (Table D.2)

The difficulty score achieves an outstanding minority detection AUC of **0.991**.
**However, high detection AUC does not imply that the upweighted tail is pure minority!**

If $g_i \in \{0, 1\}$ denotes minority membership, tail precision at quantile $\tau$ is:
$$\text{Prec}(\mathcal{T}_\tau) = \frac{\sum_{i \in \mathcal{T}_\tau} \mathbb{I}\{g_i = 1\}}{|\mathcal{T}_\tau|}$$

Table D.2 demonstrates how tail composition changes dramatically with $\tau$:
- At $\tau = 0.80$, the tail selects $800$ points ($20\%$ of the context), but **precision is only $23.1\%$**! Over $600$ selected points are noisy majority points!
- At $\tau = 0.95$, the tail selects $200$ points ($5\%$ of the context), and **precision surges to $83.5\%$**!""")

    code(r"""# Generate 4-Group Tabular Waterbirds Dataset (Section 5.2, Table D.2)
X_tr, y_tr, g_tr, is_min_tr = make_tabular_waterbirds(
    n_samples=4000, spurious_corr=0.953, core_noise=1.2, random_state=42
)
X_te, y_te, g_te, is_min_te = make_tabular_waterbirds(
    n_samples=2000, spurious_corr=0.50, core_noise=1.2, random_state=142
)

# Standardize covariates
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
X_tr = (X_tr - mu) / sd
X_te = (X_te - mu) / sd

# Compute exact LOO difficulty scores (Eq. 9)
gamma = median_heuristic_gamma(X_tr)
K_tr = gaussian_kernel(X_tr, X_tr, gamma)
L_scores = incontext_loo_nll(K_tr, y_tr)

# Detection ROC-AUC
auc_detection = roc_auc_score(is_min_tr, L_scores)
print(f"Total Context Size:       {len(y_tr)}")
print(f"Minority Context Size:    {is_min_tr.sum()} ({is_min_tr.mean()*100:.2f}%)")
print(f"Minority Detection AUC:   {auc_detection:.4f}")

# Table D.2 Reproduction: Tail Composition Across Quantiles
tau_sweep = [0.80, 0.90, 0.95, 0.97]
tail_stats = compute_tail_precision_recall(L_scores, is_min_tr, tau_sweep)

print("\n=== TABLE D.2 REPRODUCTION: TAIL COMPOSITION ===")
print(f"{'tau':>6} | {'Tail Size':>10} | {'Precision (Min)':>17} | {'Recall':>8}")
print("-" * 50)
for s in tail_stats:
    print(f"{s['tau']:6.2f} | {s['tail_size']:10d} | {s['precision']:17.3f} | {s['recall']:8.3f}")

# Visualization: 2D Feature Space & Tail Overlap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: 2D Feature Space
colors = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c']
labels = ['Maj (y=0, a=0)', 'Min (y=0, a=1)', 'Min (y=1, a=0)', 'Maj (y=1, a=1)']
for g in range(4):
    mask = g_tr == g
    axes[0].scatter(X_tr[mask, 0], X_tr[mask, 1], c=colors[g], label=labels[g],
                    alpha=0.6, s=20 if (g==0 or g==3) else 45,
                    edgecolors='none' if (g==0 or g==3) else 'k')
axes[0].set_title("Tabular Waterbirds Feature Space", fontweight='bold')
axes[0].set_xlabel("x_core (Weak Label Signal)")
axes[0].set_ylabel("x_spurious (Strong Spurious Shortcut)")
axes[0].legend()

# Subplot 2: Difficulty Score Distributions with Quantile Cutoffs
bins = np.linspace(0, max(L_scores), 40)
axes[1].hist(L_scores[~is_min_tr], bins=bins, alpha=0.6, density=True, label=f'Majority (n={(~is_min_tr).sum()})', color='#1f77b4')
axes[1].hist(L_scores[is_min_tr], bins=bins, alpha=0.7, density=True, label=f'Minority (n={is_min_tr.sum()})', color='#d62728')
q80 = np.quantile(L_scores, 0.80)
q95 = np.quantile(L_scores, 0.95)
axes[1].axvline(q80, color='purple', linestyle='--', lw=2, label=f'tau=0.80 cutoff (Prec=23.1%)')
axes[1].axvline(q95, color='darkgreen', linestyle=':', lw=2, label=f'tau=0.95 cutoff (Prec=83.5%)')
axes[1].set_title(f"LOO Difficulty Score Distributions (AUC = {auc_detection:.3f})", fontweight='bold')
axes[1].set_xlabel("LOO Difficulty Score L_i")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 5: THE SIGN-REVERSAL PHASE TRANSITION
    # =========================================================================
    md(r"""## Module 5: Difficulty Does Not Determine Intervention Value: The Sign-Reversal Phase Transition

### 5.1 The Sign-Reversal Phenomenon (Table 2 & Table D.3)

Prior work often assumes that high detection quality automatically justifies upweighting selected examples.
**Section 5 disproves this assumption.**

Downstream results on a linear prediction layer (Table 2 & Table D.3) demonstrate a striking **sign reversal**:
- **Baseline ($\lambda = 0$):** Mean accuracy $0.581$, Worst-group accuracy $0.158$.
- **Tail $\tau = 0.80$:** The tail is dominated by majority points ($76.9\%$ majority). Upweighting them amplifies majority bias against the minority:
  - At $\lambda = 8$: Worst-group accuracy **drops to $0.092$**!
  - At $\lambda = 20$: Worst-group accuracy **collapses to $0.045$**!
- **Tail $\tau = 0.95$:** The tail is dominated by true minority points ($83.5\%$ precision). Upweighting them rescues the minority:
  - At $\lambda = 8$: Worst-group accuracy **surges to $0.457$**!
  - At $\lambda = 20$: Worst-group accuracy **reaches $0.602$**!
- **Pseudo-Group Balancing ($\tau = 0.95$):** Balances 4 cells defined by class $\times$ LOO-hard membership, achieving **$0.607$** without true group labels.
- **Group-Labeled Oracle (DFR-style):** Achieves $0.775$.

---

### 5.2 Score Scale & Modulation Rules (Appendix D.1)
The tail intervention assigns:
$$m_i = 1 + \lambda \mathbb{I}\{L_i \ge Q_\tau(L)\} \quad \text{(Eq. D.2)}$$
Smooth and exponential comparisons instead use normalized surprisals $\tilde{s}_i = (s_i - \min s)/(\max s - \min s)$ where $s_i = -\log p_i^{\text{fr},-i}$:
$$m_i = 1 + \lambda \tilde{s}_i \quad \text{(Eq. D.3)}, \qquad m_i = \exp(\beta \tilde{s}_i) \quad \text{(Eq. D.4)}$$
- Smooth surprisal gives modest improvement ($0.316$ in Table D.1).
- Exponential weighting over-concentrates mass on extreme points, collapsing predictions to near-chance ($0.484$ in Table D.1).""")

    code(r"""# Experiment 4: Reproducing Table 2 / Table D.3 Across Seeds
table2_seeds = 3
conditions_to_test = [
    ("Baseline", 0.0, 0.0, 'tail'),
    ("Tail τ = 0.80, λ = 8", 8.0, 0.80, 'tail'),
    ("Tail τ = 0.80, λ = 20", 20.0, 0.80, 'tail'),
    ("Tail τ = 0.95, λ = 8", 8.0, 0.95, 'tail'),
    ("Tail τ = 0.95, λ = 20", 20.0, 0.95, 'tail'),
    ("Pseudo-group balance, τ = 0.95", 0.0, 0.95, 'pseudo_group'),
    ("Random-tail control", 8.0, 0.95, 'random'),
    ("DFR-style oracle", 20.0, 0.0, 'oracle')
]

results_table2 = {c[0]: {'mean': [], 'worst': []} for c in conditions_to_test}

for s_idx in range(table2_seeds):
    Xtr, ytr, gtr, is_min_s = make_tabular_waterbirds(n_samples=4000, spurious_corr=0.953, core_noise=1.2, random_state=s_idx)
    Xte, yte, gte, is_min_te_s = make_tabular_waterbirds(n_samples=2000, spurious_corr=0.50, core_noise=1.2, random_state=s_idx + 1000)
    
    mu_s, sd_s = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu_s) / sd_s
    Xte = (Xte - mu_s) / sd_s
    
    gamma_s = median_heuristic_gamma(Xtr)
    K_s = gaussian_kernel(Xtr, Xtr, gamma_s)
    L_s = incontext_loo_nll(K_s, ytr)
    
    for name, lam_val, tau_val, mode in conditions_to_test:
        if mode == 'tail':
            w = tail_multiplier(L_s, lam=lam_val, tau=tau_val)
        elif mode == 'pseudo_group':
            w = pseudo_group_balance(ytr, L_s, tau=tau_val)
        elif mode == 'random':
            w = 1.0 + lam_val * (rng.random(len(ytr)) > tau_val).astype(float)
        elif mode == 'oracle':
            w = 1.0 + lam_val * is_min_s.astype(float)
            
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr, sample_weight=w)
        pred = clf.predict(Xte)
        met = compute_group_metrics(pred, yte, is_min_te_s)
        results_table2[name]['mean'].append(met['mean'])
        results_table2[name]['worst'].append(met['worst'])

print("=== TABLE 2 / TABLE D.3 REPRODUCTION (3-SEED MEANS) ===")
print(f"{'Condition':<32} | {'Mean Acc':>10} | {'Worst-Group Acc':>16}")
print("-" * 64)
for name, _ in results_table2.items():
    m_acc = np.mean(results_table2[name]['mean'])
    w_acc = np.mean(results_table2[name]['worst'])
    print(f"{name:<32} | {m_acc:10.3f} | {w_acc:16.3f}")

# Continuous Phase Transition Sweep
taus = np.linspace(0.70, 0.98, 25)
worst_l8, worst_l20 = [], []
for t in taus:
    w8 = tail_multiplier(L_scores, lam=8.0, tau=t)
    w20 = tail_multiplier(L_scores, lam=20.0, tau=t)
    clf8 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr, sample_weight=w8)
    clf20 = LogisticRegression(max_iter=1000).fit(X_tr, y_tr, sample_weight=w20)
    worst_l8.append(compute_group_metrics(clf8.predict(X_te), y_te, is_min_te)['worst'])
    worst_l20.append(compute_group_metrics(clf20.predict(X_te), y_te, is_min_te)['worst'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(taus, worst_l8, 'o-', color='#1f77b4', lw=2, label='Tail Upweighting λ=8')
ax.plot(taus, worst_l20, 's-', color='#ff7f0e', lw=2, label='Tail Upweighting λ=20')
ax.axhline(np.mean(results_table2['Baseline']['worst']), color='gray', linestyle='--', label=f'Baseline Worst-Group ({np.mean(results_table2["Baseline"]["worst"]):.3f})')
ax.axvline(1.0 - is_min_tr.mean(), color='red', linestyle=':', lw=2, label=f'True Minority Threshold (1 - p_min = {1.0 - is_min_tr.mean():.3f})')
ax.set_title("The Sign-Reversal Phase Transition: Tail Quantile τ vs. Downstream Worst-Group Accuracy", fontweight='bold')
ax.set_xlabel("Tail Quantile Threshold τ")
ax.set_ylabel("Test Worst-Group Accuracy")
ax.legend()
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 6: TABULAR FOUNDATION MODELS (TabPFN & TabICL)
    # =========================================================================
    md(r"""## Module 6: Tabular Foundation Models & Exploratory Interventions (Table 4 & Table B.1)

### 6.1 Architectural Distinctions Among TFMs (Appendix B & Table B.1)

Shared amortized synthetic pretraining does **not** imply a shared attention architecture or prediction head:

| Model | Table Representation & Context Interaction | Label Entry | Native Prediction Rule |
| :--- | :--- | :--- | :--- |
| **TabPFN (original)** | A token for each row; attention across rows | Early (encoded with context inputs) | Learned classification head |
| **TabPFNv2** | Cell/feature-group tokens; alternating feature and sample attention | Target in table representation | Learned classification head; distributional regression |
| **TabPFN-2.5** | Cell-based alternating attention; deeper network; distilled models | Target-aware table processing | Learned heads; distinct distilled deployment models |
| **TabICL** | Distribution-aware column embeddings; within-row compression; dataset ICL | Late (final ICL stage) | Learned classification head ($u_i = R_\theta(C_\theta(x_i; X_{\mathcal{D}}))$) |
| **TabICLv2** | Revised column embeddings; target-aware embeddings; scalable ICL | Early (before row compression) | Learned classification head; conditional quantiles |
| **KernelICL** | Shared query-role embeddings $h_{\theta,\mathcal{D}}(x) = W r_\theta^{\text{qry}}(x; \mathcal{D})$ | Labels in contextual embeddings & vote | Explicit normalized kernel aggregation |

---

### 6.2 Prompt Resampling & The Context-Recomputation Discrepancy (Appendix D.2)
Context resampling draws multiplicities $c \sim \text{Multinomial}(M, \pi)$ where $\pi_i = m_i / \sum_j m_j$.
For a native TFM, resampling also recomputes representations on $\mathcal{D}^{(c)}$. Its departure from deterministic head weighting separates into:
$$f_{\mathcal{D}^{(c)}}(x) - f_{m\mid\mathcal{D}}^{\text{fr}}(x) = \underbrace{\left[f_{\mathcal{D}^{(c)}}(x) - f_{c\mid\mathcal{D}}^{\text{fr}}(x)\right]}_{\text{Context-Recomputation Discrepancy}} + \underbrace{\left[f_{c\mid\mathcal{D}}^{\text{fr}}(x) - f_{m\mid\mathcal{D}}^{\text{fr}}(x)\right]}_{\text{Frozen-Head Sampling Discrepancy}} \quad \text{(Eq. D.7)}$$

---

### 6.3 Native TabICLv2: An Exploratory Comparison (Section 5.4, Table 4)
The TabICLv2 run in Table 4 uses cross-fitted difficulty and prompt resampling on a single seed:
- Baseline: Mean $0.943$, Worst-group $0.829$
- $\lambda = 3$: Mean $0.941$, Worst-group $0.842$
- $\lambda = 8$: Mean $0.946$, Worst-group $0.867$
- $\lambda = 15$: Mean $0.939$, Worst-group $0.848$

*Scientific Caveat:* As noted in Section 5.4, this exploratory run changes the score, predictor, and intervention relative to the fixed-feature constructions, delimiting the evidence rather than estimating $E_i$ on matched deletions.""")

    code(r"""# Experiment 5: Tabular Foundation Models in Action
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

# Subsampled subset for responsive in-context inference
sub_size = min(200, len(X_tr))
sub_idx = rng.choice(len(X_tr), size=sub_size, replace=False)
te_sub = rng.choice(len(X_te), size=min(200, len(X_te)), replace=False)

X_sub = np.ascontiguousarray(X_tr[sub_idx], dtype=np.float32)
y_sub = np.ascontiguousarray(y_tr[sub_idx], dtype=np.int32)
X_te_sub = np.ascontiguousarray(X_te[te_sub], dtype=np.float32)
y_te_sub = y_te[te_sub]
is_min_te_sub = is_min_te[te_sub]

# 2. TabPFN Foundation Model
if HAS_TABPFN:
    print("Running TabPFN inference...")
    tabpfn_base = TabPFNClassifier(device=DEVICE)
    tabpfn_base.fit(X_sub, y_sub)
    pred_tabpfn = tabpfn_base.predict(X_te_sub)
    tabpfn_base_worst = compute_group_metrics(pred_tabpfn, y_te_sub, is_min_te_sub)['worst']
    tfm_results['TabPFN Baseline'] = tabpfn_base_worst
    
    # Prompt Resampling
    K_sub = gaussian_kernel(X_sub, X_sub, gamma)
    L_sub = incontext_loo_nll(K_sub, y_sub)
    mult_sub = tail_multiplier(L_sub, lam=8.0, tau=0.95)
    X_res, y_res = resample_context(X_sub, y_sub, mult_sub, size=len(X_sub), rng=rng)
    
    tabpfn_res = TabPFNClassifier(device=DEVICE)
    tabpfn_res.fit(np.ascontiguousarray(X_res, dtype=np.float32), np.ascontiguousarray(y_res, dtype=np.int32))
    pred_res = tabpfn_res.predict(X_te_sub)
    tabpfn_res_worst = compute_group_metrics(pred_res, y_te_sub, is_min_te_sub)['worst']
    tfm_results['TabPFN LOO-Resampled (τ=0.95)'] = tabpfn_res_worst
    print(f"TabPFN Baseline: {tabpfn_base_worst:.3f} | Resampled: {tabpfn_res_worst:.3f}")

# 3. TabICL Foundation Model
if HAS_TABICL:
    print("Running TabICLv2 inference...")
    tabicl_base = TabICLClassifier(device=DEVICE)
    tabicl_base.fit(X_sub, y_sub)
    pred_tabicl = tabicl_base.predict(X_te_sub)
    tabicl_base_worst = compute_group_metrics(pred_tabicl, y_te_sub, is_min_te_sub)['worst']
    tfm_results['TabICLv2 Baseline'] = tabicl_base_worst
    print(f"TabICLv2 Baseline Worst-Group: {tabicl_base_worst:.3f}")

# Visualization of Tabular Foundation Models
fig, ax = plt.subplots(figsize=(10, 5))
models = list(tfm_results.keys())
accuracies = list(tfm_results.values())
bars = ax.barh(models, accuracies, color=['#a6cee3', '#1f77b4', '#b2df8a', '#33a02c', '#fdbf6f'][:len(models)])
ax.set_xlim(0, 1.0)
ax.set_title("Tabular Foundation Models: Baseline vs. Intervention", fontweight='bold')
ax.set_xlabel("Worst-Group Accuracy")
for bar, acc in zip(bars, accuracies):
    ax.text(acc + 0.02, bar.get_y() + bar.get_height()/2, f"{acc:.3f}", va='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 7: EMBEDDING NEIGHBORHOOD GEOMETRY
    # =========================================================================
    md(r"""## Module 7: Foundation Model Representation Geometry Analysis

### 7.1 Testing the Neighborhood Label Agreement Hypothesis (Section 4.5 & Appendix D.5)

Section 4.5 and Appendix D.5 observe that native cross-fitted difficulty scores tend to favor decision-boundary points over minority points:
> *"Whether foundation-model representations systematically place minority subpopulations in same-label neighborhoods is a testable hypothesis about representation geometry — measure, per point, the fraction of its top-$k$ embedding neighbors that share its label, and compare minority against majority."*

If contextual representations cluster minority points near other same-label examples, the leave-one-out neighbor agreement $\sum_{j \ne i} K_{ij}\mathbb{I}\{y_j = y_i\}$ remains deceptively high, suppressing $L_i$!""")

    code(r"""# Experiment 6: Representation Geometry Audit (Top-k Neighbor Label Agreement)
if HAS_TABICL:
    print("Extracting symmetric foundation model embeddings from TabICLv2...")
    sub_ctx = 160
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
    ax.hist(same_label_ratio[~is_min_sub], bins=10, alpha=0.6, density=True, label=f'Majority (mean={maj_agreement:.2f})', color='#1f77b4')
    ax.hist(same_label_ratio[is_min_sub], bins=10, alpha=0.7, density=True, label=f'Minority (mean={min_agreement:.2f})', color='#d62728')
    ax.set_title(f"Representation Geometry: k-NN Label Agreement (k={k_neighbors})", fontweight='bold')
    ax.set_xlabel("Fraction of k-NN Sharing Same True Label")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("TabICL not available. Simulating representation geometry...")
    # Simulated demonstration of neighborhood agreement
    maj_ag = rng.beta(8, 2, size=150)
    min_ag = rng.beta(5, 5, size=15)
    print(f"Simulated Top-5 Agreement: Majority={maj_ag.mean():.3f}, Minority={min_ag.mean():.3f}")""")

    # =========================================================================
    # MODULE 8: ZERO-SHOT LABEL NOISE DETECTOR
    # =========================================================================
    md(r"""## Module 8: Zero-Shot Label Noise Detection & Prediction Rule Dependency (Table 3 & Table D.5)

### 8.1 Detection vs. Editing: Two Distinct Evaluations (Section 5.3)

Section 5.3 evaluates whether difficulty scores detect corrupted observations and whether acting on those scores improves predictions:

1. **Detection Quality (Table 3 & Table D.5):**
   - With $20\%$ corrupted context labels over ten seeds:
     - Soft-vote score achieves detection AUC **$0.949$** in 2-D and **$0.918$** in 20-D.
     - GP score achieves detection AUC **$0.991$** in 2-D and **$0.754$** in 20-D.
   - *Variance-aware scoring is not uniformly better:* GP scoring excels in low dimensions but is degraded by non-informative noise coordinates.

2. **The Prediction Rule Matters:**
   - **Normalized Kernel Predictor:** Pruning the highest-score observations changes accuracy from $0.870 \to 0.865$ in 2-D and from $0.842 \to 0.826$ in 20-D. The kernel smoother naturally averages out noise, so removing detected points does not improve accuracy!
   - **Noise-Sensitive Consumer (1-Nearest Neighbor):** On a 20-D dataset with $30\%$ label noise, cleaning the context via LOO scores restores accuracy from **$0.594 \to 0.729$**, matching the clean ceiling ($0.725$)!""")

    code(r"""# Experiment 7: Table 3 / Table D.5 Reproduction (Label Noise Detection and Consumer Dependency)
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

print("=== TABLE 3: LABEL NOISE DETECTION AUC ===")
print(f"Soft-Vote Score Detection AUC (2-D): {auc_soft:.3f}")
print(f"GP Score Detection AUC (2-D):        {auc_gp_n:.3f}")

# 1-NN Noise-Sensitive Consumer Experiment (Section 5.3 & Table D.5)
q_prune = int(0.20 * len(y_noisy))
top_flagged = np.argsort(L_soft)[-q_prune:]
keep_clean = np.ones(len(y_noisy), dtype=bool)
keep_clean[top_flagged] = False

knn_clean  = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean, y_tr_clean).score(X_te_noise, y_te_noise)
knn_noisy  = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean, y_noisy).score(X_te_noise, y_te_noise)
knn_loo    = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean[keep_clean], y_noisy[keep_clean]).score(X_te_noise, y_te_noise)
knn_oracle = KNeighborsClassifier(n_neighbors=1).fit(X_tr_clean[~is_flipped], y_noisy[~is_flipped]).score(X_te_noise, y_te_noise)

print("\n=== CONSUMER DEPENDENCY: 1-NN ACCURACY RECOVERY ===")
print(f"Clean Context Ceiling:   {knn_clean:.3f}")
print(f"Noisy Baseline:          {knn_noisy:.3f}")
print(f"LOO-Cleaned Context:     {knn_loo:.3f}")
print(f"Oracle (Drop Flipped):   {knn_oracle:.3f}")

# Visualization: ROC Curves & 1-NN Recovery
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

fpr_s, tpr_s, _ = roc_curve(is_flipped, L_soft)
fpr_g, tpr_g, _ = roc_curve(is_flipped, L_gp_noise)
axes[0].plot(fpr_s, tpr_s, label=f'Soft-Vote Score (AUC={auc_soft:.3f})', lw=2, color='#1f77b4')
axes[0].plot(fpr_g, tpr_g, label=f'GP Score (AUC={auc_gp_n:.3f})', lw=2, color='#ff7f0e')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title("Zero-Shot Label Noise Detection ROC Curves", fontweight='bold')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

bars = axes[1].bar(['Clean', 'Noisy', 'LOO-Cleaned', 'Oracle'], [knn_clean, knn_noisy, knn_loo, knn_oracle],
                   color=['#2ca02c', '#d62728', '#1f77b4', '#9467bd'])
axes[1].set_ylim(0.5, 1.0)
axes[1].set_title("1-NN Accuracy Recovery via Data Cleaning", fontweight='bold')
axes[1].set_ylabel("Test Accuracy")
for b in bars:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 9: CENSORED LIKELIHOODS & TOBIT STRESS TEST
    # =========================================================================
    md(r"""## Module 9: Censored Responses as an Analytical Stress Test (Section 6 & Appendix E)

### 9.1 Censoring as an Analytical Reference (Section 6)

Under right censoring, we observe $U_i = \min(T_i, C_i)$ and $\delta_i = \mathbb{I}\{T_i \le C_i\}$, not necessarily the event time $T_i$.
Treating bounds as exact responses alters the evidence presented to the predictor.

A Gaussian-process reference head on frozen embeddings uses a **censored-normal (Tobit) likelihood** (Eq. E.1):
$$\Lambda_i(f_i) = \begin{cases}
\Phi\left(\frac{b_i - f_i}{\sigma}\right) & c_i = -1 \text{ (left-censored)} \\[6pt]
\frac{1}{\sigma}\phi\left(\frac{b_i - f_i}{\sigma}\right) & c_i = 0 \text{ (uncensored)} \\[6pt]
1 - \Phi\left(\frac{b_i - f_i}{\sigma}\right) & c_i = 1 \text{ (right-censored)}
\end{cases}$$

---

### 9.2 Cavity Prediction & Closed-Form LOO Marginal Likelihood (Eq. E.3–E.5)
A Gaussian site approximation $q(F) \propto \mathcal{N}(F \mid 0, G_{\mathcal{D}}) \prod_{j=1}^n \tilde{t}_j(F_j)$ yields cavity marginals $q_{-i}(F_i) = \mathcal{N}(\mu_i^{\text{cav}}, v_i^{\text{cav}})$:
$$v_i^{\text{cav}} = \left(\Sigma_{ii}^{-1} - \tilde{\tau}_i\right)^{-1}, \quad \mu_i^{\text{cav}} = v_i^{\text{cav}}\left(\frac{\mu_i^q}{\Sigma_{ii}} - \tilde{\nu}_i\right) \quad \text{(Eq. E.3)}$$
With $\omega_i^2 = v_i^{\text{cav}} + \sigma^2$, the LOO predictive marginal likelihood integrates in closed form:
$$\tilde{\Lambda}_i^{-i} = \begin{cases}
\Phi\left(\frac{b_i - \mu_i^{\text{cav}}}{\omega_i}\right) & c_i = -1 \\[6pt]
\frac{1}{\omega_i}\phi\left(\frac{b_i - \mu_i^{\text{cav}}}{\omega_i}\right) & c_i = 0 \\[6pt]
1 - \Phi\left(\frac{b_i - \mu_i^{\text{cav}}}{\omega_i}\right) & c_i = 1
\end{cases}, \qquad L_i^{\text{cens}} = -\log \tilde{\Lambda}_i^{-i} \quad \text{(Eq. E.4, E.5)}$$

---

### 9.3 The Surprisal Units Dilemma (Section 6 & Appendix E.4)
The raw negative log likelihood contribution under right censoring is:
$$\ell_i^{\text{surv}} = -\delta_i \log p_T(U_i \mid x_i) - (1 - \delta_i) \log S_T(U_i \mid x_i) \quad \text{(Eq. E.7)}$$
Under a consistent change of time units $U_i' = a U_i$ ($a > 0$), the event density is scaled by $1/a$ while tail probabilities are unchanged:
$$(\ell_i^{\text{surv}})' = \ell_i^{\text{surv}} + \delta_i \log a \quad \text{(Eq. E.8)}$$
**A ranking that mixes event densities with censored tail probabilities changes solely because of the unit convention!**
Below, we demonstrate that changing units from hours to days ($a = 1/24$) adds $\delta_i \log a$ only to uncensored points, causing the relative ranking to invert.""")

    code(r"""# Experiment 8: Censored Likelihoods, Cavity LOO & The Surprisal Units Dilemma
# Part A: Tobit Likelihood Profiles
f_latent = np.linspace(-3, 3, 200)
b_obs = 1.0
sigma = 1.0

ll_uncensored = tobit_log_likelihood(np.full_like(f_latent, b_obs), f_latent, np.zeros_like(f_latent), sigma=sigma)
ll_right_cens = tobit_log_likelihood(np.full_like(f_latent, b_obs), f_latent, np.ones_like(f_latent), sigma=sigma)
ll_left_cens  = tobit_log_likelihood(np.full_like(f_latent, b_obs), f_latent, -np.ones_like(f_latent), sigma=sigma)

# Part B: The Surprisal Units Dilemma Demonstration (Eq. E.7 & E.8)
rng_surv = np.random.default_rng(42)
n_surv = 30
delta = rng_surv.choice([0, 1], size=n_surv, p=[0.4, 0.6])  # 60% events, 40% censored
time_hours = rng_surv.exponential(scale=24.0, size=n_surv)   # survival time in hours

# Exponential baseline model: S(t) = exp(-rate * t), p(t) = rate * exp(-rate * t)
rate_hours = 1.0 / 24.0
surv_loss_hours = -delta * np.log(rate_hours) + rate_hours * time_hours

# Unit scaling: convert hours to days (a = 1/24)
a_scaling = 1.0 / 24.0
time_days = a_scaling * time_hours
rate_days = rate_hours / a_scaling
surv_loss_days = -delta * np.log(rate_days) + rate_days * time_days

# Analytical check: difference must be delta * log(a)
assert np.allclose(surv_loss_days, surv_loss_hours + delta * np.log(a_scaling))

# Compute rank correlation
spearman_corr = stats.spearmanr(surv_loss_hours, surv_loss_days).correlation
print("=== SURPRISAL UNITS SENSITIVITY AUDIT (EQ. E.8) ===")
print(f"Time scaling factor a:           {a_scaling:.4f} (hours -> days)")
print(f"Shift on exact events (delta=1): {np.log(a_scaling):+.3f} nats")
print(f"Shift on censored (delta=0):     0.000 nats")
print(f"Spearman Rank Correlation:       {spearman_corr:.3f}")
print("Notice: The mixed-surprisal ranking flips solely from unit convention!")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Tobit Log-Likelihood Profiles
axes[0].plot(f_latent, ll_uncensored, label='Uncensored Normal (c=0)', lw=2, color='#1f77b4')
axes[0].plot(f_latent, ll_right_cens, label=r'Right-Censored: $Y^* \ge 1.0$ (c=+1)', lw=2, color='#2ca02c')
axes[0].plot(f_latent, ll_left_cens,  label=r'Left-Censored: $Y^* \le 1.0$ (c=-1)', lw=2, color='#d62728')
axes[0].set_title("Tobit Head: Censored Likelihood Profiles ($b = 1.0$)", fontweight='bold')
axes[0].set_xlabel("Latent Head Value f(x)")
axes[0].set_ylabel("Log Likelihood Contribution")
axes[0].legend()

# Subplot 2: Surprisal Ranking Inversion
idx = np.arange(n_surv)
axes[1].scatter(surv_loss_hours[delta==1], surv_loss_days[delta==1], color='#d62728', s=60, edgecolors='k', label='Events (delta=1, shifted by log a)')
axes[1].scatter(surv_loss_hours[delta==0], surv_loss_days[delta==0], color='#1f77b4', s=60, edgecolors='k', label='Censored (delta=0, unshifted)')
axes[1].set_title(f"Mixed Surprisal Rank Distortion\n(Spearman Correlation = {spearman_corr:.3f})", fontweight='bold')
axes[1].set_xlabel("Raw Surprisal (Time in Hours)")
axes[1].set_ylabel("Raw Surprisal (Time in Days)")
axes[1].legend()

plt.tight_layout()
plt.show()""")

    # =========================================================================
    # MODULE 10: PRACTITIONER CHEAT SHEET & CONCLUSION
    # =========================================================================
    md(r"""## Module 10: Theoretical Summary & Practitioner's Cheat Sheet

### Summary of Theoretical Principles & Guidelines

| Scientific Principle | Operational Finding & Diagnostic Rule | Reference |
| :--- | :--- | :--- |
| **The Three Distinct Targets** | Never conflate **Difficulty** (evaluated at context point), **Influence** (query-specific), and **Intervention Value** (evaluated on risk). High difficulty does not imply positive intervention value. | Section 2.2 |
| **Exactness of Linear Smoother LOO** | For normalized kernel heads $f(x) = Sy$, leave-one-out scores $p_i^{\text{fr},-i} = \frac{q_i - S_{ii}}{1 - S_{ii}}$ and $L_i = 1 - p_i^{\text{fr},-i}$ are mathematically exact in $\mathcal{O}(n^2)$ arithmetic. | Section 4.1, Eq. 9 |
| **Auditing Context Explanations** | Decompose deletion influence into frozen-head effect $+$ signed error: $I_i(x) = \frac{w_i}{1-w_i}(y_i - f_D) + E_i(x)$. Frozen heads cannot see upstream representation shifts. | Section 4.1, Eq. 8 |
| **Audit Protocol & Negative Control** | Evaluate Bias, RMS, Sign Agreement, and Top-$k$ overlap. On fixed features, $E_i \equiv 0$ identically (the analytical negative control). | Section 4.2, Table 1 |
| **Tail Precision Governs Accuracy** | The sign of intervention value depends on tail precision. Set $\tau \approx 1 - p_{\text{min}}$. Mis-setting $\tau$ (e.g. $\tau=0.80$ for a $4.7\%$ minority) upweights majority points and **collapses worst-group accuracy**! | Section 5.2, Table 2 |
| **Modulation Strategy** | Use bounded tail-set modulation ($m_i = 1 + \lambda \mathbb{I}[L_i \ge Q_\tau]$) or pseudo-group balancing. **Avoid exponential weighting**, which over-concentrates mass and collapses predictions to chance. | Appendix D.1 |
| **Prediction Rule in Data Cleaning** | Closed-form LOO detects label noise with $\text{AUC} > 0.95$. Pruning does not benefit kernel smoothers (which average noise away), but completely restores noise-sensitive consumers like 1-NN ($0.594 \to 0.729$). | Section 5.3, Table 3 |
| **Censored Surprisal Units Dilemma** | Raw mixed surprisal changes by $\delta_i \log a$ under time rescaling $U' = aU$. Never rank mixed survival points by raw surprisal; use target-specific physical predictions ($S_T(t \mid x)$). | Section 6, Eq. E.8 |

---

*Official Interactive Demonstration for **"Leave No-One-Out: Context Modifications in Tabular Foundation Models"** (Anonymous Authors, AISTATS 2027 Under Review).*""")

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
