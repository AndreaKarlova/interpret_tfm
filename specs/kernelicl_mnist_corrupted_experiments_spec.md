# Spec: KernelICL on corrupted MNIST — context-edit audits

**Deliverable:** `notebooks/kernelicl_mnist_corrupted.ipynb`, plus small extensions to existing code (`kernelicl/kernelicl_diagnostics.py`, `kernelicl/kernelicl_clinical.py`, `kernel_louis/audit.py`, `loo.py`, `evaluation.py`; see §7) and tests in `tests/test_kernelicl.py` (shared with the Waterbirds spec). No new module. The existing notebooks are not changed.

## 1. Goal

Test the paper's central question on a real TFM: **does the head-level explanation predict what actually happens when the context changes?**

KernelICL is the only predictor in this project with both a readable kernel vote (Eq. 4) and embeddings that move with the context. So the paper's audit, frozen effect vs recomputed effect, can be measured for every kind of context edit the paper defines (App. B.6):

| Edit | Frozen (closed form) | Paper | Section |
|---|---|---|---|
| Row deletion | Eq. 8 (coordinatewise) | §4.1–4.2, Table 1 | 3 |
| Label replacement | $w_i(\mathbf e_v - \mathbf e_{y_i})$ | App. C.3 | 4.1 |
| Admission (observed / pseudo-label) | Eq. 11–12 | §4.3, App. C.4 | 4.2 |
| Head-only reweighting | Eq. 13 | App. D.2, Eq. D.7 | 5 |

The paper states that none of these has been measured on a TFM (§4.2, §7, App. F).

**MNIST-C-specific question:** the rare corrupted digits are correctly labelled and fail only through under-representation. Does the frozen explanation fail more when a rare point is edited? Does this differ from Waterbirds, where the rare groups break a shortcut?

## 2. What this notebook does not repeat

Already in `kernel_icl_leave_one_out_mnist_corrupted.ipynb`, with fixed CNN features:
- the pilot that chooses the corruptions;
- Table A: reweighting value;
- Table B: tail composition;
- the ρ sweep;
- Table C: label noise;
- Table D: native-TabICL resampling;
- the fixed-feature and GP difficulty scores.

This notebook reports accuracy only where it is needed to interpret an audit. It does not ask whether an intervention helps worst-group accuracy.

Builds on two earlier prototype notebooks:
- `kernel_icl.ipynb`: KernelICL from `paper.pt` on a 60-point toy dataset. It compares the kernel head with the MLP decoder and frozen with refit leave-one-out (as −log p: frozen median 0.008, refit median 0.19; Spearman 0.93). Sections 1–2 scale this to real data, per group, over seeds.
- `visualise_tabicl_embeddings_classifier.ipynb`: a shrinking-context animation with a kernel on *native* TabICL embeddings. Section 3 makes that idea quantitative with KernelICL's own head, a fixed γ, and single deletions from the full context.

Out of scope everywhere:
- fine-tuning KernelICL on MNIST-C (option (b));
- the censored-response analysis (paper §6, analytical only).

## 3. Model

| Item | Value |
|---|---|
| Checkpoint | `paper.pt`: TabICLv2 start, fine-tuned 5,000 steps on synthetic `graph_scm` tasks, Gaussian kernel, `d_k = 512`, `max_classes = 10` (exactly MNIST's 10). Path: `CHECKPOINT_PATH` (Google Drive on Colab). |
| Fine-tuning on MNIST-C | none (option (a)) |
| Forward | `model.forward_kernel(X, y_train, gamma=γ)` from the repo's `src/tabicl` fork returns class probabilities and weights `w[b, j, i]`. Embeddings `h = W·E` (512-d) are read for context and queries. |
| Ensembling / shuffling | none: one forward pass, fixed feature and row order |
| Bandwidth γ | Per seed, the KernelICL paper's protocol, as implemented in `kernelicl_clinical.fit_explainer`: stratified 5-fold cross-validation on the context, each fold **re-embedded** with its validation rows as label-free queries; pick the sparsest scale (lowest weight perplexity) within 0.01 of the best fold accuracy. Fixed for every edit of that seed. **Not** frozen-LOO likelihood on the context embeddings: that criterion uses the label-leaking quantity behind the 0.008 collapse. |
| Preprocessing | Two layers, both fitted once per seed on the full context and **reused unchanged for every edited context** (paper §4.2: matched preprocessing): (1) our standardise + PCA; (2) TabICL's own preprocessing (`CustomStandardScaler`, `OutlierRemover`, `UniqueFeatureFilter`, fitted inside `TabICLClassifier.fit` even with `norm_method="none"`). The model's column embedding still sees the edited context; that is the representation change being measured. Note: `kernelicl_diagnostics.loo_proba(mode="refit")` refits layer (2) for every deletion, so it is **not** used for the audits (see §7). |

The checkpoint was fine-tuned on contexts of ≤ 1,024 rows and ≤ 100 features. That sets the sizes below.

## 4. Data

| Item | Value |
|---|---|
| Rare corruptions | Fixed to the existing run's pilot choice: `translate`, `stripe`, `canny_edges`. No pilot section here. |
| Splits | As in the existing notebook (`kernel_louis.mnist_c.make_index_splits`, `SPLIT_SEED = 12345`) |
| Features | The same small CNN, trained on clean CNN-pool images (`kernel_louis.cnn`), 128-d. Per seed: standardise, then PCA to 64, both fitted on the context. |
| Context | n = 1,000, 100 per digit: 94 clean + 2 per rare corruption (ρ = 0.06; 5% does not divide over 3 corruptions). For the 3.4 sweep only: a nested n = 2,000 superset (200 per digit: 188 clean + 4 per rare corruption). |
| Queries | Paired: 25 test digits per class in all 4 versions = 1,000 (250 per group), redrawn per seed |
| Admission candidates | Per seed, 20 per group = 80 context-pool images **not** in that seed's context |
| Seeds | 10 (smoke test: 2); each redraws the context, queries and candidates |

## 5. Notebook structure

### Section 0. Setup
- **0.1** Clone and install (same cell as the existing notebooks); on Colab also `pip install umap-learn` (figure 6).
- **0.2** Config: all constants, `SMOKE_TEST`, `CHECKPOINT_PATH`, `USE_DRIVE`, `RARE_CORRUPTIONS`.
- **0.3** Helpers (from `kernel_louis`).
- **0.4** Formula checks on small random multiclass data (assert):
  - deletion (Eq. 8), label replacement and admission (Eq. 11) frozen formulas equal brute-force re-normalisation;
  - the GP frozen-deletion formula equals brute-force GP refitting;
  - on fixed features, frozen equals recomputed for all three edits.
- **0.5** Data and splits (`kernel_louis.mnist_c`; sanity check 1: identical labels in every folder).
- **0.6** CNN features (trained on the CNN pool, as before).
- **0.7** Load KernelICL (`kernelicl_diagnostics.load_kernelicl`); print the checkpoint summary (`kernelicl_finetune.describe_checkpoint`). Assert:
  - determinism: two identical passes differ by < 1e-6;
  - row-order invariance: a permuted context changes predictions by < 1e-5.

  If either fails (e.g. GPU nondeterminism), do not abort: report the size, record it in Section 8, and use it as the noise floor below which $E_i$ is not interpreted.
  
  Time one forward pass and print a runtime estimate.
- **0.8** Per-seed data: context, queries, reduction, γ, full-context pass (probabilities, weights, embeddings).

### Section 1. KernelICL baseline
- Per-group test accuracy, worst-group and balanced mean of KernelICL.
- **Agreement with two MLP-head predictors** on the same context:
  - **native TabICLv2** (the original checkpoint): the native predictor of paper App. B.5;
  - **the fine-tuned backbone's own MLP decoder** (as in `kernel_icl.ipynb`; that decoder was not trained during fine-tuning, so it may be degraded).

  For each: prediction agreement and mean total-variation distance between probability vectors. Paper §4 / App. B.5: a replaced head must be checked against the native predictor.
- γ: the selected value vs the head default; effective number of neighbours (weight perplexity).
- Sanity check 7: above chance (0.1), and clean-minus-worst gap ≥ 0.10 (warn only).

### Section 2. Difficulty: frozen vs refit leave-one-out (replication on real data)
- **Frozen** $L_i = 1 - p^{\text{fr},-i}(y_i)$ (Eq. 9) on KernelICL context embeddings, vs **refit** $L_i^{\text{rec}} = 1 - p_{\mathcal D^{-i}}(y_i \mid x_i)$ (Eq. 1): delete row i, rerun. (The prototype's numbers are −log p; same ranking.)
- **Matched refit:** preprocessing as in §3 (fitted on the full context, reused). One pass per context point, batched over B. The same passes give the Section 3 recomputed predictions.
- Report:
  - detection AUC (rare vs clean) for both scores;
  - Spearman correlation between them;
  - the distribution of frozen $L_i$.
- **Expected:** frozen $L_i \approx 0$, because each context embedding already contains its own label (paper §4.1). `kernel_icl.ipynb` showed this on toy data. **What is new here:** real data, rare vs clean groups, and seeds.

### Section 3. Deletion audit (paper §4.1–4.2, Eq. 5–10, Table 1)
For each context point i and query x:
- original, frozen (Eq. 5 via Eq. 8, coordinatewise) and recomputed probability vectors;
- $I^{\text{head}}_i$ (Eq. 6), $I_i$ (Eq. 2) and $E_i = I_i - I^{\text{head}}_i$ (Eq. 7).

**Output** (multiclass, declared as paper §4.2 requires):
- **main:** the probability of the query's true class;
- **secondary:** the total-variation size ½‖·‖₁ of the change vectors.

- **3.1 Audit metrics** (Table 1):
  - Bias and RMS of $E_i$ (Eq. 10), next to RMS of $I_i$;
  - sign agreement on $|I_i| > 10^{-3}$;
  - top-10 overlap per query.
  
  **Control row:** the fixed-feature kernel head on the same features, queries and code path (γ by frozen-LOO likelihood, which is honest on fixed features). Must give $E_i = 0$ (sanity check 6).

  **End-to-end row** (on the n = 1,000 audit set of 3.4): the same deletions, with TabICL's preprocessing refitted, as `kernelicl_diagnostics` refit mode does. Its difference from the matched row shows how much preprocessing drift would add (paper §4.2: an end-to-end audit is allowed but must be labelled).
- **3.2 By group:** metrics by the deleted point's group (rare vs clean) and by the query's group. Same-digit and same-corruption pairs are reported separately.
- **3.3 Where the error comes from:** $I^{\text{head}}_i$ vs $I_i$ scatter; $E_i$ against $w_i(x)$ and against $L_i^{\text{rec}}$.
- **3.4 Context length** (open question, paper §7):
  - nested contexts n ∈ {250, 500, 1,000, 2,000}: each smaller context is a random subset of the next larger one, stratified by **group** (not group × digit); rare counts are rounded, and the realised ρ is reported. n = 2,000 is **beyond the fine-tuning range** (≤ 1,024 rows) and is labelled so: it checks whether the smaller main contexts hide a trend;
  - audit set = all rare points + 60 random clean points;
  - RMS($E_i$) vs n.

### Section 4. Label replacement and admission audits
Sign convention: effects here are **after minus before** (paper App. C.3/C.4), unlike deletion's $I_i = f_{\mathcal D} - f_{\mathcal D^{-i}}$. In every case the error is recomputed effect minus frozen effect.

- **4.1 Label replacement** (App. C.3):
  - replace $y_i$ by a random other class (seeded) for the 3.4 audit set;
  - frozen $w_i(x)(\mathbf e_v - \mathbf e_{y_i})$ vs recomputed (label replaced, everything else fixed);
  - Bias / RMS / sign, rare vs clean.
  
  Contrasts with deletion: covariates and context size stay the same, but label-conditioned embeddings can still move (App. B.4).
- **4.2 Admission** (§4.3, Eq. 11–12):
  - append each candidate z with label v;
  - frozen $a_z(x)[\mathbf e_v - \mathbf p(x)]$ vs recomputed $\mathbf p_{\mathcal D^{+z,v}}(x) - \mathbf p(x)$; remainder $\rho_z$;
  - two labels per candidate: **observed** (true label) and **pseudo-label** (KernelICL's prediction for z on the current context), recorded separately as the paper requires;
  - report $\rho_z$ (Bias / RMS / sign) by candidate group, and for pseudo-labels split by correct vs wrong.
  
  No admission policy is evaluated (paper §4.3).

### Section 5. Reweighting audit (Eq. 13 vs Eq. D.7)
- **Conditions:**
  - uniform resampling;
  - the pre-registered tail (τ = 0.95, λ = 8, refit $L^{\text{rec}}$ score);
  - its random-tail control.
- For each, 3 resampled contexts (n rows, probability ∝ m), recomputed. Decompose (Eq. D.7):
  $f_{\mathcal D^{(c)}} - f^{\text{fr}}_{m|\mathcal D} = [f_{\mathcal D^{(c)}} - f^{\text{fr}}_{c|\mathcal D}]_{\text{recomputation}} + [f^{\text{fr}}_{c|\mathcal D} - f^{\text{fr}}_{m|\mathcal D}]_{\text{sampling}}$,
  on the true-class probability.
- Report the RMS of each term over queries, by query group. Accuracy changes appear only as context for those terms.
- Exact here because KernelICL has the kernel head. The existing notebook's Table D control (d) could only approximate it.

### Section 6. GP head on KernelICL embeddings (App. A.6; option (c))
- GP regression on one-hot labels (10 outputs), Gaussian covariance on the 512-d KernelICL embeddings.
- **Regression, not a GP classifier:** regression has exact leave-one-out and frozen-deletion formulas. A classifier needs an approximation (Laplace / EP), which would mix approximation error into the audit (the paper's warning in §6 / Eq. E.6). Outputs are not calibrated probabilities: accuracy uses the argmax; the audit uses the true-class output.
- γ_GP and σ² chosen per context by **cross-validated held-out log-likelihood**, on the same 5 re-embedded folds used for γ (no extra passes); grid γ_GP ∈ `kernelicl_clinical.GAMMA_GRID` × σ² ∈ {0.01, 0.1, 1}. Backbone and W frozen. Not GP leave-one-out (Eq. A.20) on the context embeddings: those encode their own labels, and paper App. A.6 warns that the conditional identities then do not give genuine held-out prediction.
- **6.1 Prediction:** per-group accuracy of the GP head (argmax of the posterior mean) vs the kernel vote on the same embeddings.
- **6.2 GP deletion audit:**
  - frozen-covariance deletion per class, $\mu^{-i}(x) = \mu(x) - (g_x^\top Q_{:,i})\,\alpha_i / Q_{ii}$, vs recomputed embeddings with the same γ_GP, σ²;
  - no new forward passes: inside the Section 2/3 deletion loop, each pass's recomputed embeddings are used immediately to refit the GP (one (n−1)×(n−1) solve) and predict the queries; embeddings are **not stored** (1,000 deletions × ~2,000 rows × 512 dims would be ~4 GB per seed);
  - metrics as in 3.1–3.2.
  
  Hyperparameters are held fixed. Refitting them would add a third term (Eq. 14 / E.6); noted, not run.

### Section 7. Sanity-check summary
1. Labels identical across folders.
2. Exact context composition.
3. Checkpoint limits: ≤ 10 classes, ≤ 100 features, and ≤ 1,024 rows for every context except the labelled n = 2,000 sweep point.
4. Formula checks (0.4).
5. Determinism and row-order invariance (or the measured noise floor, see 0.7).
6. Fixed-feature control row: $E_i = 0$.
7. Baseline gap (warn).

### Section 8. Decisions
Every default in this spec's §6 (Declared defaults), any added figures, and every deviation found while running.

### Section 9. Results summary
Printed from the result tables. No hard-coded numbers.

### Section 10. Interpretation (template)
Guardrails:
- A small error supports the explanation only on the audited contexts and edits.
- A large error does not mean the recomputed predictor is worse.
- Frozen $L_i$ is not label-honest.
- Pseudo-label results are not an admission policy.
- Comparisons with Waterbirds keep the different failure mechanism in mind.

### Figures

Six figures, in total. Everything else is reported as tables. Each figure gets its own cell, is shown inline and is followed by a "How to read this" note. Rare groups are shown in oranges and clean groups in greys/blues, as in the existing notebooks.

| # | Section | Figure | Question it answers |
|---|---|---|---|
| 1 | 2 | **Frozen vs refit difficulty:** scatter of frozen $L_i$ against refit $L_i^{\text{rec}}$ (log axes), coloured rare vs clean | Does the label leak collapse the frozen score on real data, and does the ranking survive? |
| 2 | 3.1 | **Explanation vs reality:** $I^{\text{head}}_i$ against $I_i$ with a y = x line, coloured by the deleted point's group. Second panel: the fixed-feature control, where all points lie on the line. Random subsample of (point, query) pairs, stated in the title. | How much does the frozen explanation miss? (paper Fig. 1(b) vs 1(c), measured) |
| 3 | 9 | **Where the explanation fails:** relative error RMS(E)/RMS(I) per edit type (deletion, label replacement, admission observed, admission pseudo-label, reweighting recomputation term, GP-head deletion), with two bars each: rare vs clean. Mean ± SE over seeds. | Is the explanation worse for rare points, and for which kinds of edit? |
| 4 | 3.4 | **Context length:** relative error against n ∈ {250, 500, 1,000, 2,000} (2,000 marked as beyond the fine-tuning range), one line for rare and one for clean deletions, ± SE | Does the frozen explanation improve or degrade as the context grows? (paper §7, open question) |
| 5 | 3.3 | **One worked example:** KernelICL embeddings of context and queries in 2-D before and after deleting one rare point, with arrows for the movement. PCA is fitted once on the full-context embeddings, and both states are projected into the same coordinates (not UMAP: its distances and re-fits would distort the arrows). Next to it: the deleted image and the 5 queries whose prediction changed most. | What does "the representation moves" look like? |
| 6 | 1 | **Embedding map:** UMAP of the **test-query** embeddings (label-free; context rows are not plotted, because their embeddings contain their own labels), two panels coloured by digit and by group. Backed by a table: for each group, at three stages (raw features, TabICL's label-free row stage, KernelICL's in-context embedding; as in `kernelicl_embeddings.py` T6), the share of each point's 10 nearest neighbours with the same digit and with the same group. | do `translate` / `stripe` / `canny_edges` digits cluster by **digit** or by **corruption**? (paper App. D.5 / F: the neighbourhood organisation of minority examples is unmeasured) |

**Relative error** RMS(E)/RMS(I) is used in figures 3–4 because raw errors depend on effect sizes and cannot be compared across groups, edits or datasets. Per edit, I is the recomputed effect and E the error: deletion and label replacement as in Sections 3 / 4.1; admission E = $\rho_z$ and I = $f_{\mathcal D^{+z,v}} - f_{\mathcal D}$; reweighting E = the recomputation term and I = $f_{\mathcal D^{(c)}} - f_{\mathcal D}$; GP head as in 6.2.

**Additional figures:** if building or running the notebook reveals something interesting that the tables do not show well, add a figure for it. Use the same conventions (one cell, inline, "How to read this" note) and list it in Section 8 (Decisions). Do not add figures that only repeat a table.

## 6. Declared defaults

| Decision | Default | Why |
|---|---|---|
| Context / features | 1,000 rows (ρ = 0.06), PCA 64; one labelled n = 2,000 point in the 3.4 sweep | inside the fine-tuning range (≤ 1,024 rows, ≤ 100 features), so audit errors are not out-of-range effects; the 2,000 point checks that the smaller contexts hide no trend |
| Rare corruptions | `translate`, `stripe`, `canny_edges` | comparability with the existing run |
| Fine-tuning on MNIST-C | none | no training on evaluation labels |
| γ (KernelICL) | 5-fold re-embedded cross-validation per seed (`kernelicl_clinical` protocol), then fixed | label-honest; matches the KernelICL paper; §4.2: kernel parameters fixed |
| γ (fixed-feature control row) | frozen-LOO likelihood (`kernel_louis.loo.select_gamma_loo`) | fixed features carry no labels, so frozen LOO is honest there; same rule as the existing notebooks |
| Preprocessing | ours and TabICL's fitted once per seed, reused for edited contexts; end-to-end refit only as a labelled extra row | paper §4.2 |
| Audited output | true-class probability; TV size secondary | the true-class probability is signed, so Bias and sign agreement are defined; TV is always ≥ 0 and only measures size (paper §4.2: multiclass audits must declare one) |
| Sign tolerance | $|I| > 10^{-3}$ | declared in advance (paper §4.2) |
| Label replacement | random other class (seeded) | matches symmetric noise in the existing Table C |
| Audit sets | all points (3.1–3.3); rare + 60 clean (3.4, 4.1); 80 candidates (4.2) | compute |
| GP | regression on one-hot labels; γ_GP, σ² by cross-validated held-out log-likelihood on re-embedded folds, fixed during deletions | exact frozen-deletion formulas (App. A.6, Eq. A.20); honest hyperparameter choice despite label-conditioned embeddings (App. A.6 caveat) |

## 7. Code
Identical to the Waterbirds spec, §7: the same reuse table, the same extensions to `kernelicl_diagnostics`, `kernelicl_clinical`, `kernel_louis.audit`, `loo` and `evaluation`, and no new module. This notebook additionally reuses `kernel_louis.mnist_c` and `cnn`. The multiclass parts of the `audit.py` extension (true-class coordinate, TV size) are needed here.

## 8. Compute (estimate; measured in 0.7)

| Step | Passes per seed |
|---|---|
| Sections 2–3 (refit LOO + deletion audit) | 1,000 |
| 3.4 context length (incl. n = 2,000) | ~360 |
| 4.1 label replacement | ~120 |
| 4.2 admission | 160 |
| 3.1 end-to-end row (preprocessing refitted) | ~120 |
| 0.8 γ calibration (5 re-embedded folds) + full-context pass | 6 |
| 1 agreement (native TabICLv2, MLP decoder) | 2 |
| 5 reweighting | 9 |
| 6 GP | 0 passes; one (n−1)×(n−1) GP solve per deletion inside the Section 2/3 loop |

If too slow: 5 seeds for Sections 3.4–5 (declared).

## 9. Open question for the supervisor
1. Confirm option (a) for the model (no fine-tuning on this dataset) and option (c) for the GP head: does "fine-tune the head with GP" mean fitting only γ_GP and σ² per context, not the projection W?

Resolved as declared defaults (§6): staying within the fine-tuning range plus one n = 2,000 check; GP regression rather than a GP classifier; true-class probability (TV secondary) as the audited output.
