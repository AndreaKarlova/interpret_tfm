# Spec: KernelICL on Waterbirds — context-edit audits

**Deliverable:** `notebooks/kernelicl_waterbirds.ipynb`, plus small extensions to existing code (`kernelicl/kernelicl_diagnostics.py`, `kernelicl/kernelicl_clinical.py`, `kernel_louis/audit.py`, `loo.py`, `evaluation.py`; see §7) and tests in `tests/test_kernelicl.py` (shared with the MNIST-C spec). No new module. The existing notebooks are not changed.

## 1. Goal

Test the paper's central question on a real TFM: **does the head-level explanation predict what actually happens when the context changes?**

KernelICL is the only predictor in this project with both a readable kernel vote (Eq. 4) and embeddings that move with the context. So the paper's audit, frozen effect vs recomputed effect, can be measured for every kind of context edit the paper defines (App. B.6):

| Edit | Frozen (closed form) | Paper | Section |
|---|---|---|---|
| Row deletion | Eq. 8 | §4.1–4.2, Table 1 | 3 |
| Label replacement | $w_i(v - y_i)$ | App. C.3 | 4.1 |
| Admission (observed / pseudo-label) | Eq. 11–12 | §4.3, App. C.4 | 4.2 |
| Head-only reweighting | Eq. 13 | App. D.2, Eq. D.7 | 5 |

The paper states that none of these has been measured on a TFM (§4.2, §7, App. F).

**Waterbirds-specific question:** the rare groups (landbird / water, waterbird / land) break a background shortcut. Does the frozen explanation fail more for them?

## 2. What this notebook does not repeat

Already in `kernel_icl_leave_one_out_waterbirds.ipynb`, with fixed ResNet features:
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
- fine-tuning KernelICL on Waterbirds (option (b));
- the censored-response analysis (paper §6, analytical only).

## 3. Model

| Item | Value |
|---|---|
| Checkpoint | `paper.pt`: TabICLv2 start, fine-tuned 5,000 steps on synthetic `graph_scm` tasks, Gaussian kernel, `d_k = 512`, `max_classes = 10`. Path: `CHECKPOINT_PATH` (Google Drive on Colab). |
| Fine-tuning on Waterbirds | none (option (a)) |
| Forward | `model.forward_kernel(X, y_train, gamma=γ)` from the repo's `src/tabicl` fork returns class probabilities and weights `w[b, j, i]`. Embeddings `h = W·E` (512-d) are read for context and queries. |
| Ensembling / shuffling | none: one forward pass, fixed feature and row order |
| Bandwidth γ | Per seed, the KernelICL paper's protocol, as implemented in `kernelicl_clinical.fit_explainer`: stratified 5-fold cross-validation on the context, each fold **re-embedded** with its validation rows as label-free queries; pick the sparsest scale (lowest weight perplexity) within 0.01 of the best fold accuracy. Fixed for every edit of that seed. **Not** frozen-LOO likelihood on the context embeddings: that criterion uses the label-leaking quantity behind the 0.008 collapse. |
| Preprocessing | Two layers, both fitted once per seed on the full context and **reused unchanged for every edited context** (paper §4.2: matched preprocessing): (1) our standardise + PCA; (2) TabICL's own preprocessing (`CustomStandardScaler`, `OutlierRemover`, `UniqueFeatureFilter`, fitted inside `TabICLClassifier.fit` even with `norm_method="none"`). The model's column embedding still sees the edited context; that is the representation change being measured. Note: `kernelicl_diagnostics.loo_proba(mode="refit")` refits layer (2) for every deletion, so it is **not** used for the audits (see §7). |

The checkpoint was fine-tuned on contexts of ≤ 1,024 rows and ≤ 100 features. That sets the sizes below.

## 4. Data

| Item | Value |
|---|---|
| Groups | g = 2·y + place; rare = {1, 2} (`kernel_louis.waterbirds`) |
| Features | Cached ImageNet ResNet-50 features (2048-d). Per seed: standardise, then PCA to 64, both fitted on the context. |
| Context | n = 1,000 from the train split: 500 per bird, 25 per bird with the contradicting background (ρ = 0.05). For the 3.4 sweep only: a nested n = 2,000 superset (1,000 per bird, 50 contradicting; within the 56 available), the same design as the existing notebook. |
| Queries | Fixed stratified subsample of the official test split: 250 per group = 1,000, the same for every seed |
| Admission candidates | Per seed, 20 per group = 80 train images **not** in that seed's main (n = 1,000) context. They may appear in the n = 2,000 sweep superset, which is not used for admission (it takes 50 of the 56 waterbirds on land). |
| Seeds | 10 (smoke test: 2); each redraws the context and candidates |

## 5. Notebook structure

### Section 0. Setup
- **0.1** Clone and install (same cell as the existing notebooks); on Colab also `pip install umap-learn` (figure 6).
- **0.2** Config: all constants, `SMOKE_TEST`, `CHECKPOINT_PATH`, `USE_DRIVE`.
- **0.3** Helpers (from `kernel_louis`).
- **0.4** Formula checks on small random data (assert):
  - deletion (Eq. 8), label replacement and admission (Eq. 11) frozen formulas equal brute-force re-normalisation;
  - the GP frozen-deletion formula equals brute-force GP refitting;
  - on fixed features, frozen equals recomputed for all three edits.
- **0.5** Data and splits (`kernel_louis.waterbirds`; sanity check 1).
- **0.6** ResNet features (cached).
- **0.7** Load KernelICL (`kernelicl_diagnostics.load_kernelicl`); print the checkpoint summary (`kernelicl_finetune.describe_checkpoint`). Assert:
  - determinism: two identical passes differ by < 1e-6;
  - row-order invariance: a permuted context changes predictions by < 1e-5.

  If either fails (e.g. GPU nondeterminism), do not abort: report the size, record it in Section 8, and use it as the noise floor below which $E_i$ is not interpreted.
  
  Time one forward pass and print a runtime estimate.
- **0.8** Per-seed data: context, reduction, γ, full-context pass (probabilities, weights, embeddings).

### Section 1. KernelICL baseline
- Per-group test accuracy, worst-group and balanced mean of KernelICL.
- **Agreement with two MLP-head predictors** on the same context:
  - **native TabICLv2** (the original checkpoint): the native predictor of paper App. B.5;
  - **the fine-tuned backbone's own MLP decoder** (as in `kernel_icl.ipynb`; that decoder was not trained during fine-tuning, so it may be degraded).

  For each: prediction agreement and mean |Δp|. Paper §4 / App. B.5: a replaced head must be checked against the native predictor.
- γ: the selected value vs the head default; effective number of neighbours (weight perplexity).
- Sanity check 7: above chance, and common-minus-worst gap ≥ 0.10 (warn only).

### Section 2. Difficulty: frozen vs refit leave-one-out (replication on real data)
- **Frozen** $L_i = 1 - p^{\text{fr},-i}(y_i)$ (Eq. 9) on KernelICL context embeddings, vs **refit** $L_i^{\text{rec}} = 1 - p_{\mathcal D^{-i}}(y_i \mid x_i)$ (Eq. 1): delete row i, rerun. (The prototype's numbers are −log p; same ranking.)
- **Matched refit:** preprocessing as in §3 (fitted on the full context, reused). One pass per context point, batched over B. The same passes give the Section 3 recomputed predictions.
- Report:
  - detection AUC (rare vs common) for both scores;
  - Spearman correlation between them;
  - the distribution of frozen $L_i$.
- **Expected:** frozen $L_i \approx 0$, because each context embedding already contains its own label (paper §4.1). `kernel_icl.ipynb` showed this on toy data. **What is new here:** real data, rare vs common groups, and seeds.

### Section 3. Deletion audit (paper §4.1–4.2, Eq. 5–10, Table 1)
For each context point i and query x:
- original, frozen (Eq. 5 via Eq. 8) and recomputed predictions;
- $I^{\text{head}}_i$ (Eq. 6), $I_i$ (Eq. 2) and $E_i = I_i - I^{\text{head}}_i$ (Eq. 7).

**Output:** $p(\text{waterbird} \mid x)$.

- **3.1 Audit metrics** (Table 1):
  - Bias and RMS of $E_i$ (Eq. 10), next to RMS of $I_i$;
  - sign agreement on $|I_i| > 10^{-3}$;
  - top-10 overlap per query.
  
  **Control row:** the fixed-feature kernel head on the same features, queries and code path (γ by frozen-LOO likelihood, which is honest on fixed features). Must give $E_i = 0$ (sanity check 6).

  **End-to-end row** (on the n = 1,000 audit set of 3.4): the same deletions, with TabICL's preprocessing refitted, as `kernelicl_diagnostics` refit mode does. Its difference from the matched row shows how much preprocessing drift would add (paper §4.2: an end-to-end audit is allowed but must be labelled).
- **3.2 By group:** metrics by the deleted point's group (rare vs common) and by the query's group.
- **3.3 Where the error comes from:** $I^{\text{head}}_i$ vs $I_i$ scatter; $E_i$ against $w_i(x)$ and against $L_i^{\text{rec}}$.
- **3.4 Context length** (open question, paper §7):
  - nested contexts n ∈ {250, 500, 1,000, 2,000}: each smaller context is a random subset of the next larger one, stratified by **group** (not group × bird); rare counts are rounded, and the realised ρ is reported. n = 2,000 is **beyond the fine-tuning range** (≤ 1,024 rows) and is labelled so: it checks whether the smaller main contexts hide a trend;
  - audit set = all rare points + 50 random common points;
  - RMS($E_i$) vs n.

### Section 4. Label replacement and admission audits
Sign convention: effects here are **after minus before** (paper App. C.3/C.4), unlike deletion's $I_i = f_{\mathcal D} - f_{\mathcal D^{-i}}$. In every case the error is recomputed effect minus frozen effect.

- **4.1 Label replacement** (App. C.3):
  - flip $y_i$ to the other bird for the 3.4 audit set;
  - frozen $w_i(x)(v - y_i)$ vs recomputed (label replaced, everything else fixed);
  - Bias / RMS / sign, rare vs common.
  
  Contrasts with deletion: covariates and context size stay the same, but label-conditioned embeddings can still move (App. B.4).
- **4.2 Admission** (§4.3, Eq. 11–12):
  - append each candidate z with label v;
  - frozen $a_z(x)[v - f(x)]$ vs recomputed $f_{\mathcal D^{+z,v}}(x) - f(x)$; remainder $\rho_z$;
  - two labels per candidate: **observed** (true label) and **pseudo-label** (KernelICL's prediction for z on the current context), recorded separately as the paper requires;
  - report $\rho_z$ (Bias / RMS / sign) by candidate group, and for pseudo-labels split by correct vs wrong.
  
  No admission policy is evaluated (paper §4.3).

### Section 5. Reweighting audit (Eq. 13 vs Eq. D.7)
- **Conditions:**
  - uniform resampling;
  - the pre-registered tail (τ = 0.95, λ = 8, refit $L^{\text{rec}}$ score);
  - its random-tail control.
- For each, 3 resampled contexts (n rows, probability ∝ m), recomputed. Decompose (Eq. D.7):
  $f_{\mathcal D^{(c)}} - f^{\text{fr}}_{m|\mathcal D} = [f_{\mathcal D^{(c)}} - f^{\text{fr}}_{c|\mathcal D}]_{\text{recomputation}} + [f^{\text{fr}}_{c|\mathcal D} - f^{\text{fr}}_{m|\mathcal D}]_{\text{sampling}}$.
- Report the RMS of each term over queries, by query group. Accuracy changes appear only as context for those terms.
- Exact here because KernelICL has the kernel head. The existing notebook's Table D control (d) could only approximate it.

### Section 6. GP head on KernelICL embeddings (App. A.6; option (c))
- GP regression on one-hot labels, Gaussian covariance on the 512-d KernelICL embeddings.
- **Regression, not a GP classifier:** regression has exact leave-one-out and frozen-deletion formulas. A classifier needs an approximation (Laplace / EP), which would mix approximation error into the audit (the paper's warning in §6 / Eq. E.6). Outputs are not calibrated probabilities: accuracy uses the argmax; the audit uses the true-class output.
- γ_GP and σ² chosen per context by **cross-validated held-out log-likelihood**, on the same 5 re-embedded folds used for γ (no extra passes); grid γ_GP ∈ `kernelicl_clinical.GAMMA_GRID` × σ² ∈ {0.01, 0.1, 1}. Backbone and W frozen. Not GP leave-one-out (Eq. A.20) on the context embeddings: those encode their own labels, and paper App. A.6 warns that the conditional identities then do not give genuine held-out prediction.
- **6.1 Prediction:** per-group accuracy of the GP head vs the kernel vote on the same embeddings.
- **6.2 GP deletion audit:**
  - frozen-covariance deletion $\mu^{-i}(x) = \mu(x) - (g_x^\top Q_{:,i})\,\alpha_i / Q_{ii}$ vs recomputed embeddings with the same γ_GP, σ²;
  - no new forward passes: inside the Section 2/3 deletion loop, each pass's recomputed embeddings are used immediately to refit the GP (one (n−1)×(n−1) solve) and predict the queries; embeddings are **not stored** (1,000 deletions × ~2,000 rows × 512 dims would be ~4 GB per seed);
  - metrics as in 3.1–3.2.
  
  Hyperparameters are held fixed. Refitting them would add a third term (Eq. 14 / E.6); noted, not run.

### Section 7. Sanity-check summary
1. Official split sizes.
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

### Figures

Six figures, in total. Everything else is reported as tables. Each figure gets its own cell, is shown inline and is followed by a "How to read this" note. Rare groups are shown in oranges and common groups in greys/blues, as in the existing notebooks.

| # | Section | Figure | Question it answers |
|---|---|---|---|
| 1 | 2 | **Frozen vs refit difficulty:** scatter of frozen $L_i$ against refit $L_i^{\text{rec}}$ (log axes), coloured rare vs common | Does the label leak collapse the frozen score on real data, and does the ranking survive? |
| 2 | 3.1 | **Explanation vs reality:** $I^{\text{head}}_i$ against $I_i$ with a y = x line, coloured by the deleted point's group. Second panel: the fixed-feature control, where all points lie on the line. Random subsample of (point, query) pairs, stated in the title. | How much does the frozen explanation miss? (paper Fig. 1(b) vs 1(c), measured) |
| 3 | 9 | **Where the explanation fails:** relative error RMS(E)/RMS(I) per edit type (deletion, label replacement, admission observed, admission pseudo-label, reweighting recomputation term, GP-head deletion), with two bars each: rare vs common. Mean ± SE over seeds. | Is the explanation worse for rare points, and for which kinds of edit? |
| 4 | 3.4 | **Context length:** relative error against n ∈ {250, 500, 1,000, 2,000} (2,000 marked as beyond the fine-tuning range), one line for rare and one for common deletions, ± SE | Does the frozen explanation improve or degrade as the context grows? (paper §7, open question) |
| 5 | 3.3 | **One worked example:** KernelICL embeddings of context and queries in 2-D before and after deleting one rare point, with arrows for the movement. PCA is fitted once on the full-context embeddings, and both states are projected into the same coordinates (not UMAP: its distances and re-fits would distort the arrows). Next to it: the deleted image and the 5 queries whose prediction changed most. | What does "the representation moves" look like? |
| 6 | 1 | **Embedding map:** UMAP of the **test-query** embeddings (label-free; context rows are not plotted, because their embeddings contain their own labels), two panels coloured by bird and by group. Backed by a table: for each group, at three stages (raw features, TabICL's label-free row stage, KernelICL's in-context embedding; as in `kernelicl_embeddings.py` T6), the share of each point's 10 nearest neighbours with the same bird and with the same group. | do waterbirds on land sit with other waterbirds (the model sees the bird) or with landbirds (it sees the background, the shortcut)? (paper App. D.5 / F: the neighbourhood organisation of minority examples is unmeasured) |

**Relative error** RMS(E)/RMS(I) is used in figures 3–4 because raw errors depend on effect sizes and cannot be compared across groups, edits or datasets. Per edit, I is the recomputed effect and E the error: deletion and label replacement as in Sections 3 / 4.1; admission E = $\rho_z$ and I = $f_{\mathcal D^{+z,v}} - f_{\mathcal D}$; reweighting E = the recomputation term and I = $f_{\mathcal D^{(c)}} - f_{\mathcal D}$; GP head as in 6.2.

**Additional figures:** if building or running the notebook reveals something interesting that the tables do not show well, add a figure for it. Use the same conventions (one cell, inline, "How to read this" note) and list it in Section 8 (Decisions). Do not add figures that only repeat a table.

## 6. Declared defaults

| Decision | Default | Why |
|---|---|---|
| Context / features | 1,000 rows, PCA 64; one labelled n = 2,000 point in the 3.4 sweep | inside the fine-tuning range (≤ 1,024 rows, ≤ 100 features), so audit errors are not out-of-range effects; the 2,000 point checks that the smaller contexts hide no trend |
| Fine-tuning on Waterbirds | none | no training on evaluation labels |
| γ (KernelICL) | 5-fold re-embedded cross-validation per seed (`kernelicl_clinical` protocol), then fixed | label-honest; matches the KernelICL paper; §4.2: kernel parameters fixed |
| γ (fixed-feature control row) | frozen-LOO likelihood (`kernel_louis.loo.select_gamma_loo`) | fixed features carry no labels, so frozen LOO is honest there; same rule as the existing notebooks |
| Preprocessing | ours and TabICL's fitted once per seed, reused for edited contexts; end-to-end refit only as a labelled extra row | paper §4.2 |
| Audited output | $p(\text{waterbird} \mid x)$ | signed, so Bias and sign agreement are defined; binary, so one coordinate is the full vector |
| Sign tolerance | $|I| > 10^{-3}$ | declared in advance (paper §4.2) |
| Audit sets | all points (3.1–3.3); rare + 50 common (3.4, 4.1); 80 candidates (4.2) | compute |
| GP | regression on one-hot labels; γ_GP, σ² by cross-validated held-out log-likelihood on re-embedded folds, fixed during deletions | exact frozen-deletion formulas (App. A.6, Eq. A.20); honest hyperparameter choice despite label-conditioned embeddings (App. A.6 caveat) |

## 7. Code

**Principle:** no new module. Every addition goes into the existing file that owns that concept, and nothing that already exists is rewritten.

**Reused as-is**

| From | What | Used for |
|---|---|---|
| `kernelicl/kernelicl_diagnostics.py` | `load_kernelicl`; `KernelICLPredictor`: `predict_proba_and_weights`, `E_train`, `context_gram`, `loo_proba("frozen")` | loading, full-context pass, frozen $L_i$; `loo_proba("refit")` only for the end-to-end row |
| `kernelicl/kernelicl_clinical.py` | `_embed(clf, X, return_row_repr=True)`, `_make_folds` | embeddings plus the label-free row stage (figure 6 table); folds for calibration |
| `kernelicl/kernelicl_finetune.py` | `describe_checkpoint` | 0.7 |
| `kernel_louis.heads` | `kernel_head_proba` (stable, multiplier-aware) | **every** frozen kernel-vote prediction from projected embeddings: deletion ($m_i = 0$), Eq. 13 reweighting, realised counts (Section 5). One implementation; checked against `KernelHead` in 0.4. |
| `kernel_louis.heads`, `loo` | `gp_predict`, `gp_loo_precision`, `gp_loo_score_multiclass` | GP head (Section 6) |
| `kernel_louis.loo`, `kernels` | `select_gamma_loo`, `log_gaussian_kernel` | fixed-feature control row |
| `kernel_louis.evaluation` | `group_accuracy_metrics(rare_groups=…)`, `detection_auc`, `weight_perplexity`, `mean_and_se`, `paired_differences` | all summaries |
| `kernel_louis.modulation` | `tail_multiplier`, `random_tail_multiplier`, `resample_context(return_indices=True)` | Section 5 |
| `kernel_louis.waterbirds` / `mnist_c`, `cnn` | data, splits, features | Section 0 |

**Extended or refactored** (behaviour-preserving unless stated)

| File | Change |
|---|---|
| `kernelicl/kernelicl_diagnostics.py` | `KernelICLPredictor.with_context(X, y)`: a predictor for an edited context that **reuses** this one's fitted TabICL preprocessing (matched; §3), plus a batched `edit_passes(...)` for delete / replace label / append row. `loo_proba("refit")` stays as the end-to-end variant. |
| `kernelicl/kernelicl_clinical.py` | Factor the fold loop out of `fit_explainer` into `fold_embeddings(...)` (one re-embedded pass per fold) and `calibrate_scale(...)` (accuracy, sparsest within tolerance). `fit_explainer` calls them, unchanged in behaviour. The GP hyperparameter search reuses `fold_embeddings`. |
| `kernel_louis/audit.py` | Generalise to multiclass and to measured recomputed predictions: vectorised frozen effects for deletion (Eq. 8), label replacement and admission (Eq. 11) on a chosen output coordinate. **Fix `compute_audit_metrics` to paper App. C.3:** sign agreement only where the *true* effect exceeds the tolerance (now: either effect); top-k overlap *per query over context points* (now: over queries). Existing tests and callers updated. |
| `kernel_louis/loo.py` | `gp_frozen_deletion(Q, alpha, g_x, i)` (next to `gp_loo_precision`); `select_gp_hyperparameters_cv(...)` (held-out log-likelihood on fold embeddings). |
| `kernel_louis/evaluation.py` | Move `purity` here from `kernelicl_embeddings.py` (a script that cannot be imported); the script imports it back. |

**Not reused, and why**
- `kernel_louis.embeddings.extract_symmetric_embeddings`: hooks *native* TabICL before its final LayerNorm. Superseded by the model's own `embed()` via `KernelICLPredictor`.
- `kernel_louis.models.KernelICLClassifier`: a fixed-feature, binary-only kernel, not KernelICL.
- `ClinicalExplainer` views (`influence`, `triage`, `audit_labels`, `equity`): clinical summaries, not paper quantities.
- Training code in `kernelicl_finetune.py` (option (a): no fine-tuning), and the clinical experiment scripts (`kernelicl_shift`, `feature_corruption`, `cross-programme`, `analysis`).

**Tests:**
- extend `tests/test_loo_exactness.py` (audit metrics, multiclass frozen formulas, GP frozen deletion);
- add `tests/test_kernelicl.py`:
  - `with_context` keeps preprocessing unchanged;
  - batched edits equal one-at-a-time passes;
  - `kernel_head_proba` equals `KernelHead`;
  - `calibrate_scale` refactor equals the old `fit_explainer` choice;
  - determinism and row-order invariance on a small randomly initialised TabICL (no checkpoint needed).

## 8. Compute (estimate; measured in 0.7)

| Step | Passes per seed |
|---|---|
| Sections 2–3 (refit LOO + deletion audit) | 1,000 |
| 3.4 context length (incl. n = 2,000) | ~300 |
| 4.1 label replacement | ~100 |
| 4.2 admission | 160 |
| 3.1 end-to-end row (preprocessing refitted) | ~100 |
| 0.8 γ calibration (5 re-embedded folds) + full-context pass | 6 |
| 1 agreement (native TabICLv2, MLP decoder) | 2 |
| 5 reweighting | 9 |
| 6 GP | 0 passes; one (n−1)×(n−1) GP solve per deletion inside the Section 2/3 loop |

If too slow: 5 seeds for Sections 3.4–5 (declared).

## 9. Open question for the supervisor
1. Confirm option (a) for the model (no fine-tuning on this dataset) and option (c) for the GP head: does "fine-tune the head with GP" mean fitting only γ_GP and σ² per context, not the projection W?

Resolved as declared defaults (§6): staying within the fine-tuning range plus one n = 2,000 check; GP regression rather than a GP classifier; $p(\text{waterbird} \mid x)$ as the audited output.
