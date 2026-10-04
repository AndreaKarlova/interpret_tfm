import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, brier_score_loss, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, precision_recall_curve, auc,
)

# Reporting order, used by both the metric function and the table
METRICS = ['Balanced Accuracy', 'PPV', 'NPV', 'Sensitivity', 'Specificity',
           'F1', 'MCC', 'AUROC', 'AUPRC', 'Brier']


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCHEMA ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def align_external(X_ext, reference, verbose=True):
    """
    Reshape the external frame to the schema the preprocessor was fitted on.

    Absent columns are added as NaN and handled downstream by the imputer;
    extra columns are dropped. Dtypes are restored to match `reference`,
    because filling an absent categorical column with NaN would make it
    float64 and break the fitted OneHotEncoder.
    """
    X       = X_ext.copy()
    columns = list(reference.columns)

    missing = [c for c in columns if c not in X.columns]
    extra   = [c for c in X.columns if c not in columns]

    for c in missing:
        X[c] = np.nan
    X = X[columns]

    recast, failed = [], []
    for c in columns:
        want = reference[c].dtype
        if X[c].dtype == want:
            continue
        try:
            X[c] = (X[c].astype(object).where(X[c].notna(), np.nan)
                    if want == object else X[c].astype(want))
            recast.append(c)
        except (TypeError, ValueError):
            failed.append(c)

    if verbose:
        print('\n── Schema alignment ──')
        print(f'  Rows            : {len(X):,}')
        print(f'  Columns         : {len(columns)}')
        if missing: print(f'  Added as NaN    : {missing}')
        if extra:   print(f'  Dropped         : {extra}')
        if recast:  print(f'  Recast          : {recast}')
        if failed:  print(f'  ⚠ Cast failed   : {failed}')
        if not (missing or extra or recast or failed):
            print('  Schema matches exactly ✅')

    return X


def report_unseen_categories(X_int, X_ext, categorical_cols, top_n=6):
    """
    Categories present externally but absent from training.

    These one-hot encode to all zeros, so the model has no information from
    that column for the affected rows.
    """
    print('── Unseen categories ──')
    found = False
    for c in categorical_cols:
        if c not in X_int.columns or c not in X_ext.columns:
            continue
        unseen = set(X_ext[c].dropna().unique()) - set(X_int[c].dropna().unique())
        if not unseen:
            continue
        found = True
        share = X_ext[c].isin(unseen).mean()
        shown = sorted(unseen)[:top_n]
        more  = f' … +{len(unseen) - top_n}' if len(unseen) > top_n else ''
        print(f'  {c:<24}{len(unseen):>3} unseen  {share:>6.1%} of rows  '
              f'{shown}{more}')
    if not found:
        print('  None ✅')


# ══════════════════════════════════════════════════════════════════════════════
# 2. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_prob, threshold):
    """All reported metrics at one operating point."""
    y_true = np.asarray(y_true)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec, rec, _   = precision_recall_curve(y_true, y_prob)

    return {
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'PPV'              : precision_score(y_true, y_pred, zero_division=0),
        'NPV'              : tn / (tn + fn) if (tn + fn) else 0.0,
        'Sensitivity'      : recall_score(y_true, y_pred, zero_division=0),
        'Specificity'      : tn / (tn + fp) if (tn + fp) else 0.0,
        'F1'               : f1_score(y_true, y_pred, zero_division=0),
        'MCC'              : matthews_corrcoef(y_true, y_pred),
        'AUROC'            : roc_auc_score(y_true, y_prob),
        'AUPRC'            : auc(rec, prec),
        'Brier'            : brier_score_loss(y_true, y_prob),
    }


def youden_threshold(y_true, y_prob):
    """Threshold maximising sensitivity + specificity − 1."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def predict(spec, X_raw, X_proc):
    """Route each model to the input format it was fitted on."""
    X = X_raw if spec['raw'] else X_proc
    return spec['model'].predict_proba(X)[:, 1]


# ══════════════════════════════════════════════════════════════════════════════
# 3. RECALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def recalibrate(prob_fit, y_fit, prob_apply, method='platt'):
    """
    Map raw scores onto calibrated probabilities.

    platt    — sigmoid, two parameters, stable on small calibration sets
    isotonic — arbitrary monotone step function, more flexible but needs
               more data to avoid overfitting
    """
    if method == 'platt':
        model = LogisticRegression(C=1e6, solver='lbfgs')
        model.fit(prob_fit.reshape(-1, 1), y_fit)
        return model.predict_proba(prob_apply.reshape(-1, 1))[:, 1]

    if method == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(prob_fit, y_fit)
        return model.predict(prob_apply)

    raise ValueError(f"method must be 'platt' or 'isotonic', got {method!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-PROGRAMME EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def cross_programme_evaluation(X_ext_raw, y_ext, X_int_raw, X_int_proc, y_int,
                               preprocessor, model_registry,
                               methods=('platt', 'isotonic'),
                               calib_frac=0.5, random_state=42):
    """
    Evaluate internally trained models on an external cohort.

    Rows per model:

      Internal, as-is           reference performance on the internal test set
      External, as-is (full)    deployed model and threshold, unchanged, on the
                                whole external cohort — the validation result
      External, as-is (eval)    the same, restricted to the evaluation half so
                                it is directly comparable to the rows below
      External, recalibrated    one row per calibration method, each fitted on
                                the fit half and scored on the evaluation half
      External, re-thresholded  threshold re-tuned on the fit half, raw scores

    Every fitted condition finds its threshold on the fit half and is scored
    on the evaluation half, so nothing is evaluated on data it was fitted to.
    """
    y_ext      = np.asarray(y_ext)
    y_int      = np.asarray(y_int)
    X_ext      = align_external(X_ext_raw, X_int_raw)

    needs_proc = any(not s['raw'] for s in model_registry.values())
    X_ext_proc = preprocessor.transform(X_ext) if needs_proc else None

    n_cols   = len(X_int_raw.columns)
    all_nan  = [c for c in X_ext.columns if X_ext[c].isna().all()]
    coverage = 1 - len(all_nan) / n_cols
    print(f'  Schema coverage : {coverage:.0%} '
          f'({n_cols - len(all_nan)} of {n_cols} columns populated)')

    fit_idx, eval_idx = train_test_split(
        np.arange(len(y_ext)), test_size=1 - calib_frac,
        stratify=y_ext, random_state=random_state)

    print('\n── Cohorts ──')
    print(f'  Internal   : n={len(y_int):,}   '
          f'non-adherence={y_int.mean():.1%}')
    print(f'  External   : n={len(y_ext):,}   '
          f'non-adherence={y_ext.mean():.1%}')
    print(f'  External split: {len(fit_idx):,} to fit, '
          f'{len(eval_idx):,} to score')

    rows = []
    for name, spec in model_registry.items():
        thr      = spec['threshold']
        prob_int = predict(spec, X_int_raw, X_int_proc)
        prob_ext = predict(spec, X_ext,     X_ext_proc)

        conditions = [
            ('Internal', 'as-is',             len(y_int),    thr,
             y_int,           prob_int),
            ('External', 'as-is (full)',      len(y_ext),    thr,
             y_ext,           prob_ext),
            ('External', 'as-is (eval half)', len(eval_idx), thr,
             y_ext[eval_idx], prob_ext[eval_idx]),
        ]

        # ── One row per calibration method ────────────────────────────────────
        for method in methods:
            # Fitted on the fit half; applied there to tune a threshold,
            # and to the eval half to produce the scores that get reported.
            cal_fit  = recalibrate(prob_ext[fit_idx], y_ext[fit_idx],
                                   prob_ext[fit_idx], method)
            thr_cal  = youden_threshold(y_ext[fit_idx], cal_fit)
            cal_eval = recalibrate(prob_ext[fit_idx], y_ext[fit_idx],
                                   prob_ext[eval_idx], method)
            conditions.append(
                ('External', f'recalibrated ({method})', len(eval_idx),
                 thr_cal, y_ext[eval_idx], cal_eval))

        # ── Threshold only, no calibration ────────────────────────────────────
        thr_new = youden_threshold(y_ext[fit_idx], prob_ext[fit_idx])
        conditions.append(
            ('External', 're-thresholded', len(eval_idx), thr_new,
             y_ext[eval_idx], prob_ext[eval_idx]))

        for cohort, condition, n, t, y, p in conditions:
            rows.append({'Model': name, 'Cohort': cohort,
                         'Condition': condition, 'n': n,
                         'Threshold': round(t, 4),
                         **compute_metrics(y, p, t)})

    return pd.DataFrame(rows)
    

# ══════════════════════════════════════════════════════════════════════════════
# 5. TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_cross_programme_table(df, metrics=METRICS, label_w=46, col_w=13):
    """
    Console table of cross-programme results.

    label_w must clear the longest condition label — currently
    'External, recalibrated (isotonic) (n=1,956)' at 45 characters.
    """
    metrics = [m for m in metrics if m in df.columns]
    width   = label_w + len(metrics) * col_w + 2

    print('\n' + '═' * width)
    print('  Cross-Programme Generalisation')
    print('═' * width)
    print(f'  {"Model / condition":<{label_w}}'
          + ''.join(f'{m:>{col_w}}' for m in metrics))
    print('  ' + '─' * (width - 2))

    for model in df['Model'].unique():
        print(f'\n  {model}')
        for _, r in df[df['Model'] == model].iterrows():
            label = f'  {r["Cohort"]}, {r["Condition"]} (n={int(r["n"]):,})'
            print(f'  {label:<{label_w}}'
                  + ''.join(f'{r[m]:>{col_w}.3f}' for m in metrics))

    print('\n' + '═' * width)
    print('  External, as-is (full) is the external validation result: the')
    print('  deployed model and threshold, unchanged, on the whole cohort.')
    print('  The remaining external rows share the same evaluation half, so')
    print('  differences between them reflect the condition rather than the')
    print('  sample. Recalibration and re-thresholding are fitted on the other')
    print('  half, and are diagnostic rather than validation.')
    print('═' * width + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# USAGE
# ══════════════════════════════════════════════════════════════════════════════

report_unseen_categories(X_test_raw, X_ext_raw, list(categorical_cols))

cross_df = cross_programme_evaluation(
    X_ext_raw      = X_ext_raw,
    y_ext          = y_ext,
    X_int_raw      = X_test_raw,
    X_int_proc     = X_test,
    y_int          = y_test,
    preprocessor   = preprocessor,
    model_registry = MODELS,
    methods        = ('platt', 'isotonic'),
)

print_cross_programme_table(cross_df)
cross_df.to_csv('cross_programme_results.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — THE TWO PROGRAMMES IN ONE EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
# The table says how much performance transfers. It cannot say whether the external
# cohort is the same kind of population seen from a different angle, or a genuinely
# different one the model is extrapolating into. This figure answers that, one row per
# programme:
#
#   column 1  where the cohort lands in the learned embedding, coloured by recorded
#             outcome — the shape of the population itself
#   column 2  which training cases one patient's prediction rests on
#   column 3  the same evidence, laid over the embedding
#
# Both rows share one projection, fitted once on the internal training cases. The
# training context is identical for both programmes by construction (column statistics
# attend to training rows, and the kernel keys are the context), so distance between
# the rows is distance between the cohorts, not between two separately fitted layouts.

import textwrap

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

if 'fit_explainer' not in globals():
    import os
    import sys
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    from kernelicl_clinical import fit_explainer

SEED = 0
CASES = None           # one test row per programme; None draws them at random
FINETUNED = None       # path to a kernelicl_finetune checkpoint, or None
CLASS_LABELS = globals().get('CLASS_LABELS', {})   # e.g. {0: 'Adherence', 1: 'Non-adherence'}
PROGRAMME_NAMES = ('Programme 1', 'Programme 2')

SAVE_FIGURES = None    # e.g. 'figures'
FIG_DPI = 300
FIG_FORMATS = ('png', 'pdf')

C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_2, INK_3 = '#0b0b0b', '#52514e', '#8a8984'
SURFACE, GRID = '#fcfcfb', '#e8e7e3'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK_3, 'axes.labelcolor': INK_2, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'font.size': 10,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'figure.dpi': 130,
})


def finish(fig, name):
    """Show a figure, and write a publication copy when SAVE_FIGURES is set."""
    if SAVE_FIGURES:
        from pathlib import Path
        directory = Path(SAVE_FIGURES)
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in FIG_FORMATS:
            fig.savefig(directory / f'{name}.{suffix}', dpi=FIG_DPI,
                        bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines.values():
        side.set_visible(False)
    return ax


def limits(v, margin=0.06):
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * margin or 1.0
    return lo - pad, hi + pad


# ── The explainer, and the fixed reference frame ─────────────────────────────────
if 'ex' not in globals():
    ex = fit_explainer(X_train_raw, y_train, X_test_raw, finetuned=FINETUNED)

head, clf = ex.head, ex.clf
y_train_enc = clf.y_encoder_.transform(ex.y_train)
n_classes = len(clf.classes_)

# Only the first three palette slots clear the all-pairs colourblind gate, so any
# further classes fold into one neutral rather than inventing a fourth hue.
keep = list(np.argsort(-np.bincount(y_train_enc, minlength=n_classes))[:3])
class_colors = [C_BLUE, C_ORANGE, C_AQUA]


def class_color(c):
    return class_colors[keep.index(c)] if c in keep else INK_3


def class_name(c):
    raw = clf.classes_[c]
    return f'{CLASS_LABELS.get(raw, raw)} ({raw})' if raw in CLASS_LABELS else f'class {raw}'


with torch.no_grad():
    H_train = head.embed(ex.E_train)[0].cpu().numpy().astype(np.float64)

try:
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=SEED).fit(H_train)
    PROJ_NAME = 'UMAP'
except ImportError:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, svd_solver='full').fit(H_train)
    PROJ_NAME = 'PCA'
    print('umap-learn not installed; using PCA, which shows only linear structure')

with np.errstate(all='ignore'):
    proj = np.asarray(reducer.transform(H_train))
order = np.argsort(proj[:, 0])
XLIM, YLIM = limits(proj[:, 0]), limits(proj[:, 1])


# ── Score both programmes against the same library, scale and thresholds ────────
# Judging the external cohort against its own spread of distances would make it
# unfamiliar by exactly the quantile and nothing else. Programme 1 sets the reference,
# so "beyond anything familiar" means beyond what the internal cohort looks like.
REFERENCE = ex.triage()['nearest_distance'].to_numpy()

programmes = [
    (PROGRAMME_NAMES[0], X_test_raw, np.asarray(y_test)),
    (PROGRAMME_NAMES[1], align_external(X_ext_raw, X_train_raw, verbose=False),
     np.asarray(y_ext)),
]

rows, rs = [], np.random.RandomState(SEED)
for i, (label, X_prog, y_prog) in enumerate(programmes):
    ex_p = ex.for_test_set(X_prog, reference_distances=REFERENCE)
    with torch.no_grad():
        H_test = head.embed(ex_p.E_test)[0].cpu().numpy().astype(np.float64)
    with np.errstate(all='ignore'):
        proj_test = np.asarray(reducer.transform(H_test))
    case = CASES[i] if CASES is not None else int(rs.randint(ex_p.w.shape[0]))
    rows.append({'label': label, 'ex': ex_p, 'y': y_prog,
                 'proj_test': proj_test, 'case': case})
    print(f'\n{label}: n={ex_p.m:,}  '
          f'{CLASS_LABELS.get(1, "positive")} rate {np.mean(y_prog == 1):.1%}')
    print(f'  evidence base    {np.median(ex_p.evidence_cases):.0f} of {ex_p.n:,} '
          f'cases (median)')
    print(f'  beyond Programme 1  {ex_p.is_novel.mean():.1%} of cases')
    print(f'  following test row {case}')

# One weight scale across both rows: normalising per row would draw a weight of 0.006
# as large as one of 0.89, and the point of the figure is that the rows may differ.
peak = max(float(r['ex'].w[r['case']].max()) for r in rows)


# ── The figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.7 * len(rows)),
                         gridspec_kw={'hspace': 0.40, 'wspace': 0.18})
axes = np.atleast_2d(axes)

for r, row in enumerate(rows):
    ex_p, proj_test, case = row['ex'], row['proj_test'], row['case']
    y_enc = clf.y_encoder_.transform(row['y'])
    wt = ex_p.w[case]

    # (1) the cohort in the embedding, coloured by what actually happened.
    ax = axes[r, 0]
    ax.scatter(proj[:, 0], proj[:, 1], s=4, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        at = y_enc == c
        if at.any():
            ax.scatter(proj_test[at, 0], proj_test[at, 1], s=7, color=class_color(c),
                       alpha=0.55, linewidths=0, zorder=2)
    bare(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_title(f'Cohort by recorded outcome\nn={ex_p.m:,}, '
                 f'{ex_p.is_novel.mean():.0%} beyond {PROGRAMME_NAMES[0]}',
                 fontsize=10, color=INK_2, loc='left', pad=8)

    # (2) which training cases this one patient's prediction rests on.
    ax = axes[r, 1]
    ax.vlines(np.arange(len(order)), 0, wt[order], color=C_BLUE, linewidth=0.7)
    ax.set_ylim(0, peak * 1.08)
    ax.set_xlim(0, len(order))
    ax.set_ylabel('contribution')
    ax.set_xlabel('training cases, ordered by embedding')
    ax.set_title(f'Evidence for test row {case}\n'
                 f'{ex_p.evidence_cases[case]:.0f} of {ex_p.n:,} cases carry it',
                 fontsize=10, color=INK_2, loc='left', pad=8)
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    # (3) the same evidence, in the embedding, over recorded training outcomes.
    ax = axes[r, 2]
    heavy = wt > wt.max() * 0.02
    ax.scatter(proj[~heavy, 0], proj[~heavy, 1], s=4, color=GRID, linewidths=0, zorder=1)
    for c in range(n_classes):
        at = heavy & (y_train_enc == c)
        if at.any():
            ax.scatter(proj[at, 0], proj[at, 1], s=8 + 260 * wt[at] / peak,
                       color=class_color(c), alpha=0.7, linewidths=0.5,
                       edgecolors=SURFACE, zorder=2)
    # An out-of-frame case is worth seeing, but letting one point rescale the axes
    # would squash the cloud, so clamp it to the frame and say so.
    tx, ty = proj_test[case]
    inside = XLIM[0] <= tx <= XLIM[1] and YLIM[0] <= ty <= YLIM[1]
    ax.scatter(np.clip(tx, *XLIM), np.clip(ty, *YLIM), marker='X', s=150,
               color=INK if inside else SURFACE, edgecolors=INK, linewidths=1.5, zorder=3)
    if not inside:
        ax.text(0.5, 0.02, 'case sits outside the training cloud', transform=ax.transAxes,
                ha='center', fontsize=8, color=INK_2, style='italic')
    bare(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    predicted = CLASS_LABELS.get(ex_p.pred[case], ex_p.pred[case])
    actual = CLASS_LABELS.get(row['y'][case], row['y'][case])
    ax.set_title(f'Evidence in the embedding\npredicted {predicted}, '
                 f'actually {actual}', fontsize=10, color=INK_2, loc='left', pad=8)

    axes[r, 0].text(-0.13, 0.5, textwrap.fill(row['label'], 26),
                    transform=axes[r, 0].transAxes, rotation=90, va='center',
                    ha='center', fontsize=11, color=INK)

for ax, letter in zip(axes.ravel(), 'abcdef'):
    ax.text(-0.04, 1.14, f'({letter})', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top', ha='right', color=INK)

handles = [Line2D([], [], marker='o', linestyle='', markersize=8, color=class_color(c),
                  label=class_name(c)) for c in range(n_classes)]
handles += [Line2D([], [], marker='o', linestyle='', markersize=5, color=GRID,
                   label='training case, negligible contribution'),
            Line2D([], [], marker='X', linestyle='', markersize=9, color=INK,
                   label='the case being predicted')]
fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f'Both programmes in the same learned embedding ({PROJ_NAME} of '
             f'{PROGRAMME_NAMES[0]} training cases)\nColumn 1 colours test cases by '
             f'recorded outcome; column 3 colours training cases by recorded outcome.',
             color=INK, fontsize=11, x=0.01, y=0.995, ha='left', va='top')
finish(fig, 'figure10_cross_programme')

# Notes
# -----
# Read column 1 first. If Programme 2 lands on the same regions as Programme 1 with
# the outcomes arranged the same way, performance should transfer and the table will
# say so. If it lands somewhere Programme 1 never occupied, the model is extrapolating
# and the metric gap has a cause you can point at rather than just measure.
#
# Column 2 is the operational consequence. A prediction built on a handful of cases is
# one a clinician should see; if the evidence base collapses for Programme 2 while it
# held for Programme 1, that is the transfer problem stated per patient.
#
# CASES is drawn at random so the figure is not cherry-picked. To pin the pair most
# worth showing:
#
#   CASES = [int(np.argmin(r['ex'].evidence_cases)) for r in rows]   thinnest evidence
#   CASES = [int(np.argmax(r['ex']._nn_test)) for r in rows]         least familiar
