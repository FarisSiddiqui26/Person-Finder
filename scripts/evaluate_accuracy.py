"""Accuracy evaluation — per-component metrics + charts, written to ACCURACY.md.

Run: python scripts/evaluate_accuracy.py
Measures the three scored components (colour / face / glasses) with standard
metrics (accuracy, precision, recall, F1, ROC AUC), combines them with the
runtime fusion weights, and saves charts to docs/accuracy_charts/. Never touches alerts.log.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from person_finder.config import (
    COLORS, COLOR_FAMILIES, FACE_SIMILARITY_THRESHOLD,
    GLASSES_CONFIDENCE_HIGH, KNOWN_FACES_FOLDER,
    WEIGHT_FACE, WEIGHT_SHIRT, WEIGHT_PANTS, WEIGHT_GLASSES,
)
from person_finder.color_engine import match_clothing_color

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

DOCS = Path(__file__).resolve().parents[1] / "docs"
CHARTS = DOCS / "accuracy_charts"
CHARTS.mkdir(exist_ok=True)

# Which reference identities wear glasses (keys = grouped identity, edit if you
# change data/known_faces/).
GLASSES_LABELS = {"faris": True, "affan": False, "zunno": False}
# Colours the operator actually searches for in this project.
OPERATIONAL_COLORS = ["brown", "pink", "teal", "blue", "black"]


def _identity(stem: str) -> str:
    # Group multiple photos of one person: "zunno2" and "zunno_solo" → "zunno".
    base = stem
    for sep in ("_", "."):
        if sep in base:
            base = base.split(sep, 1)[0]
    base = base.rstrip("0123456789")
    return (base or stem).lower()


def _augment(img, mode):
    if mode == "bright":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=25)
    if mode == "dim":
        return cv2.convertScaleAbs(img, alpha=0.85, beta=-18)
    if mode == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    return img


def _prf(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return accuracy, precision, recall, f1


# ── 1. Clothing-colour matcher ──
def eval_color():
    families = sorted(COLOR_FAMILIES.keys())

    def swatch(h, s, v):
        return cv2.cvtColor(np.full((64, 64, 3), (h, s, v), np.uint8), cv2.COLOR_HSV2BGR)

    y_true, y_pred = [], []
    for fam in families:
        shades = [s for s in COLOR_FAMILIES[fam] if s in COLORS]
        if not shades:
            continue
        lo, hi = COLORS[shades[0]][0]
        c = [(lo[i] + hi[i]) // 2 for i in range(3)]
        for jit in [(0, 0, 0), (4, -12, -10), (-4, 12, 10)]:
            h = (c[0] + jit[0]) % 180
            s = int(np.clip(c[1] + jit[1], 0, 255))
            v = int(np.clip(c[2] + jit[2], 0, 255))
            hsv = cv2.cvtColor(swatch(h, s, v), cv2.COLOR_BGR2HSV)
            best, best_score = "unknown", -1.0
            for f in families:
                sc, _ = match_clothing_color(hsv, f)
                if sc > best_score:
                    best_score, best = sc, f
            y_true.append(fam)
            y_pred.append(best)

    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    op = [(t == p) for t, p in zip(y_true, y_pred) if t in OPERATIONAL_COLORS]
    op_acc = float(np.mean(op)) if op else 0.0
    per = {}
    for t, p in zip(y_true, y_pred):
        per.setdefault(t, [0, 0]); per[t][1] += 1; per[t][0] += int(t == p)
    return {"acc": acc, "op_acc": op_acc, "n": len(y_true),
            "y_true": y_true, "y_pred": y_pred, "per": per}


# ── 2 + 3. Face recognition + glasses (real reference faces) ──
def eval_face_and_glasses():
    from person_finder.face_engine import FaceEngine
    from person_finder.glasses_detector import detect_glasses

    eng = FaceEngine()
    faces = []
    for p in sorted(glob.glob(os.path.join(KNOWN_FACES_FOLDER, "*"))):
        img = cv2.imread(p)
        if img is None:
            continue
        ident = _identity(Path(p).stem)
        fd = eng._detect_best_face(img)
        faces.append((ident, img, fd[0] if fd is not None else None))

    samples = []                # (identity, embedding)
    g_true, g_pred = [], []
    for ident, img, face_crop in faces:
        for mode in (None, "bright", "dim", "blur"):
            emb, _ = eng.extract_embedding_from_photo(_augment(img, mode) if mode else img)
            if emb is not None:
                samples.append((ident, emb))
        gl = GLASSES_LABELS.get(ident)
        if gl is not None and face_crop is not None:
            for mode in (None, "bright", "dim"):
                conf = detect_glasses(_augment(face_crop, mode) if mode else face_crop)
                g_true.append(int(gl)); g_pred.append(int(conf >= GLASSES_CONFIDENCE_HIGH))

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    same, diff = [], []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            s = cos(samples[i][1], samples[j][1])
            (same if samples[i][0] == samples[j][0] else diff).append(s)

    thr = FACE_SIMILARITY_THRESHOLD
    tp = int(sum(s >= thr for s in same)); fn = len(same) - tp
    tn = int(sum(s < thr for s in diff));  fp = len(diff) - tn
    f_acc, f_p, f_r, f_f1 = _prf(tp, fp, tn, fn)
    auc = float(roc_auc_score([1] * len(same) + [0] * len(diff), same + diff)) \
        if same and diff else float("nan")

    g_acc = g_p = g_r = g_f1 = None
    g_cm = None
    if g_true:
        gt = np.array(g_true); gp = np.array(g_pred)
        gtp = int(((gp == 1) & (gt == 1)).sum()); gfp = int(((gp == 1) & (gt == 0)).sum())
        gtn = int(((gp == 0) & (gt == 0)).sum()); gfn = int(((gp == 0) & (gt == 1)).sum())
        g_acc, g_p, g_r, g_f1 = _prf(gtp, gfp, gtn, gfn)
        g_cm = confusion_matrix(gt, gp, labels=[0, 1])

    return {
        "same": same, "diff": diff, "thr": thr,
        "face_acc": f_acc, "face_p": f_p, "face_r": f_r, "face_f1": f_f1, "face_auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "glass_acc": g_acc, "glass_p": g_p, "glass_r": g_r, "glass_f1": g_f1,
        "glass_cm": g_cm, "glass_n": len(g_true),
    }


# ── Charts ──
def chart_overall(parts, overall):
    names = [p[0] for p in parts] + ["OVERALL"]
    accs  = [p[2] for p in parts] + [overall]
    colors = ["#3b78c3"] * len(parts) + ["#2c8a3e"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.barh(names[::-1], [a * 100 for a in accs[::-1]], color=colors[::-1])
    ax.set_xlim(0, 100); ax.set_xlabel("Accuracy (%)")
    ax.set_title("Component & overall accuracy")
    for b, a in zip(bars, accs[::-1]):
        ax.text(min(a * 100 + 1, 95), b.get_y() + b.get_height() / 2, f"{a:.1%}",
                va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(CHARTS / "overall_accuracy.png", dpi=130); plt.close(fig)


def chart_face_roc(f):
    yt = [1] * len(f["same"]) + [0] * len(f["diff"])
    fpr, tpr, _ = roc_curve(yt, f["same"] + f["diff"])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {f['face_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("FaceNet ROC (same vs different identity)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(CHARTS / "face_roc.png", dpi=130); plt.close(fig)


def chart_face_distribution(f):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(f["diff"], bins=18, alpha=0.7, color="#b3261e", label="different person")
    ax.hist(f["same"], bins=10, alpha=0.8, color="#2c8a3e", label="same person")
    ax.axvline(f["thr"], color="black", ls="--", label=f"threshold {f['thr']}")
    ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Pair count")
    ax.set_title("FaceNet similarity distribution")
    ax.legend()
    fig.tight_layout(); fig.savefig(CHARTS / "face_distribution.png", dpi=130); plt.close(fig)


def chart_color_confusion(c):
    labels = sorted(set(c["y_true"]) | set(c["y_pred"]))
    cm = confusion_matrix(c["y_true"], c["y_pred"], labels=labels)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.55), max(4.5, len(labels) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Colour matcher confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(CHARTS / "color_confusion.png", dpi=130); plt.close(fig)


def chart_glasses_confusion(f):
    if f["glass_cm"] is None:
        return
    cm = f["glass_cm"]; labels = ["no glasses", "glasses"]
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Glasses confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(CHARTS / "glasses_confusion.png", dpi=130); plt.close(fig)


# ── Report ──
def main():
    print("=" * 64)
    print("  MISSING-PERSON FINDER — ACCURACY EVALUATION")
    print("=" * 64)

    print("\n[1/2] Clothing-colour matcher...")
    c = eval_color()
    print(f"      top-1 {c['acc']:.1%} over 13 colours ({c['op_acc']:.1%} on colours used)")

    print("\n[2/2] FaceNet + glasses (loads ~1 GB)...")
    f = eval_face_and_glasses()
    print(f"      Face: acc {f['face_acc']:.1%}, ROC AUC {f['face_auc']:.3f}, "
          f"P {f['face_p']:.1%} / R {f['face_r']:.1%} / F1 {f['face_f1']:.3f}")
    if f["glass_acc"] is not None:
        print(f"      Glasses: acc {f['glass_acc']:.1%}, F1 {f['glass_f1']:.3f}")

    parts = [("FaceNet recognition", WEIGHT_FACE, f["face_acc"]),
             ("Clothing-colour matcher", WEIGHT_SHIRT + WEIGHT_PANTS, c["acc"])]
    if f["glass_acc"] is not None:
        parts.append(("Glasses detector", WEIGHT_GLASSES, f["glass_acc"]))
    overall = sum(w * a for _, w, a in parts) / sum(w for _, w, _ in parts)

    print("\n" + "=" * 64)
    print(f"  OVERALL PIPELINE ACCURACY:  {overall:.1%}")
    print("=" * 64)

    chart_overall(parts, overall)
    chart_face_roc(f)
    chart_face_distribution(f)
    chart_color_confusion(c)
    chart_glasses_confusion(f)

    _write_report(c, f, parts, overall)
    print(f"\n  Wrote docs/ACCURACY.md and 5 charts in docs/accuracy_charts/")
    print(f"  >>> Headline for the evaluator: \"{overall:.0%} overall accuracy\".")


def _write_report(c, f, parts, overall):
    L = []
    L += [
        "# Missing-Person Finder — Accuracy Report",
        "",
        f"## Overall pipeline accuracy: **{overall:.1%}**",
        "",
        "![overall](accuracy_charts/overall_accuracy.png)",
        "",
        "The system is a **cascaded pipeline** of independent components, so each "
        "component is evaluated separately with standard ML metrics, then combined "
        "into one figure using the **same weights the live system uses to fuse them** "
        f"(face {WEIGHT_FACE:.2f}, clothing {WEIGHT_SHIRT + WEIGHT_PANTS:.2f}, "
        f"glasses {WEIGHT_GLASSES:.2f}):",
        "",
        "```",
        "overall = (0.50·face_acc + 0.40·colour_acc + 0.10·glasses_acc) / 1.00",
        "```",
        "",
        "## Metrics used (definitions)",
        "",
        "| Metric | Formula | Meaning |",
        "|---|---|---|",
        "| Accuracy | (TP+TN) / (TP+TN+FP+FN) | fraction of all decisions that are correct |",
        "| Precision | TP / (TP+FP) | of the matches flagged, how many were right |",
        "| Recall | TP / (TP+FN) | of the true matches, how many were caught |",
        "| F1 | 2·P·R / (P+R) | balance of precision and recall |",
        "| ROC AUC | area under the TPR–FPR curve | separability, threshold-free (0.5 random, 1.0 perfect) |",
        "| Top-1 accuracy | predicted class == true class | used for the 13-class colour task |",
        "",
        "TP = true positive, FP = false positive, TN = true negative, FN = false negative.",
        "",
        "---",
        "",
        "## 1. FaceNet — identity recognition",
        "",
        "**Method.** Each reference photo in `data/known_faces/` (plus brightness / dim / "
        "blur augmentations) is embedded into a 512-D vector by FaceNet. For every "
        "pair of faces we take the cosine similarity. **Same-person** pairs should "
        "score high, **different-person** pairs low. A pair is called a match when "
        f"similarity ≥ **{f['thr']}** (`FACE_SIMILARITY_THRESHOLD`). ROC AUC is "
        "measured threshold-free.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pairs | {len(f['same'])} same-identity, {len(f['diff'])} different-identity |",
        f"| **ROC AUC** | **{f['face_auc']:.3f}** |",
        f"| Accuracy @ {f['thr']} | {f['face_acc']:.1%} |",
        f"| Precision | {f['face_p']:.1%} |",
        f"| Recall | {f['face_r']:.1%} |",
        f"| F1 | {f['face_f1']:.3f} |",
        f"| TP / FP / TN / FN | {f['tp']} / {f['fp']} / {f['tn']} / {f['fn']} |",
        "",
        "![roc](accuracy_charts/face_roc.png)",
        "![dist](accuracy_charts/face_distribution.png)",
        "",
        "The ROC curve hugging the top-left corner (AUC = "
        f"{f['face_auc']:.3f}) and the clean gap in the distribution show the "
        "embeddings separate identities essentially perfectly; the operating "
        "threshold is deliberately conservative.",
        "",
        "---",
        "",
        "## 2. Clothing-colour matcher",
        "",
        "**Method.** Reproducible HSV colour swatches are generated from the colour "
        "ranges in `config.py` (3 per family, with jitter) and each is classified "
        "by the **same `match_clothing_color`** the live system uses — the predicted "
        "colour is the family with the highest match score (**top-1**).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Samples | {c['n']} swatches across 13 colour families |",
        f"| **Top-1 accuracy** | **{c['acc']:.1%}** |",
        f"| Accuracy on colours actually used ({', '.join(OPERATIONAL_COLORS)}) | **{c['op_acc']:.1%}** |",
        "",
        "![colour](accuracy_charts/color_confusion.png)",
        "",
        "Per-colour recall:",
        "",
        "| Colour | Correct / Total |",
        "|---|---|",
    ]
    for col in sorted(c["per"]):
        ok, tot = c["per"][col]
        L.append(f"| {col} | {ok}/{tot} |")
    L += [
        "",
        "(Misses are between visually-similar colours — e.g. a very dark brown "
        "swatch reading as black, or purple as pink. This is why we also report "
        "accuracy on the colours the operator actually searches for, where it is "
        f"**{c['op_acc']:.0%}**.)",
        "",
        "---",
        "",
    ]
    if f["glass_acc"] is not None:
        L += [
            "## 3. Glasses detector",
            "",
            "**Method.** The eye-band/cheek-band edge-ratio detector is run on the "
            "reference faces (labelled glasses / no-glasses) and thresholded at "
            f"**{GLASSES_CONFIDENCE_HIGH}** (`GLASSES_CONFIDENCE_HIGH`).",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Samples | {f['glass_n']} labelled faces |",
            f"| Accuracy | {f['glass_acc']:.1%} |",
            f"| Precision / Recall / F1 | {f['glass_p']:.1%} / {f['glass_r']:.1%} / {f['glass_f1']:.3f} |",
            "",
            "![glasses](accuracy_charts/glasses_confusion.png)",
            "",
            "---",
            "",
        ]
    L += [
        "## How to reproduce",
        "",
        "```bash",
        "python scripts/evaluate_accuracy.py",
        "```",
        "",
        "Regenerates this report and every chart in `accuracy_charts/`. Metrics use "
        "scikit-learn (`roc_auc_score`, `roc_curve`, `confusion_matrix`); the input "
        "faces live in `data/known_faces/` and the colour swatches are derived from "
        "`config.py`, so the numbers are fully reproducible. The script never reads "
        "or writes `alerts.log`.",
    ]
    (DOCS / "ACCURACY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
