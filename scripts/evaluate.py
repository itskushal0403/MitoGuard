"""Evaluate every system on every fold's held-out cases.

Systems (per fold):
  yolo                 Stage 1 alone, threshold on p_mitosis
  yolo+<backbone>      Stage 1 candidate pool filtered by the Stage 2 MLP on frozen <backbone> features
  *_nostain            same, with the Stage 1 detector trained without stain augmentation (ablation)

Every threshold (detection and Platt calibration) is fitted on the fold's validation cases, which
come from the training domains only.  Metrics are computed on the held-out domain.
Outputs results/folds.csv (one row per fold x system) and results/cases.csv (one row per case).
"""
import argparse
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import WORK_ROOT, load_cases, make_folds  # noqa: E402
from mitoguard.metrics import best_threshold, evaluate_set, label_candidates, large_error_label, safe_auroc  # noqa: E402
from mitoguard.reliability import region_scores  # noqa: E402

POOL_THR = 0.05
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def platt(scores_val, labels_val):
    lr = LogisticRegression(C=1e3).fit(np.log(np.clip(scores_val, 1e-6, 1)[:, None]), labels_val)
    return lambda s: lr.predict_proba(np.log(np.clip(s, 1e-6, 1))[:, None])[:, 1] if len(s) else s


def systems_for(tag, backbones):
    out = []
    for variant in ("stain", "nostain"):
        run = f"{tag}_{variant}"
        if not os.path.exists(os.path.join(WORK_ROOT, "cands", run + ".pkl")):
            continue
        sfx = "" if variant == "stain" else "_nostain"
        out.append((f"yolo{sfx}", run, None, "yolo"))
        for bb in backbones:
            if os.path.exists(os.path.join(WORK_ROOT, "stage2", f"{run}__{bb}.pkl")):
                out.append((f"yolo+{bb}{sfx}", run, bb, "filter"))
                out.append((f"yolo+{bb}+fused{sfx}", run, bb, "fused"))  # sqrt(p_yolo * p_filter)
    return out


def eval_fold(job):
    protocol, fold, backbones = job
    cases = load_cases()
    fold_rows, case_rows = [], []
    for system, run, bb, mode in systems_for(fold.tag, backbones):
        cands = pickle.load(open(os.path.join(WORK_ROOT, "cands", run + ".pkl"), "rb"))
        s2 = pickle.load(open(os.path.join(WORK_ROOT, "stage2", f"{run}__{bb}.pkl"), "rb"))["scores"] if bb else None
        # detection score and (for reliability) per-member scores, aligned with the candidate pool
        dets, members = {}, {}
        for cid in fold.val + fold.test:
            c = cands[cid]
            pool = c["p_mit"] >= POOL_THR
            score = {"yolo": lambda: c["p_mit"][pool], "filter": lambda: s2[cid]["p2"][pool],
                     "fused": lambda: np.sqrt(c["p_mit"][pool] * s2[cid]["p2"][pool])}[mode]()
            dets[cid] = {"xy": c["xy"][pool], "score": score}
            members[cid] = s2[cid]["p2_members"][pool] if mode == "filter" else None
        thr, val_f1 = best_threshold(dets, fold.val)
        # calibrated probabilities for the reliability score (Stage 2 is already Platt-calibrated)
        if mode == "filter":
            cal = lambda s: s  # noqa: E731
            cal_thr = thr
        else:
            sv = np.concatenate([dets[c]["score"] for c in fold.val])
            yv = np.concatenate([label_candidates(c, dets[c]["xy"]) == 1 for c in fold.val]).astype(int)
            cal = platt(sv, yv)
            cal_thr = float(cal(np.array([thr]))[0])
        res = evaluate_set(dets, fold.test, thr)
        per_case = {r["case_id"]: r for r in res.pop("per_case")}
        res["pool_recall"] = evaluate_set(dets, fold.test, 0.0)["recall"]  # ceiling for any filter
        val_case = {r["case_id"]: r for r in evaluate_set(dets, fold.val, thr)["per_case"]}
        rel_rows = []
        for cid in fold.val + fold.test:
            p = cal(dets[cid]["score"])
            rs = region_scores(p, cal_thr, members[cid])
            rel_rows.append({"case_id": cid, **rs, "split": "test" if cid in per_case else "val"})
        rel = pd.DataFrame(rel_rows).set_index("case_id")
        test_rel = rel[rel.split == "test"]
        gts = np.array([per_case[c]["n_gt"] for c in test_rel.index])
        y_large = large_error_label(test_rel["count"], gts, 0.2)
        y_large3 = large_error_label(test_rel["count"], gts, 0.2, min_abs=3)
        row = {"protocol": protocol, "fold": fold.name, "system": system, "threshold": thr, "val_f1": val_f1,
               **res, "n_test_cases": len(fold.test), "frac_large_err": float(y_large.mean())}
        for u in ("u_var", "u_relstd", "u_ens", "u_gap"):
            row[f"auroc_{u}"] = safe_auroc(y_large, test_rel[u].fillna(0))
            row[f"auroc3_{u}"] = safe_auroc(y_large3, test_rel[u].fillna(0))
        fold_rows.append(row)
        for cid in fold.test + fold.val:
            split = "test" if cid in per_case else "val"
            pc = per_case[cid] if split == "test" else val_case[cid]
            case_rows.append({"protocol": protocol, "fold": fold.name, "system": system, "case_id": cid,
                              "split": split,
                              "tumor": cases.loc[cid, "tumor_short"], "scanner": cases.loc[cid, "scanner_short"],
                              "official_split": cases.loc[cid, "official_split"],
                              **{k: pc[k] for k in ("tp", "fp", "fn", "n_gt", "imp_hit", "imp_total")},
                              **rel.loc[cid].drop("split").to_dict()})
        print(f"{protocol:8s} {fold.name:6s} {system:22s} thr={thr:.2f} val_F1={val_f1:.3f} "
              f"test F1={res['f1']:.3f} P={res['precision']:.3f} R={res['recall']:.3f} "
              f"impFP={res['imposter_fp_rate']:.3f} MAE={res['count_mae']:.2f} "
              f"AUROC(u_var)={row['auroc_u_var']:.3f}", flush=True)
    return fold_rows, case_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["loto", "loso", "indomain"])
    ap.add_argument("--backbones", nargs="+", default=["phikon", "phikon2", "resnet50", "uni"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    jobs = [(p, f, args.backbones) for p in args.protocols for f in make_folds(p)]
    fold_rows, case_rows = [], []
    with Pool(min(args.workers, len(jobs))) as pool:
        for fr, cr in pool.imap(eval_fold, jobs):
            fold_rows += fr
            case_rows += cr
    pd.DataFrame(fold_rows).to_csv(os.path.join(RESULTS, "folds.csv"), index=False)
    pd.DataFrame(case_rows).to_csv(os.path.join(RESULTS, "cases.csv"), index=False)


if __name__ == "__main__":
    main()
