"""Evaluate every system on every fold's held-out cases.

Systems (per fold):
  yolo                 Stage 1 alone, threshold on p_mitosis
  yolo+<backbone>      Stage 1 candidate pool filtered by the Stage 2 MLP on frozen <backbone> features
  *_nostain            same, with the Stage 1 detector trained without stain augmentation (ablation)

Every threshold (detection and Platt calibration) is fitted on the fold's validation cases, which
come from the training domains only.  Metrics are computed on the held-out domain.
Outputs results/folds.csv (one row per fold x system) and results/cases.csv.gz (one row per case).
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
from mitoguard.adapt import em_prior_shift  # noqa: E402
from mitoguard.stage2 import apply_stacker, fit_stacker, in_pool, s2_name  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def platt(scores_val, labels_val):
    lr = LogisticRegression(C=1e3).fit(np.log(np.clip(scores_val, 1e-6, 1)[:, None]), labels_val)
    return lambda s: lr.predict_proba(np.log(np.clip(s, 1e-6, 1))[:, None])[:, 1] if len(s) else s


def systems_for(tag, backbones, pools=("m05",)):
    """(system name, yolo run, feature spec or None, mode, candidate pool)."""
    out = []
    for variant in ("stain", "nostain", "union"):
        run = f"{tag}_{variant}"
        if not os.path.exists(os.path.join(WORK_ROOT, "cands", run + ".pkl")):
            continue
        sfx = "" if variant == "stain" else f"_{variant}"
        out.append((f"yolo{sfx}", run, None, "yolo", "m05"))
        for bb in backbones:
            for pool in pools:
                if os.path.exists(os.path.join(WORK_ROOT, "stage2", s2_name(run, bb, pool) + ".pkl")):
                    tag_ = bb if pool == "m05" else f"{bb}@{pool}"
                    out.append((f"yolo+{tag_}{sfx}", run, bb, "filter", pool))
                    out.append((f"yolo+{tag_}+fused{sfx}", run, bb, "fused", pool))  # sqrt(p_yolo * p_filter)
                    out.append((f"yolo+{tag_}+stack{sfx}", run, bb, "stack", pool))  # learned on validation candidates
    return out


def em_variants(dets, fold, members):
    """Label-shift adapted copies of calibrated detections: per region and per site (val / test batch)."""
    pi_s = float(np.concatenate([dets[c]["score"] for c in fold.val]).mean())
    region = {cid: {**d, "score": em_prior_shift(d["score"], pi_s)} for cid, d in dets.items()}
    site = {}
    for ids in (fold.val, fold.test):
        allp = np.concatenate([dets[c]["score"] for c in ids])
        adapted = em_prior_shift(allp, pi_s)
        k = 0
        for cid in ids:
            n = len(dets[cid]["score"])
            site[cid] = {**dets[cid], "score": adapted[k:k + n]}
            k += n
    none = {cid: None for cid in dets}
    return [("+emregion", region, none), ("+emsite", site, none)]


def eval_fold(job):
    """All systems of one fold (sequential)."""
    protocol, fold, backbones, pools, adapt = job
    fold_rows, case_rows = [], []
    for spec in systems_for(fold.tag, backbones, pools):
        fr, cr = eval_task((protocol, fold, spec, adapt))
        fold_rows += fr
        case_rows += cr
    return fold_rows, case_rows


def eval_task(job):
    """One (fold, system) pair -> fold rows and case rows (parallelised over pairs in main)."""
    protocol, fold, spec, adapt = job
    cases = load_cases()
    fold_rows, case_rows = [], []
    for base_system, run, bb, mode, pool_name in [spec]:
        cands = pickle.load(open(os.path.join(WORK_ROOT, "cands", run + ".pkl"), "rb"))
        s2 = pickle.load(open(os.path.join(WORK_ROOT, "stage2", s2_name(run, bb, pool_name) + ".pkl"), "rb"))["scores"] if bb else None
        # detection score and (for reliability) per-member scores, aligned with the candidate pool
        dets, members = {}, {}
        for cid in fold.val + fold.test:
            c = cands[cid]
            pool = in_pool(c, pool_name)
            score = {"yolo": lambda: c["p_mit"][pool], "filter": lambda: s2[cid]["p2"][pool],
                     "fused": lambda: np.sqrt(c["p_mit"][pool] * s2[cid]["p2"][pool]),
                     "stack": lambda: s2[cid]["p2"][pool]}[mode]()
            dets[cid] = {"xy": c["xy"][pool], "score": score, "p_mit": c["p_mit"][pool], "p_imp": c["p_imp"][pool]}
            members[cid] = s2[cid]["p2_members"][pool] if mode == "filter" else None
        if mode == "stack":
            # 3-feature logistic regression fitted on validation candidates, where YOLO's scores are
            # out-of-sample (on its own training cases YOLO is over-confident, so fitting there would leak)
            cat = lambda k: np.concatenate([dets[c][k] for c in fold.val])  # noqa: E731
            yv = np.concatenate([label_candidates(c, dets[c]["xy"]) == 1 for c in fold.val]).astype(int)
            lr = fit_stacker(cat("score"), cat("p_mit"), cat("p_imp"), yv)
            os.makedirs(os.path.join(WORK_ROOT, "stackers"), exist_ok=True)
            pickle.dump(lr, open(os.path.join(WORK_ROOT, "stackers", f"{fold.tag}__{base_system}.pkl"), "wb"))
            for d in dets.values():
                d["score"] = apply_stacker(lr, d["score"], d["p_mit"], d["p_imp"])
        variants = [(base_system, dets, members)]
        if adapt and mode in ("filter", "stack"):
            variants += [(base_system + sfx, d, m) for sfx, d, m in em_variants(dets, fold, members)]
        for system, dets, members in variants:
            score_variant(protocol, fold, cases, system, mode, dets, members, fold_rows, case_rows)
    return fold_rows, case_rows


def score_variant(protocol, fold, cases, system, mode, dets, members, fold_rows, case_rows):
    thr, val_f1 = best_threshold(dets, fold.val)
    # calibrated probabilities for the reliability score (Stage 2 is already Platt-calibrated)
    if mode in ("filter", "stack"):  # already calibrated probabilities
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", nargs="+", default=["loto", "loso", "indomain"])
    ap.add_argument("--backbones", nargs="+", default=["phikon", "phikon2", "resnet50", "uni"])
    ap.add_argument("--pools", nargs="+", default=["m05"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--adapt", action="store_true", help="also evaluate EM label-shift adapted variants")
    ap.add_argument("--out", default=RESULTS, help="output directory for folds.csv / cases.csv.gz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(p, f, spec, args.adapt) for p in args.protocols for f in make_folds(p)
            for spec in systems_for(f.tag, args.backbones, args.pools)]
    print(f"{len(jobs)} (fold, system) evaluations on {args.workers} workers", flush=True)
    fold_rows, case_rows = [], []
    with Pool(min(args.workers, len(jobs))) as pool:
        for fr, cr in pool.imap_unordered(eval_task, jobs, chunksize=1):
            fold_rows += fr
            case_rows += cr
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out, "folds.csv"), index=False)
    pd.DataFrame(case_rows).to_csv(os.path.join(args.out, "cases.csv.gz"), index=False)


if __name__ == "__main__":
    main()
