"""Stage 2 v2: fine-tune a pathology foundation model (LoRA) as a mitosis-vs-imposter cell classifier.

Per fold:
  candidates  = union of the fold's two Stage 1 detectors (with / without stain augmentation),
                every box with max(p_mitosis, p_imposter) >= 0.05, mapped to crop-bank points
  train set   = training-case candidates (hard negatives + detected mitoses) + all labelled cells
  val set     = validation-case candidates (training domains only): early stopping
  scored set  = validation + held-out candidates, scored with 8-view dihedral TTA
Output: WORK/ft/<fold>__<backbone>.npz with bank rows and probabilities (raw, uncalibrated).

Example: python scripts/finetune_stage2.py --fold loto_HBC --backbone phikon2
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import IMPOSTER, MITOSIS, WORK_ROOT, cells_of, get_fold  # noqa: E402
from mitoguard.ft import CellClassifier, augment, predict  # noqa: E402
from mitoguard.metrics import label_candidates  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

CAND_THR = 0.05
BANK = os.path.join(WORK_ROOT, "bank")


def fold_rows(fold, bank, off, cands, split):
    """Unique bank rows + labels (1 = within matching radius of a mitosis) for a split."""
    rows = []
    for cid in getattr(fold, split):
        tree = cKDTree(bank[cid])
        pts = [c[cid]["xy"][np.maximum(c[cid]["p_mit"], c[cid]["p_imp"]) >= CAND_THR] for c in cands]
        if split == "train":
            pts += [cells_of(cid, MITOSIS), cells_of(cid, IMPOSTER)]
        pts = np.concatenate([p for p in pts if len(p)]) if any(len(p) for p in pts) else np.zeros((0, 2))
        if len(pts) == 0:
            continue
        d, i = tree.query(pts)
        assert d.max() <= 4.01
        rows.append(off[cid][0] + np.unique(i))
    rows = np.unique(np.concatenate(rows))
    # labels from the bank point positions
    case_of = np.empty(len(rows), np.int64)
    starts = np.array(sorted(v[0] for v in off.values()))
    cids = np.array(sorted(off, key=lambda k: off[k][0]))
    case_of = cids[np.searchsorted(starts, rows, side="right") - 1]
    labels = np.zeros(len(rows), np.int64)
    for cid in np.unique(case_of):
        m = case_of == cid
        local = rows[m] - off[cid][0]
        labels[m] = (label_candidates(int(cid), bank[cid][local]) == 1).astype(np.int64)
    return rows, labels


def load_crops(rows):
    mm = np.load(os.path.join(BANK, "crops224.npy"), mmap_mode="r")
    order = np.argsort(rows)
    out = np.empty((len(rows), 224, 224, 3), np.uint8)
    for k in range(0, len(rows), 4096):
        idx = order[k:k + 4096]
        out[idx] = mm[rows[idx]]
    return torch.from_numpy(out)


def batches(t, bs):
    for k in range(0, len(t), bs):
        yield t[k:k + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--backbone", default="phikon2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--zoom", type=int, default=112)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=0, help="subsample training rows (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix for the output name (ablations)")
    ap.add_argument("--self-train-only", action="store_true",
                    help="skip training: reload the saved model + scores and run only the self-training round")
    ap.add_argument("--st-rounds", type=int, default=1, help="self-training rounds (output suffix +st, +st2, ...)")
    ap.add_argument("--st-budget", choices=["fixed", "prevalence"], default="fixed",
                    help="pseudo-positive budget: fixed top 10%% (pre-specified) or prevalence-aware: 70%% of the "
                         "site's expected mitosis count from validation-calibrated probabilities, capped at 10%% "
                         "(post-hoc fix after the fixed budget failed on a low-prevalence site); suffix +stpa")
    ap.add_argument("--self-train", action="store_true",
                    help="also run one round of class-balanced self-training on the held-out site's unlabelled "
                         "candidates (unsupervised site adaptation); saved separately with suffix '+st'")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"
    out_dir = os.path.join(WORK_ROOT, "ft")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.fold}__{args.backbone}{args.tag}"
    fold = get_fold(args.fold)
    bank = pickle.load(open(os.path.join(BANK, "bank.pkl"), "rb"))
    off = pickle.load(open(os.path.join(BANK, "offsets.pkl"), "rb"))
    cands = [pickle.load(open(os.path.join(WORK_ROOT, "cands", f"{args.fold}_{v}.pkl"), "rb")) for v in ("stain", "nostain")]
    t0 = time.time()
    tr_rows, tr_y = fold_rows(fold, bank, off, cands, "train")
    va_rows, va_y = fold_rows(fold, bank, off, cands, "val")
    te_rows, _ = fold_rows(fold, bank, off, cands, "test")
    if args.max_train:
        sel = np.random.default_rng(0).choice(len(tr_rows), min(args.max_train, len(tr_rows)), replace=False)
        tr_rows, tr_y = tr_rows[sel], tr_y[sel]
    print(f"{name}: train {len(tr_rows)} (pos {tr_y.sum()}), val {len(va_rows)} (pos {va_y.sum()}), test {len(te_rows)}"
          f"  [{time.time() - t0:.0f}s]", flush=True)
    if args.self_train_only:
        model = CellClassifier(args.backbone, lora_r=args.lora_r, zoom=args.zoom).to(dev)
        model.load_state_dict(torch.load(os.path.join(out_dir, name + ".pt"), map_location=dev), strict=False)
        saved = np.load(os.path.join(out_dir, name + ".npz"))
        assert np.array_equal(saved["rows"], np.concatenate([va_rows, te_rows]))
        xva, xte = load_crops(va_rows), load_crops(te_rows)
        self_train(args, model, name, out_dir, tr_rows, tr_y, va_rows, va_y, xva, te_rows, xte,
                   saved["prob"][len(va_rows):], dev, t0)
        return
    xtr, xva = load_crops(tr_rows), load_crops(va_rows)
    print(f"crops loaded [{time.time() - t0:.0f}s]", flush=True)

    model = CellClassifier(args.backbone, lora_r=args.lora_r, zoom=args.zoom).to(dev)
    params = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")], "lr": args.lr},
              {"params": model.head.parameters(), "lr": args.lr * 5}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    steps = args.epochs * (len(tr_rows) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 5], total_steps=steps, pct_start=0.1)
    pos_weight = torch.tensor((tr_y == 0).sum() / max(tr_y.sum(), 1), device=dev)
    ytr = torch.from_numpy(tr_y).float()
    best, best_state, hist = -1, None, []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr_rows))
        tl, n = 0.0, 0
        for k in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[k:k + args.batch]
            x = augment(xtr[idx].to(dev, non_blocking=True), args.zoom)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), ytr[idx].to(dev), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            sched.step()
            tl, n = tl + loss.item(), n + 1
        pv = predict(model, batches(xva, 256), args.zoom, tta=1)
        ap_ = average_precision_score(va_y, pv)
        hist.append({"epoch": ep + 1, "train_loss": tl / max(n, 1), "val_ap": ap_})
        print(f"epoch {ep + 1}: loss {tl / max(n, 1):.4f} val AP {ap_:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if ap_ > best:
            best = ap_
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k or k.startswith("head")}
    model.load_state_dict(best_state, strict=False)
    del xtr
    xte = load_crops(te_rows)
    p_val = predict(model, batches(xva, 128), args.zoom, tta=args.tta)
    p_test = predict(model, batches(xte, 128), args.zoom, tta=args.tta)
    print(f"final val AP (TTA {args.tta}) {average_precision_score(va_y, p_val):.4f} [{time.time() - t0:.0f}s]", flush=True)
    np.savez(os.path.join(out_dir, name + ".npz"), rows=np.concatenate([va_rows, te_rows]),
             prob=np.concatenate([p_val, p_test]).astype(np.float32))
    torch.save(best_state, os.path.join(out_dir, name + ".pt"))
    json.dump({"history": hist, "best_val_ap": best, "args": vars(args)}, open(os.path.join(out_dir, name + ".json"), "w"), indent=1)
    if args.self_train:
        self_train(args, model, name, out_dir, tr_rows, tr_y, va_rows, va_y, xva, te_rows, xte, p_test, dev, t0)


def self_train(args, model, name, out_dir, tr_rows, tr_y, va_rows, va_y, xva, te_rows, xte, p_test, dev, t0):
    """Class-balanced self-training (Zou et al., ECCV 2018) on the target site's unlabelled candidates.

    Class-balanced, rank-based selection (robust to miscalibration under shift): pseudo-positives are the
    top 10% of target candidates (below the ~15% mitosis share seen on validation, i.e. conservative),
    pseudo-negatives are sampled from the bottom 50% (3x the positives). One epoch of LoRA updates on pseudo-labelled target crops mixed 1:1 with
    labelled source crops (anchors the decision boundary), at 1/4 of the source learning rate.
    """
    rng = np.random.default_rng(0)
    cal = None
    if args.st_budget == "prevalence":
        from sklearn.linear_model import LogisticRegression
        pv = predict(model, batches(xva, 128), args.zoom, tta=args.tta)
        lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))  # noqa: E731
        platt = LogisticRegression(C=1e3).fit(lg(pv)[:, None], va_y)
        cal = lambda p: platt.predict_proba(lg(p)[:, None])[:, 1]  # noqa: E731
    for rnd in range(args.st_rounds):
        p_test = _self_train_round(args, model, rng, tr_rows, tr_y, xte, p_test, dev, cal)
        if p_test is None:
            return
    p_val = predict(model, batches(xva, 128), args.zoom, tta=args.tta)
    print(f"after {args.st_rounds} self-training round(s): val AP {average_precision_score(va_y, p_val):.4f} [{time.time() - t0:.0f}s]", flush=True)
    sfx = ("+st" if args.st_rounds == 1 else f"+st{args.st_rounds}") + ("pa" if args.st_budget == "prevalence" else "")
    np.savez(os.path.join(out_dir, name + sfx + ".npz"), rows=np.concatenate([va_rows, te_rows]),
             prob=np.concatenate([p_val, p_test]).astype(np.float32))


def _self_train_round(args, model, rng, tr_rows, tr_y, xte, p_test, dev, cal=None):
    """One round: select pseudo-labels from current target scores, 1 epoch of LoRA updates, re-score target."""
    order = np.argsort(-p_test)
    k = int(0.10 * len(order))
    if cal is not None:  # prevalence-aware: never claim more mitoses than the site is expected to contain
        expected = float(cal(p_test).sum())
        k = int(min(k, 0.7 * expected))
        print(f"prevalence-aware budget: expected mitoses {expected:.0f} of {len(order)} candidates -> {k} pseudo-positives", flush=True)
    pos = order[:k]
    bottom = order[len(order) // 2:]
    neg = rng.choice(bottom, min(len(bottom), 3 * max(len(pos), 1)), replace=False)
    src = rng.choice(len(tr_rows), min(len(tr_rows), len(pos) + len(neg)), replace=False)
    print(f"self-training: {len(pos)} pseudo-positives, {len(neg)} pseudo-negatives, {len(src)} source anchors", flush=True)
    if len(pos) < 10:
        return None
    xs = load_crops(tr_rows[src])
    x_all = torch.cat([xte[pos], xte[neg], xs])
    y_all = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg)), torch.from_numpy(tr_y[src]).float()])
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr / 4, weight_decay=0.05)
    pw = torch.tensor(float((y_all == 0).sum() / max((y_all == 1).sum(), 1)), device=dev)
    model.train()
    perm = torch.randperm(len(x_all))
    for k in range(0, len(perm) - args.batch + 1, args.batch):
        idx = perm[k:k + args.batch]
        x = augment(x_all[idx].to(dev), args.zoom)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), y_all[idx].to(dev), pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return predict(model, batches(xte, 128), args.zoom, tta=args.tta)


if __name__ == "__main__":
    main()
