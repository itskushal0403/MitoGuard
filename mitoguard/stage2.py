"""Stage 2: imposter filter on frozen foundation-model features.

A small MLP ensemble separates true mitotic figures from imposters.  Training data per fold:
  positives      : labelled mitoses + Stage 1 candidates matching a mitosis (training cases)
  negatives      : labelled imposters + Stage 1 false positives (hard negatives, training cases)
The ensemble is early-stopped and Platt-calibrated on the validation cases, which come from the
training domains only; the held-out domain is never touched.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


POOLS = {
    # Stage 1 candidate pool definitions: which detections Stage 2 gets to see
    "m05": lambda c: c["p_mit"] >= 0.05,  # mitosis score only (default)
    "m05i10": lambda c: (c["p_mit"] >= 0.05) | (c["p_imp"] >= 0.10),  # + boxes YOLO called "imposter"
    "a05": lambda c: np.maximum(c["p_mit"], c["p_imp"]) >= 0.05,  # any class (widest; for strong filters)
}


def in_pool(c: dict, pool: str = "m05") -> np.ndarray:
    return POOLS[pool](c)


def s2_name(run: str, feature: str, pool: str = "m05") -> str:
    """File stem of a Stage 2 result; the default pool keeps the short name."""
    return f"{run}__{feature}" if pool == "m05" else f"{run}__{feature}@{pool}"


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_features(p_filter, p_mit, p_imp) -> np.ndarray:
    return np.stack([_logit(p_filter), _logit(p_mit), _logit(p_imp)], 1)


def fit_stacker(p_filter, p_mit, p_imp, y) -> LogisticRegression:
    """Learned fusion of the Stage 2 probability with YOLO's two class scores.

    Fitted on validation-case candidates only: there YOLO's scores are out-of-sample, whereas on its own
    training cases YOLO is over-confident and a stacker would learn to over-trust it.
    """
    return LogisticRegression(C=1.0).fit(stack_features(p_filter, p_mit, p_imp), y)


def apply_stacker(lr, p_filter, p_mit, p_imp) -> np.ndarray:
    if len(p_filter) == 0:
        return np.zeros(0, np.float32)
    return lr.predict_proba(stack_features(p_filter, p_mit, p_imp))[:, 1].astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden=256, p=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Dropout(0.1), torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Dropout(p),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_one(xtr, ytr, xva, yva, seed, epochs=40, lr=1e-3, wd=1e-2, batch=512, device="cuda"):
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = MLP(xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=dev, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xtr_t, ytr_t = torch.from_numpy(xtr).to(dev), torch.from_numpy(ytr.astype(np.float32)).to(dev)
    xva_t = torch.from_numpy(xva).to(dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr_t), generator=g).to(dev)
        for k in range(0, len(perm), batch):
            idx = perm[k:k + batch]
            loss = loss_fn(model(xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ap = average_precision_score(yva, model(xva_t).cpu().numpy()) if len(np.unique(yva)) > 1 else 0.0
        if ap > best:
            best, best_state = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best


class ImposterFilter:
    """Standardise -> MLP ensemble -> Platt calibration."""

    def __init__(self, n_models=5, device="cuda"):
        self.n_models, self.device = n_models, device

    def fit(self, xtr, ytr, xva, yva):
        self.mu = xtr.mean(0, keepdims=True)
        self.sd = xtr.std(0, keepdims=True) + 1e-6
        ztr, zva = self._z(xtr), self._z(xva)
        self.models, self.val_ap = [], []
        for s in range(self.n_models):
            m, ap = _fit_one(ztr, ytr, zva, yva, seed=s, device=self.device)
            self.models.append(m)
            self.val_ap.append(ap)
        logits = self.logits(xva).mean(1)
        self.platt = LogisticRegression(C=1e3).fit(logits[:, None], yva)
        return self

    def _z(self, x):
        return ((x.astype(np.float32) - self.mu) / self.sd).astype(np.float32)

    @torch.no_grad()
    def logits(self, x) -> np.ndarray:
        """(N, n_models) raw logits."""
        if len(x) == 0:
            return np.zeros((0, self.n_models), np.float32)
        z = torch.from_numpy(self._z(x))
        out = []
        for m in self.models:
            dev = next(m.parameters()).device
            out.append(torch.cat([m(z[k:k + 8192].to(dev)).cpu() for k in range(0, len(z), 8192)]).numpy())
        return np.stack(out, 1)

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated ensemble probability and per-member calibrated probabilities."""
        lg = self.logits(x)
        if len(lg) == 0:
            return np.zeros(0, np.float32), np.zeros((0, self.n_models), np.float32)
        p = self.platt.predict_proba(lg.mean(1)[:, None])[:, 1]
        members = np.stack([self.platt.predict_proba(lg[:, k:k + 1])[:, 1] for k in range(lg.shape[1])], 1)
        return p.astype(np.float32), members.astype(np.float32)

    def state(self):
        return {"mu": self.mu, "sd": self.sd, "platt": self.platt, "n_models": self.n_models,
                "models": [{k: v.cpu() for k, v in m.state_dict().items()} for m in self.models],
                "dim": self.mu.shape[1], "val_ap": self.val_ap}

    @classmethod
    def from_state(cls, st, device="cpu"):
        f = cls(st["n_models"], device)
        f.mu, f.sd, f.platt, f.val_ap = st["mu"], st["sd"], st["platt"], st["val_ap"]
        f.models = []
        for sd in st["models"]:
            m = MLP(st["dim"])
            m.load_state_dict(sd)
            f.models.append(m.to(device).eval())
        return f
