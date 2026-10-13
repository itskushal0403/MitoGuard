"""Precompute MitoGuard v2 outputs on held-out MIDOG++ test regions for the static website (site/).

For each domain, a 1536 px window from an official *test-split* case (ground truth = pathologist consensus):
overlay images (detections only / with rejected imposters / ground truth) + a JSON record.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.data import load_annotations, load_cases, image_path  # noqa: E402
from mitoguard.pipeline_v2 import MitoGuardV2  # noqa: E402
from mitoguard.tiling import read_region  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "site/assets/regions"
S = 1536
DOMAINS = ["HBC@S360", "HBC@CS2", "CCMCT@CS2", "CL@3DH", "CLC@3DH", "CSTS@3DH", "HMEL@XR", "HNET@XR"]
NAMES = {"HBC": "Human breast carcinoma", "CCMCT": "Canine mast cell tumour", "CL": "Canine lymphoma",
         "CLC": "Canine lung carcinoma", "CSTS": "Canine soft-tissue sarcoma", "HMEL": "Human melanoma",
         "HNET": "Human neuroendocrine tumour"}
SCAN = {"S360": "Hamamatsu S360", "CS2": "Aperio CS2", "3DH": "3D Histech", "XR": "Hamamatsu XR"}
os.makedirs(OUT, exist_ok=True)
m = MitoGuardV2("checkpoints/mitoguard_v2_bundle.pt")
cases, ann = load_cases(), load_annotations()
records = []
for dom in DOMAINS:
    cand = cases[(cases.domain == dom) & (cases.official_split == "test")]
    cnt = ann[ann.case_id.isin(cand.case_id) & (ann.label == 1)].groupby("case_id").size()
    cid = int(cnt.sort_values().index[int(len(cnt) * 0.6)])
    a = ann[ann.case_id == cid]
    w, h = cases.loc[cid, ["width", "height"]]
    best, bxy = -1, (0, 0)
    for y in range(0, h - S, 128):
        for x in range(0, w - S, 128):
            inside = (a.x >= x) & (a.x < x + S) & (a.y >= y) & (a.y < y + S)
            n = (inside & (a.label == 1)).sum() + 0.3 * (inside & (a.label == 2)).sum()
            if n > best:
                best, bxy = n, (x, y)
    x0, y0 = bxy
    img = read_region(image_path(cid))[y0:y0 + S, x0:x0 + S].copy()
    r = m(img)
    gt = a[(a.x >= x0) & (a.x < x0 + S) & (a.y >= y0) & (a.y < y0 + S)]
    slug = dom.replace("@", "_").lower()

    def save(name, im):
        cv2.imwrite(os.path.join(OUT, f"{slug}_{name}.jpg"), cv2.resize(im, (1024, 1024), interpolation=cv2.INTER_AREA)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 86])

    save("raw", img)
    det, rej, truth = img.copy(), img.copy(), img.copy()
    for (x, y), keep, py in zip(r.xy, r.kept, r.p_yolo):
        if keep:
            for im in (det, rej):
                cv2.circle(im, (int(x), int(y)), 36, (20, 200, 70), 6)
        elif py >= 0.3:
            cv2.circle(rej, (int(x), int(y)), 36, (230, 45, 45), 5)
    for row in gt.itertuples():
        col = (255, 215, 0) if row.label == 1 else (150, 150, 150)
        cv2.circle(truth, (int(row.x - x0), int(row.y - y0)), 36, col, 6 if row.label == 1 else 3)
    save("det", det)
    save("rej", rej)
    save("gt", truth)
    lo, hi = r.interval
    rec = {"id": slug, "tumour": NAMES[dom.split("@")[0]], "scanner": SCAN[dom.split("@")[1]], "case": cid,
           "count": r.count, "lo": round(float(lo)), "hi": round(float(hi)), "category": r.category,
           "ambiguous": bool(r.ambiguous), "gt": int((gt.label == 1).sum()), "imposters_gt": int((gt.label == 2).sum()),
           "rejected": int((~r.kept & (r.p_yolo >= 0.3)).sum()), "candidates": int(len(r.xy))}
    records.append(rec)
    print(rec, flush=True)
json.dump(records, open(os.path.join(OUT, "regions.json"), "w"), indent=1)
