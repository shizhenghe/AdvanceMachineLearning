"""Validation-calibrated two-expert cooperation; no retraining or test tuning."""

import json
import numpy as np
import torch
from project import OUT, Net, seed, load, predict, evaluate


def score(s, d, y, st, dt):
    # Segmentation suppresses unsupported detection regions; thresholds are distinct.
    guided = d * (s > st)
    q = evaluate(s, guided, y, dt)
    z = evaluate(s, d, y, st)
    q["IoU"] = z["IoU"]
    q["mIoU"] = z["mIoU"]
    q["threshold"] = st
    return q, guided


def main(data):
    seed()
    x, y, r, tr, va, te, names = load(data)

    def infer(mode, ids):
        net = Net(mode)
        net.load_state_dict(torch.load(OUT / f"{mode}.pt", weights_only=True))
        return predict(net, x[ids])

    vs, _ = infer("seg", va)
    # The added expert and validation search cost are explicitly disclosed.
    candidates = []
    for mode in ["full"]:
        cs, cd = infer(mode, va)
        best = (-1, None, None, None)
        for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
            s = alpha * vs + (1 - alpha) * cs
            for st in [0.35, 0.45, 0.5, 0.55, 0.65, 0.75]:
                q = evaluate(s, cd, y[va, 0].numpy(), st)
                if q["IoU"] > best[0]:
                    best = (q["IoU"], alpha, st, s)
        _, alpha, st, s = best
        for dt in [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75]:
            q, _ = score(s, cd, y[va, 0].numpy(), st, dt)
            candidates.append(
                {
                    "cooperative_mode": mode,
                    "seg_expert_weight": alpha,
                    "seg_threshold": st,
                    "det_threshold": dt,
                    "validation": q,
                    "objective": q["IoU"] + q["mAP50"],
                }
            )
    config = max(candidates, key=lambda c: c["objective"])
    # Freeze all choices before computing the sole test result for this scheme.
    ts, _ = infer("seg", te)
    cs, cd = infer(config["cooperative_mode"], te)
    s = config["seg_expert_weight"] * ts + (1 - config["seg_expert_weight"]) * cs
    q, d = score(
        s, cd, y[te, 0].numpy(), config["seg_threshold"], config["det_threshold"]
    )
    q["det_threshold"] = config["det_threshold"]
    (OUT / "ensemble.json").write_text(
        json.dumps(
            {
                "selected": config,
                "validation_candidates": candidates,
                "note": "two experts; extra parameters/compute relative to single-backbone ablations",
            },
            indent=2,
        )
    )
    results = json.loads((OUT / "results.json").read_text())
    results["ensemble"] = q
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    np.savez_compressed(OUT / "ensemble_predictions.npz", seg=s, det=d, ids=te)
    hist = json.loads((OUT / "history.json").read_text())
    selection = {
        "validation_scores": {
            m: max(e["val"]["IoU"] + e["val"]["mAP50"] for e in hh)
            for m, hh in hist.items()
            if m not in ["det", "seg"]
        },
        "policy": "maximum validation IoU + mAP50; no test metric used",
    }
    selection["validation_scores"]["ensemble"] = config["objective"]
    selection["recommended_model"] = max(
        selection["validation_scores"], key=selection["validation_scores"].get
    )
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2))
    print(
        json.dumps(
            {
                "configuration": config,
                "test": q,
                "recommended": selection["recommended_model"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/magnetic-tiles")
