"""Merge the independently run sixth ablation and select using validation only."""

import csv
import hashlib
import json
import shutil
from pathlib import Path
from project import OUT


def main():
    extra = OUT / "balanced_run"
    result = json.loads((OUT / "results.json").read_text())
    history = json.loads((OUT / "history.json").read_text())
    run = json.loads((OUT / "run.json").read_text())
    if extra.exists():
        result.update(json.loads((extra / "results.json").read_text()))
        history.update(json.loads((extra / "history.json").read_text()))
        for name in ["balanced.pt", "balanced_predictions.npz"]:
            source = extra / name
            if source.exists():
                target = OUT / name
                shutil.copy2(source, target)
                assert (
                    hashlib.sha256(source.read_bytes()).digest()
                    == hashlib.sha256(target.read_bytes()).digest()
                )
                source.unlink()
        extra_run = json.loads((extra / "run.json").read_text())
        if not run.get("balanced_merged"):
            run["seconds"] += extra_run["seconds"]
            run["balanced_merged"] = True
        run["balanced_threads"] = 1
    result = {
        m: result[m]
        for m in ["det", "seg", "joint", "fusion", "full", "balanced", "ensemble"]
        if m in result
    }
    run["modes"] = list(result)
    scores = {
        m: max(e["val"]["IoU"] + e["val"]["mAP50"] for e in h)
        for m, h in history.items()
        if m not in ["det", "seg"]
    }
    if (OUT / "ensemble.json").exists():
        scores["ensemble"] = json.loads((OUT / "ensemble.json").read_text())[
            "selected"
        ]["objective"]
    chosen = max(scores, key=scores.get)
    selection = {
        "recommended_model": chosen,
        "policy": "maximum validation checkpoint IoU + mAP50; no test metric used",
        "validation_scores": scores,
    }
    for name, obj in [
        ("results.json", result),
        ("history.json", history),
        ("run.json", run),
        ("selection.json", selection),
    ]:
        (OUT / name).write_text(json.dumps(obj, indent=2))
    with (OUT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "mode",
                "mAP50",
                "mAP50_95",
                "IoU",
                "mIoU",
                "ground_truth_boxes",
                "threshold",
                "det_threshold",
            ],
        )
        writer.writeheader()
        for m, q in result.items():
            writer.writerow({"mode": m, **q})
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    for name, source in [
        ("training.log", "/tmp/defect-training.log"),
        ("balanced.log", "/tmp/defect-balanced.log"),
        ("vae.log", "/tmp/defect-vae.log"),
    ]:
        if Path(source).exists():
            shutil.copy2(source, logs / name)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
