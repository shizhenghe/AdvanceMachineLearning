"""Check the public inference CLI against saved test predictions, including coordinates."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from project import OUT, boxes

root = Path(__file__).parent
data = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/magnetic-tiles")
files = sorted(p for p in data.rglob("*.jpg") if p.parent.name == "Imgs")
pred = np.load(OUT / "ensemble_predictions.npz")
config = json.loads((OUT / "ensemble.json").read_text())["selected"]
# Choose the first saved test case with a predicted target; no metric or tuning involved.
idx = next(i for i, m in enumerate(pred["det"]) if boxes(m > config["det_threshold"]))
source = files[int(pred["ids"][idx])]
with tempfile.TemporaryDirectory(prefix="defect-test-") as out:
    subprocess.run(
        [
            sys.executable,
            str(root / "src/infer.py"),
            str(source),
            "--mode",
            "ensemble",
            "--output",
            out,
        ],
        check=True,
    )
    original = Image.open(source)
    actual = Image.open(Path(out) / "mask.png")
    expected = Image.fromarray(
        ((pred["seg"][idx] > config["seg_threshold"]) * 255).astype("uint8")
    ).resize(original.size, Image.Resampling.NEAREST)
    assert actual.size == original.size
    assert (np.array(actual) == np.array(expected)).mean() > 0.999
    detections = json.loads((Path(out) / "detections.json").read_text())
    expected_boxes = boxes(pred["det"][idx] > config["det_threshold"])
    assert len(detections) == len(expected_boxes) > 0
    for d, (l, t, r, b) in zip(detections, expected_boxes):
        assert np.allclose(
            d["box_xyxy"],
            [
                l * original.width / 64,
                t * original.height / 64,
                r * original.width / 64,
                b * original.height / 64,
            ],
        )
        assert 0 <= d["score"] <= 1
    print(
        "Inference matches saved test masks, target count and original-image box coordinates"
    )
