"""Run a trained joint model on a user image; outputs mask and scored boxes."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from project import Net, boxes, OUT, SIZE, seed

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--output", default="prediction")
p.add_argument(
    "--mode",
    default=(
        json.loads((OUT / "selection.json").read_text())["recommended_model"]
        if (OUT / "selection.json").exists()
        else "full"
    ),
    choices=["det", "seg", "joint", "fusion", "full", "balanced", "ensemble"],
)
a = p.parse_args()
seed()
im = Image.open(a.image).convert("L")
x = torch.tensor(np.array(im.resize((SIZE, SIZE)), dtype=np.float32) / 255)[None, None]


def forward(mode):
    model = Net(mode)
    model.load_state_dict(
        torch.load(OUT / f"{mode}.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        s, d = model(x)
    return s.sigmoid()[0, 0].numpy(), d.sigmoid()[0, 0].numpy()


if a.mode == "ensemble":
    config = json.loads((OUT / "ensemble.json").read_text())["selected"]
    expert, _ = forward("seg")
    s, d = forward(config["cooperative_mode"])
    s = config["seg_expert_weight"] * expert + (1 - config["seg_expert_weight"]) * s
    th = config["seg_threshold"]
    mask = s > th
    d = d * mask
    bm = d > config["det_threshold"]
else:
    s, d = forward(a.mode)
    th = json.loads((OUT / "results.json").read_text())[a.mode]["threshold"]
    mask = s > th
    bm = d > th
    if a.mode in ["full", "balanced"]:
        bm &= mask
folder = Path(a.output)
folder.mkdir(parents=True, exist_ok=True)
if a.mode != "det":
    Image.fromarray((mask * 255).astype("uint8")).resize(
        im.size, Image.Resampling.NEAREST
    ).save(folder / "mask.png")
bs = []
for l, t, r, b in ([] if a.mode == "seg" else boxes(bm)):
    bs.append(
        {
            "box_xyxy": [
                l * im.width / SIZE,
                t * im.height / SIZE,
                r * im.width / SIZE,
                b * im.height / SIZE,
            ],
            "score": float(d[t:b, l:r].max()),
            "class": "defect",
        }
    )
if a.mode != "seg":
    (folder / "detections.json").write_text(json.dumps(bs, indent=2))
print(json.dumps({"detections": len(bs), "output": str(folder)}))
