"""Reproducible CPU benchmark: binary defect detection, segmentation and VAE."""

import argparse, json, random, time, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
SIZE = 64


def seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.set_num_threads(4)


def boxes(mask):
    lab, n = ndimage.label(mask)
    return [
        [s[1].start, s[0].start, s[1].stop, s[0].stop]
        for s in ndimage.find_objects(lab)
        if s is not None and (s[1].stop - s[1].start) * (s[0].stop - s[0].start) >= 2
    ]


def load(data):
    files = sorted(p for p in Path(data).rglob("*.jpg") if p.parent.name == "Imgs")
    xs = []
    ys = []
    cats = []
    names = []
    for p in files:
        q = p.with_suffix(".png")
        if not q.exists():
            raise FileNotFoundError(q)
        xs.append(
            np.array(Image.open(p).convert("L").resize((SIZE, SIZE)), dtype=np.float32)
            / 255
        )
        ys.append(
            (
                np.array(
                    Image.open(q)
                    .convert("L")
                    .resize((SIZE, SIZE), Image.Resampling.NEAREST)
                )
                > 0
            ).astype(np.float32)
        )
        cats.append(p.parent.parent.name)
        names.append(str(p.relative_to(data)))
    ids = np.arange(len(files))
    tr, te = train_test_split(ids, test_size=0.2, random_state=42, stratify=cats)
    tr, va = train_test_split(
        tr, test_size=0.2, random_state=42, stratify=np.array(cats)[tr]
    )
    OUT.mkdir(exist_ok=True)
    (OUT / "split.json").write_text(
        json.dumps(
            {
                k: [names[i] for i in v]
                for k, v in [("train", tr), ("val", va), ("test", te)]
            },
            indent=2,
        )
    )
    x = torch.tensor(np.array(xs))[:, None]
    y = torch.tensor(np.array(ys))[:, None]
    rect = torch.zeros_like(y)
    for i, m in enumerate(ys):
        for a, b, c, d in boxes(m):
            rect[i, 0, b:d, a:c] = 1
    return x, y, rect, tr, va, te, names


class Block(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(
            nn.Conv2d(a, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.GroupNorm(4, b),
            nn.ReLU(),
        )


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.e = Block(1, 12)
        self.b = Block(12, 24)
        self.d = Block(36, 12)
        self.seg = nn.Conv2d(12, 1, 1)
        self.det = nn.Conv2d(12, 1, 1)
        self.fuse = nn.Conv2d(13, 1, 1)

    def forward(self, x):
        x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 0.05)
        a = self.e(x)
        z = self.d(
            torch.cat(
                [
                    a,
                    F.interpolate(
                        self.b(F.max_pool2d(a, 2)),
                        size=a.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                ],
                1,
            )
        )
        s = self.seg(z)
        d = self.det(z)
        if self.mode in ["fusion", "full", "balanced"]:
            d = d + self.fuse(torch.cat([z, s.sigmoid()], 1))
        return s, d


def loss(p, y):
    return (
        F.binary_cross_entropy_with_logits(p, y, pos_weight=torch.tensor(5.0))
        + 1
        - (2 * (p.sigmoid() * y).sum() + 1) / (p.sigmoid().sum() + y.sum() + 1)
    )


def ioubox(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    z = x * y
    return z / (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - z + 1e-9
    )


def evaluate(s, d, y, threshold=0.5, refine=False):
    pred = s > threshold
    truth = y > 0
    inter = (pred & truth).sum()
    union = (pred | truth).sum()
    fg = float(inter / max(union, 1))
    bg = float(((~pred) & (~truth)).sum() / max(((~pred) | (~truth)).sum(), 1))
    gt = [boxes(m) for m in truth]
    dets = []
    for i, m in enumerate(d):
        bm = m > threshold
        if refine:
            bm = bm & (s[i] > threshold)
        for b in boxes(bm):
            a, c, e, f = b
            score = float(m[c:f, a:e].max())
            dets.append((score, i, b))
    dets.sort(reverse=True)
    aps = []
    for th in np.arange(0.5, 0.96, 0.05):
        used = [set() for _ in gt]
        tp = []
        for _, i, b in dets:
            scores = [
                ioubox(b, g) if j not in used[i] else -1 for j, g in enumerate(gt[i])
            ]
            j = int(np.argmax(scores)) if scores else -1
            ok = j >= 0 and scores[j] >= th
            tp.append(ok)
            if ok:
                used[i].add(j)
        t = np.cumsum(tp)
        fp = np.cumsum(1 - np.array(tp, dtype=float))
        recall = t / max(sum(map(len, gt)), 1)
        precision = t / (t + fp + 1e-9)
        aps.append(
            float(
                np.mean(
                    [
                        precision[recall >= r].max() if np.any(recall >= r) else 0
                        for r in np.linspace(0, 1, 101)
                    ]
                )
            )
        )
    return {
        "mAP50": aps[0],
        "mAP50_95": float(np.mean(aps)),
        "IoU": fg,
        "mIoU": (fg + bg) / 2,
        "ground_truth_boxes": sum(map(len, gt)),
    }


def predict(model, x):
    model.eval()
    with torch.no_grad():
        ps = [model(b) for b in x.split(32)]
    return tuple(
        torch.cat([p[j].sigmoid() for p in ps])[:, 0].numpy() for j in range(2)
    )


def train(data, epochs, modes=None):
    seed()
    x, y, r, tr, va, te, names = load(data)
    results = (
        json.loads((OUT / "results.json").read_text())
        if modes and (OUT / "results.json").exists()
        else {}
    )
    hist = (
        json.loads((OUT / "history.json").read_text())
        if modes and (OUT / "history.json").exists()
        else {}
    )
    start = time.time()
    for mode in modes or ["det", "seg", "joint", "fusion", "full", "balanced"]:
        seed()
        net = Net(mode)
        opt = torch.optim.Adam(net.parameters(), lr=0.002)
        best = -1
        history = []
        for ep in range(epochs):
            net.train()
            total = 0
            weights = (y[tr].sum((1, 2, 3)) > 0).float() * 3 + 1
            order = torch.multinomial(weights, len(tr), replacement=True)
            for ids in torch.tensor(tr)[order].split(32):
                xx = x[ids]
                yy = y[ids]
                rr = r[ids]
                if random.random() < 0.5:
                    xx = xx.flip(-1)
                    yy = yy.flip(-1)
                    rr = rr.flip(-1)
                s, d = net(xx)
                ls = loss(s, yy)
                ld = loss(d, rr)
                l = (
                    ld
                    if mode == "det"
                    else (
                        ls
                        if mode == "seg"
                        else (4 * ls + ld if mode == "balanced" else ls + ld)
                    )
                )
                if mode in ["full", "balanced"]:
                    l = l + 0.2 * F.relu(s.sigmoid() - d.sigmoid()).mean()
                opt.zero_grad()
                l.backward()
                opt.step()
                total += l.item()
            vs, vd = predict(net, x[va])
            v = max(
                [
                    evaluate(vs, vd, y[va, 0].numpy(), t, mode in ["full", "balanced"])
                    for t in [0.25, 0.35, 0.5, 0.65]
                ],
                key=lambda q: (
                    q["mAP50"]
                    if mode == "det"
                    else q["IoU"] if mode == "seg" else q["mAP50"] + q["IoU"]
                ),
            )
            score = (
                v["mAP50"]
                if mode == "det"
                else v["IoU"] if mode == "seg" else v["IoU"] + v["mAP50"]
            )
            history.append({"epoch": ep + 1, "loss": total, "val": v})
            if score > best:
                best = score
                torch.save(net.state_dict(), OUT / f"{mode}.pt")
            print(
                mode,
                ep + 1,
                round(total, 3),
                {k: round(v[k], 3) for k in ["mAP50", "IoU"]},
                flush=True,
            )
        net.load_state_dict(torch.load(OUT / f"{mode}.pt", weights_only=True))
        vs, vd = predict(net, x[va])
        ts, td = predict(net, x[te])
        # Calibrate the operating threshold on validation only, then freeze it for test.
        opts = []
        for threshold in [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75]:
            q = evaluate(
                vs, vd, y[va, 0].numpy(), threshold, mode in ["full", "balanced"]
            )
            score = (
                q["mAP50"]
                if mode == "det"
                else q["IoU"] if mode == "seg" else q["mAP50"] + q["IoU"]
            )
            opts.append((score, threshold))
        threshold = max(opts)[1]
        q = evaluate(ts, td, y[te, 0].numpy(), threshold, mode in ["full", "balanced"])
        q["threshold"] = threshold
        if mode == "det":
            q["IoU"] = None
            q["mIoU"] = None
        if mode == "seg":
            q["mAP50"] = None
            q["mAP50_95"] = None
        results[mode] = q
        hist[mode] = history
        np.savez_compressed(OUT / f"{mode}_predictions.npz", seg=ts, det=td, ids=te)
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        (OUT / "history.json").write_text(json.dumps(hist))
    (OUT / "run.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "size": SIZE,
                "epochs": epochs,
                "images": len(x),
                "train": len(tr),
                "val": len(va),
                "test": len(te),
                "seconds": time.time()
                - start
                + (
                    json.loads((OUT / "run.json").read_text()).get("seconds", 0)
                    if modes and (OUT / "run.json").exists()
                    else 0
                ),
                "modes": list(results),
                "torch": torch.__version__,
            },
            indent=2,
        )
    )

    cooperative = {
        m: max(e["val"]["IoU"] + e["val"]["mAP50"] for e in h)
        for m, h in hist.items()
        if m not in ["det", "seg"]
    }
    if cooperative:
        (OUT / "selection.json").write_text(
            json.dumps(
                {
                    "recommended_model": max(cooperative, key=cooperative.get),
                    "policy": "maximum validation checkpoint IoU + mAP50; no test metric used",
                    "validation_scores": cooperative,
                },
                indent=2,
            )
        )


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(32 * 16 * 16, 32)
        self.lv = nn.Linear(32 * 16 * 16, 32)
        self.fc = nn.Linear(32, 32 * 16 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def decode(self, z):
        return self.dec(self.fc(z).reshape(-1, 32, 16, 16))

    def forward(self, x):
        h = self.enc(x)
        m = self.mu(h)
        v = self.lv(h).clamp(-8, 8)
        return self.decode(m + torch.randn_like(m) * (v * 0.5).exp()), m, v


def generate(data, epochs=60):
    seed()
    x, y, r, tr, va, te, names = load(data)
    # Defect-centered crops, training partition only; no held-out images in VAE.
    patches = []
    sources = []
    for i in tr:
        for a, b, c, d in boxes(y[i, 0].numpy()):
            pad = 6
            a = max(0, a - pad)
            b = max(0, b - pad)
            c = min(SIZE, c + pad)
            d = min(SIZE, d + pad)
            patches.append(
                F.interpolate(
                    x[i : i + 1, :, b:d, a:c],
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
            sources.append(names[i])
    z = torch.stack(patches)
    net = VAE()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    history = []
    for ep in range(epochs):
        losses = []
        for ids in torch.randperm(len(z)).split(32):
            out, m, v = net(z[ids])
            l = (
                F.mse_loss(out, z[ids])
                + 0.0002 * (-0.5 * (1 + v - m.square() - v.exp())).mean()
            )
            opt.zero_grad()
            l.backward()
            opt.step()
            losses.append(l.item())
        history.append(float(np.mean(losses)))
        if ep % 10 == 0:
            print("VAE", ep, history[-1], flush=True)
    net.eval()
    folder = OUT / "generated"
    folder.mkdir(exist_ok=True)
    with torch.no_grad():
        h = net.enc(z)
        mus = net.mu(h)
        std = mus.std(0).clamp_min(0.1)
        mean = mus.mean(0)
        samples = net.decode(mean + torch.randn(120, 32) * std)
    for i, a in enumerate(samples[:, 0]):
        Image.fromarray((a.numpy() * 255).astype("uint8")).save(
            folder / f"defect_{i:03d}.png"
        )
    torch.save(net.state_dict(), OUT / "vae.pt")
    (OUT / "generation.json").write_text(
        json.dumps(
            {
                "method": "defect-crop VAE, diagonal aggregate posterior sampling",
                "count": 120,
                "epochs": epochs,
                "training_crops": len(z),
                "sources": sources,
                "loss": history,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["train", "generate"])
    p.add_argument("--data", default="/workspace/data/magnetic-tiles")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["det", "seg", "joint", "fusion", "full", "balanced"],
    )
    a = p.parse_args()
    (
        train(a.data, a.epochs, a.modes)
        if a.task == "train"
        else generate(a.data, a.epochs)
    )
