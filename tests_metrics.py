"""Metric checks with exact boxes and a false positive on a normal image."""

import sys

sys.path.insert(0, "src")
import numpy as np
from project import evaluate

m = np.zeros((2, 64, 64), dtype=np.float32)
m[0, 10:20, 15:25] = 1
q = evaluate(m, m, m)
assert (
    q["mAP50"] > 0.999 and q["mAP50_95"] > 0.999 and q["IoU"] == 1 and q["mIoU"] == 1
), q
bad = m.copy()
bad[1, 5:15, 5:15] = 1
q = evaluate(bad, bad, m)
assert q["IoU"] < 1 and q["mAP50"] < 1, q
q = evaluate(np.zeros_like(m), np.zeros_like(m), m)
assert q["mAP50"] == 0 and q["IoU"] == 0, q
print("3 metric scenarios passed")
