import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter

from core import Bundle, sha, verify
from metrics import condition_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--generated", required=True, type=Path)
parser.add_argument("--prepared", required=True)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--all", action="store_true")
args = parser.parse_args()

source = args.generated
verify(source, "manifest.json")
summary = json.loads((source / "generation_summary.json").read_text())
bundle = Bundle(args.prepared, "cpu")
protocol = summary["protocol"]
if not summary["complete"] or protocol["prepared_digest"] != bundle.digest:
    raise SystemExit("Expected a complete generation batch from this preparation")

names = bundle.state["conditions"]
conditions = protocol["conditions"]
if not conditions or len(conditions) != len(set(conditions)):
    raise SystemExit("Invalid condition inventory")
if args.all and (len(names) != 126 or set(conditions) != set(names)):
    raise SystemExit(f"--all requires all 126 conditions; found {len(conditions)}")

count = protocol["number"]
records = {
    (r["condition"], r["index"]): r
    for r in map(json.loads, (source / "quality.jsonl").read_text().splitlines())
}
if count != 50 or len(records) != len(conditions) * count:
    raise SystemExit("Expected 50 samples per condition")

axis = np.asarray(bundle.state["axis"])
patch_width = bundle.residual.configuration["patch_width"]
strength = 0.5
out = args.output
out.mkdir(parents=True, exist_ok=False)
reports = []


def png(path, x, y, low, high, label):
    aa = 3
    image = Image.new("L", (1920 * aa, 768 * aa), 255)
    draw = ImageDraw.Draw(image)
    xx = 28 + (x - x[0]) / (x[-1] - x[0]) * 1863
    yy = 700 - (y - low) / (high - low) * 650
    points = np.rint(np.column_stack((xx, yy)) * aa).astype(int)
    draw.line([tuple(p) for p in points], fill=0, width=2 * aa)
    image = image.resize((1920, 768), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.text((28, 16), label, fill=0)
    draw.text((28, 730), f"{x[0]:g} cm^-1", fill=0)
    draw.text((1891, 730), f"{x[-1]:g} cm^-1", fill=0, anchor="ra")
    image.save(path)


for condition in conditions:
    cid = names.index(condition)
    mask = bundle.mask[cid, 0].numpy().astype(bool)
    x = axis[mask]
    if len(x) < 3 or not np.all(np.diff(x) == 1):
        raise SystemExit(f"Unexpected measured axis: {condition}")
    train = bundle.raw[cid, :12][:, mask].astype(np.float64)
    ref = bundle.raw[cid, 12:16][:, mask].astype(np.float64)
    mean = bundle.mean[cid, 0].numpy()[mask].astype(np.float64)
    scale = float(bundle.scale[cid].item())

    centered = train - train.mean(axis=0)
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    tolerance = singular[0] * max(centered.shape) * np.finfo(centered.dtype).eps
    rank = min(4, int(np.count_nonzero(singular > tolerance)))
    basis = basis[:rank]
    joint_scores = centered @ basis.T
    rng = np.random.default_rng(2026 + cid)
    indices = np.concatenate(
        [rng.permutation(len(train)) for _ in range((count + 11) // 12)]
    )[:count]
    perturbation = strength * (joint_scores[indices] @ basis)
    perturbation -= perturbation.mean(axis=0)

    original, residues = [], []
    for i in range(count):
        path = source / condition / f"{i:04d}_reconstruction.npz"
        record = records[condition, i]
        if not record["approved"] or sha(path) != record["reconstruction_sha256"]:
            raise SystemExit(f"Source sample failed verification: {path}")
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["mask"], mask) or not np.array_equal(data["axis"], axis):
                raise SystemExit(f"Axis or mask mismatch: {path}")
            original.append(data["full"][mask])
            residues.append(data["z"][mask])

    original = np.stack(original).astype(np.float64)
    z = np.stack(residues).astype(np.float64)
    if not np.allclose(original, mean + scale * np.sinh(z), rtol=2e-5, atol=2e-5):
        raise SystemExit(f"Original reconstruction mismatch: {condition}")

    smoothed = mean + scale * np.sinh(median_filter(z, size=(1, 3), mode="nearest"))
    augmented = smoothed + perturbation
    augmented_z = np.arcsinh((augmented - mean) / scale)
    if not np.isfinite(augmented_z).all():
        raise SystemExit(f"Nonfinite candidate: {condition}")

    for label, values in (
        ("original", original), ("median3", smoothed), ("median3_pca", augmented)
    ):
        metrics = condition_metrics(values, ref, train, patch_width)
        reports.append(dict(condition=condition, method=label, pca_rank=rank, **metrics))

    # Export auditable candidate values and connected display PNGs.
    folder = out / condition
    folder.mkdir()
    bound = max(1e-6, 1.05 * float(np.abs(augmented_z).max()))
    lo, hi = float(augmented.min()), float(augmented.max())
    margin = max(1e-6, 0.05 * (hi - lo))
    for i, (spectrum, residual) in enumerate(zip(augmented, augmented_z)):
        full_grid = np.zeros(axis.shape)
        residual_grid = np.zeros(axis.shape)
        full_grid[mask], residual_grid[mask] = spectrum, residual
        np.savez_compressed(folder / f"{i:04d}_candidate.npz",
                            axis=axis, mask=mask, full=full_grid, z=residual_grid)
        png(folder / f"{i:04d}_full.png", x, spectrum, lo - margin, hi + margin,
            "DiT + median3 + PCA | candidate")
        png(folder / f"{i:04d}_residual.png", x, residual, -bound, bound,
            "Asinh residual | candidate")

with (out / "comparison.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
    writer.writeheader()
    writer.writerows(reports)
(out / "trial_protocol.json").write_text(json.dumps({
    "source": str(source), "pca": "condition-specific, training rows 0-11",
    "strength": strength, "seed_base": 2026, "validation_rows": [12, 13, 14, 15],
    "candidate_status": "experimental postprocessing",
}, indent=2) + "\n")

for method in ("original", "median3", "median3_pca"):
    group = [row for row in reports if row["method"] == method]
    diversity = np.median([row["pairwise_diversity_ratio"] for row in group])
    derivative = np.median([row["mean_derivative_pearson"] for row in group])
    width = np.median([row["peak_width_iqr_ratio"] for row in group])
    print(f"{method}: conditions={len(group)} diversity={diversity:.3f} "
          f"derivative_r={derivative:.4f} peak_width_IQR_ratio={width:.3f}")
print(out / "comparison.csv")
