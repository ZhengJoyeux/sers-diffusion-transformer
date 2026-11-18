"""Evaluate existing SERS batches on validation spectra without sampling again."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, peak_widths


METHODS = ("original", "median3", "median3_pca")


def mad(values, axis=None):
    center = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - center), axis=axis)


def ratio(a, b):
    return float(a / b) if b > 0 else None


def describe_peaks(spectra, axis, threshold):
    records = []
    for spectrum in spectra:
        indices, props = find_peaks(spectrum, prominence=threshold)
        widths = peak_widths(spectrum, indices, rel_height=0.5)[0]
        records.append(list(zip(axis[indices], props["prominences"], widths)))
    return records


def within_window(peaks, low, high):
    return max((p for p in peaks if low <= p[0] < high),
               key=lambda p: p[1], default=None)


def peak_protocol(train, axis):
    threshold = max(0.02 * float(np.ptp(train)),
                    6 * float(np.median(mad(np.diff(train, axis=1), axis=1))),
                    np.finfo(float).eps)
    anchors = [p[0] for p in describe_peaks([np.median(train, axis=0)], axis, threshold)[0]]
    train_peaks = describe_peaks(train, axis, threshold)
    windows, excluded = [], np.zeros(axis.size, dtype=bool)
    for i, center in enumerate(anchors):
        low = max(center - 5, (anchors[i - 1] + center) / 2) if i else center - 5
        high = min(center + 5.5, (anchors[i + 1] + center) / 2) if i + 1 < len(anchors) else center + 5.5
        if sum(within_window(p, low, high) is not None for p in train_peaks) >= 6:
            windows.append((center, low, high))
    # Exclude peak shoulders as well as peaks that occur in only a few training spectra.
    for center, _, width in [p for sample in train_peaks for p in sample]:
        excluded |= np.abs(axis - center) <= max(5, 2 * width)
    return threshold, windows, ~excluded


def spectral_audit(generated, reference, axis, protocol):
    threshold, windows, background = protocol
    norm_g = np.linalg.norm(generated, axis=1)
    norm_r = np.linalg.norm(reference, axis=1)
    valid_g, valid_r = norm_g > 0, norm_r > 0
    g_unit = generated[valid_g] / norm_g[valid_g, None]
    r_unit = reference[valid_r] / norm_r[valid_r, None]
    scores = g_unit @ r_unit.T
    metrics = dict(cosine_similarity=float(scores.mean()) if scores.size else None,
                   cosine_pairs=int(scores.size), peak_windows=len(windows),
                   background_cm1=int((background[:-1] & background[1:]).sum()),
                   prominence_threshold=float(threshold))
    g_peaks = describe_peaks(generated, axis, threshold)
    r_peaks = describe_peaks(reference, axis, threshold)
    details, errors, misses, opportunities = [], [], 0, 0
    for center, low, high in windows:
        g = [p for sample in g_peaks if (p := within_window(sample, low, high)) is not None]
        r = [p for sample in r_peaks if (p := within_window(sample, low, high)) is not None]
        target = np.median(r, axis=0) if r else None
        error = np.abs(np.asarray(g) - target) if g and r else np.empty((0, 3))
        # Require the peak in at least half the validation spectra for the miss-rate denominator.
        if len(r) * 2 >= len(reference):
            misses += len(generated) - len(g)
            opportunities += len(generated)
            if error.size:
                errors.append(error)
        details.append(dict(anchor_cm1=center, window_low=low, window_high_exclusive=high,
                            generated_detected=len(g), reference_detected=len(r),
                            reference_peak_cm1=float(target[0]) if r else None,
                            **error_summary(error)))
    metrics.update(peak_misses=misses, peak_opportunities=opportunities,
                   peak_miss_rate=ratio(misses, opportunities),
                   **error_summary(np.concatenate(errors) if errors else np.empty((0, 3))))
    edges = background[:-1] & background[1:]
    for label, spectra, peaks in (("generated", generated, g_peaks),
                                  ("reference", reference, r_peaks)):
        metrics[label + "_background_diff_mad"] = (
            float(np.median(mad(np.diff(spectra, axis=1)[:, edges], axis=1)))
            if edges.any() else None)
        background_peaks = sum(background[np.searchsorted(axis, p[0])]
                               for sample in peaks for p in sample)
        metrics[label + "_background_peaks_per_100cm1"] = ratio(
            100 * background_peaks, len(spectra) * int(edges.sum()))
    ref_mad = metrics["reference_background_diff_mad"]
    metrics["background_diff_mad_ratio"] = (
        ratio(metrics["generated_background_diff_mad"], ref_mad)
        if ref_mad is not None else None)
    return metrics, details


def error_summary(errors):
    return {f"peak_{name}_absolute_error_{label}":
            float(np.quantile(errors[:, col], q)) if len(errors) else None
            for col, name in enumerate(("position_cm1", "prominence", "width_cm1"))
            for label, q in (("median", 0.5), ("p90", 0.9))}


def png_breaks(path, mask, candidate):
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (1920, 768):
            raise ValueError(f"Unexpected PNG format or size: {path}")
        pixels = np.asarray(image.convert("L"))
    # Original coordinates follow the measured mask. The candidate plot spans
    # the same measured axis from x=28 to 1891, with captions outside this crop.
    ink = (pixels[35:720, 28:1892] if candidate else pixels[:, :mask.size][:, mask]) <= 127
    left, right = ink[:, :-1], ink[:, 1:]
    linked = (left & right).any(axis=0)
    linked |= (left[1:] & right[:-1]).any(axis=0)
    linked |= (left[:-1] & right[1:]).any(axis=0)
    transitions = np.flatnonzero(np.diff(np.r_[False, ~linked, False]))
    lengths = transitions[1::2] - transitions[::2]
    return len(lengths), int(lengths.max()) if lengths.size else 0


def table(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--batch", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    sys.path.insert(0, str(args.project.resolve()))
    from core import Bundle, sha, verify
    from metrics import condition_metrics

    verify(args.project, "PACKAGE_SHA256.json")
    bundle = Bundle(str(args.prepared), "cpu")
    names = bundle.state["conditions"]
    axis = np.asarray(bundle.state["axis"])
    if len(names) != 126 or len(set(names)) != 126:
        raise ValueError("Expected exactly 126 prepared conditions")
    inventory = {}
    for batch in args.batch:
        for directory in sorted((batch / "pca").iterdir()):
            if not (directory / "comparison.csv").is_file():
                continue
            condition = directory.name
            if condition not in names or condition in inventory:
                raise ValueError(f"Unexpected or duplicate condition: {directory}")
            inventory[condition] = directory
    if set(inventory) != set(names):
        raise ValueError(f"Missing conditions: {sorted(set(names) - inventory.keys())}")

    rows, peaks, images, verified = [], [], [], set()
    for number, name in enumerate(names, 1):
        directory = inventory[name]
        trial = json.loads((directory / "trial_protocol.json").read_text())
        source = Path(trial["source"])
        if source not in verified:
            verify(source, "manifest.json")
            verified.add(source)
        summary = json.loads((source / "generation_summary.json").read_text())
        protocol = summary["protocol"]
        if (not summary["complete"] or summary["rejected"] or name not in protocol["conditions"]
                or protocol["prepared_digest"] != bundle.digest or protocol["number"] != 50):
            raise ValueError(f"Source generation mismatch: {source}")
        records = [json.loads(line) for line in (source / "quality.jsonl").read_text().splitlines()
                   if line.strip()]
        records = [r for r in records if r["condition"] == name]
        indexed = {r["index"]: r for r in records}
        if len(records) != 50 or set(indexed) != set(range(50)):
            raise ValueError(f"Source inventory mismatch: {name}")
        cid = names.index(name)
        mask = bundle.mask[cid, 0].numpy().astype(bool)
        x = axis[mask]
        if len(x) < 3 or not np.all(np.diff(x) == 1):
            raise ValueError(f"Expected measured axis spaced by 1 cm^-1: {name}")
        train = bundle.raw[cid, :12][:, mask].astype(float)
        ref = bundle.raw[cid, 12:16][:, mask].astype(float)
        mean = bundle.mean[cid, 0].numpy()[mask].astype(float)
        scale = float(bundle.scale[cid].item())
        original, residuals, candidate = [], [], []
        by_method = {m: [] for m in METHODS}
        for index in range(50):
            stem = f"{index:04d}"
            for method, folder, suffix, values in (
                ("original", source / name, "reconstruction", original),
                ("median3_pca", directory / name, "candidate", candidate),
            ):
                path = folder / f"{stem}_{suffix}.npz"
                record = indexed[index]
                if method == "original" and (
                    not record["approved"] or sha(path) != record["reconstruction_sha256"]
                ):
                    raise ValueError(f"Original sample failed verification: {path}")
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(data["axis"], axis) or not np.array_equal(data["mask"], mask):
                        raise ValueError(f"Axis or mask mismatch: {path}")
                    full, z = data["full"][mask], data["z"][mask]
                    if not np.isfinite(full).all() or not np.allclose(
                        full, mean + scale * np.sinh(z), rtol=2e-5, atol=2e-5
                    ):
                        raise ValueError(f"Inverse reconstruction mismatch: {path}")
                    values.append(full)
                    if method == "original":
                        residuals.append(z)
                for kind in ("full", "residual"):
                    png = folder / f"{stem}_{kind}.png"
                    count, longest = png_breaks(png, mask, method == "median3_pca")
                    item = dict(condition=name, method=method, index=index, kind=kind,
                                break_runs=count, longest_break_edges_px=longest, path=str(png))
                    images.append(item)
                    by_method[method].append(item)
        smooth = mean + scale * np.sinh(median_filter(np.stack(residuals), size=(1, 3), mode="nearest"))
        peak_rules = peak_protocol(train, x)
        for method, spectra in zip(METHODS, (np.stack(original), smooth, np.stack(candidate))):
            values, details = spectral_audit(spectra, ref, x, peak_rules)
            row = dict(condition=name, method=method, split="validation",
                       generated_count=len(spectra), reference_count=len(ref),
                       **condition_metrics(spectra, ref, train, bundle.residual.configuration["patch_width"]),
                       **values)
            for kind in ("full", "residual"):
                items = [v for v in by_method[method] if v["kind"] == kind]
                row[f"{kind}_png_audited"] = len(items)
                row[f"{kind}_png_break_rate"] = ratio(sum(v["break_runs"] > 0 for v in items), len(items))
                row[f"{kind}_png_break_runs_mean"] = ratio(sum(v["break_runs"] for v in items), len(items))
                row[f"{kind}_png_longest_break_edges_px"] = max(
                    (v["longest_break_edges_px"] for v in items), default=None)
            rows.append(row)
            peaks.extend(dict(condition=name, method=method, **detail) for detail in details)
        print(f"[{number}/126] {name}", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    table(args.output / "condition_metrics.csv", rows)
    table(args.output / "peak_metrics.csv", peaks)
    table(args.output / "png_audit.csv", images)
    paired = []
    for start in range(0, len(rows), len(METHODS)):
        original_row, smooth_row, pca_row = rows[start:start + len(METHODS)]
        for before, after in ((original_row, smooth_row), (smooth_row, pca_row)):
            entry = dict(condition=before["condition"], before=before["method"], after=after["method"])
            for key, value in before.items():
                if isinstance(value, (int, float)) and isinstance(after[key], (int, float)):
                    entry[key] = after[key] - value
            paired.append(entry)
    table(args.output / "paired_deltas.csv", paired)
    summary = []
    for method in METHODS:
        for key in rows[0]:
            values = [r[key] for r in rows if r["method"] == method and isinstance(r[key], (float, int))]
            if values:
                summary.append(dict(method=method, metric=key, conditions=len(values),
                                    p10=np.quantile(values, .1), median=np.median(values),
                                    p90=np.quantile(values, .9)))
    table(args.output / "metric_summary.csv", summary)
    (args.output / "protocol.json").write_text(json.dumps(dict(
        conditions=126, train_rows=list(range(12)), reference_rows=list(range(12, 16)),
        batches=[str(p) for p in args.batch], prepared_digest=bundle.digest,
        png_rule="dark <=127; adjacent columns must have an 8-neighbor dark edge",
        peak_rule="training median anchors; +/-5 cm^-1, split overlapping windows at midpoints; "
                  "at least 6/12 training detections; miss denominator requires 2/4 validation detections",
        prominence_rule="max(2% of training intensity span, 6 times median training first-difference MAD)",
        width_rule="width at half prominence, in cm^-1",
        median3_png="no artifact in supplied workflow",
        median3_pca="existing experimental postprocessed output",
    ), indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
