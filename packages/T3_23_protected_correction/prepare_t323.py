"""Create an isolated, verified snapshot. Original code/data/weights are read-only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parent
DEFAULT_BASELINE = "/home/wqzheng/project_transformer/outputs/t3_18_random48_train_h8_l6_d128_real_generated_20261008_145007_1824752"
DEFAULT_PREVIOUS = "/home/wqzheng/project_transformer/outputs/t3_22_reference_joint_train_20261009_125205_2457417"


def text_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def verify_code(directory, expectations):
    for name, expected in expectations.items():
        if text_hash(Path(directory)/name) != expected:
            raise RuntimeError(f"Source differs from verified snapshot: {Path(directory)/name}; nothing overwritten")


def data_inventory(project):
    result = {}
    for source in ("real", "generated"):
        root = (Path(project)/"data"/source).resolve(strict=True)
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"})
        if len(files) != 126:
            raise RuntimeError(f"Expected 126 {source} source files, found {len(files)}")
        print(f"Fingerprinting {source}: {len(files)} files", flush=True)
        result[source] = {"root": str(root), "files": {str(p.relative_to(root)): file_hash(p) for p in files}}
    return result


def assert_unchanged(manifest, check_data=False):
    verify_code(manifest["project"], manifest["project_sources"])
    verify_code(manifest["encoder_package"], manifest["encoder_sources"])
    verify_code(PACKAGE/"runtime_project", manifest["project_sources"])
    if file_hash(manifest["baseline_checkpoint"]) != manifest["baseline_checkpoint_sha256"]:
        raise RuntimeError("The original 1-D checkpoint changed")
    if check_data and data_inventory(manifest["project"]) != manifest["data_inventory"]:
        raise RuntimeError("Input files or data links changed since preparation; stop this comparison")
    for name,sha in manifest["previous_checkpoint_hashes"].items():
        if not Path(name).is_file() or file_hash(name)!=sha:raise RuntimeError("An existing T3.22 checkpoint changed")
    print("Original 1-D code/checkpoint and isolated snapshot: UNCHANGED", flush=True)


def prepare(project, encoder_package, baseline_run,previous_run=DEFAULT_PREVIOUS):
    project = Path(project).expanduser().resolve(strict=True)
    encoder_package = Path(encoder_package).expanduser().resolve(strict=True)
    baseline_run = Path(baseline_run).expanduser().resolve(strict=True)
    previous_run=Path(previous_run).expanduser().resolve(strict=True)
    historical=json.loads((PACKAGE/"previous_validation_reference.json").read_text())
    previous_checkpoint_hashes={}
    for variant in ("reference","no_reference"):
        summary=json.loads((previous_run/variant/"run_summary.json").read_text())
        expected=historical["validation_metrics"][variant]
        for group in ("all","ternary","soil_ternary","water_ternary"):
            for key in ("head_correct","profile_correct"):
                if summary["recommended_validation_metrics"][group][key]!=expected[group][key]:
                    raise RuntimeError("Previous T3.22 result differs from uploaded diagnosis")
        for name in ("recommended.pt","best_candidate.pt"):
            path=previous_run/variant/"checkpoints"/name
            if not path.is_file():raise FileNotFoundError(path)
            previous_checkpoint_hashes[str(path)]=file_hash(path)
    project_sources = json.loads((PACKAGE/"expected_project_sources.json").read_text())
    encoder_sources = json.loads((PACKAGE/"expected_encoder_sources.json").read_text())
    verify_code(project, project_sources)
    verify_code(encoder_package, encoder_sources)
    reference = json.loads((PACKAGE/"reference_random48.json").read_text())
    current = json.loads((baseline_run/"run_summary.json").read_text())
    if current["model_config"] != reference["model_config"]:
        raise RuntimeError("Baseline model configuration is not the current 8-head/6-layer/128-dim model")
    for key, value in reference["training_config"].items():
        if key != "output_directory" and current["training_config"].get(key) != value:
            raise RuntimeError(f"Baseline training configuration differs: {key}")
    if current["data_config"]["train_source_counts"] != {"real": 1512, "generated": 6048}:
        raise RuntimeError("Baseline is not the 12 real + 48 generated per-condition experiment")
    checkpoint = baseline_run/"checkpoints"/"best_ordinal.pt"
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    state_path = PACKAGE/"prepared_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if (state["project"], state["encoder_package"], state["baseline_run"]) != (str(project),str(encoder_package),str(baseline_run)):
            raise RuntimeError("This package is already prepared for a different source; use a fresh package directory")
        if state["previous_run"]!=str(previous_run):raise RuntimeError("Previous run path changed; use a fresh package directory")
        assert_unchanged(state, check_data=True)
        return state
    runtime = PACKAGE/"runtime_project"
    if runtime.exists(): raise RuntimeError("Incomplete runtime_project exists; use a fresh package directory")
    # Read and validate all source information before creating the snapshot.
    inventory = data_inventory(project)
    state = {"project": str(project), "encoder_package": str(encoder_package),
             "baseline_run": str(baseline_run), "baseline_checkpoint": str(checkpoint),
             "baseline_checkpoint_sha256": file_hash(checkpoint),
             "project_sources": project_sources, "encoder_sources": encoder_sources,
             "data_inventory": inventory,
             "current_training_config": current["training_config"],
             "current_model_config": current["model_config"],"previous_run":str(previous_run),
             "previous_checkpoint_hashes":previous_checkpoint_hashes}
    runtime.mkdir()
    for name in project_sources: shutil.copy2(project/name, runtime/name)
    # Optional historical module is unused but keeps the snapshot self-contained.
    if (project/"T3_14_Calibration.py").is_file():
        shutil.copy2(project/"T3_14_Calibration.py", runtime/"T3_14_Calibration.py")
    (runtime/"data").mkdir()
    for source in ("real", "generated"):
        (runtime/"data"/source).symlink_to(inventory[source]["root"], target_is_directory=True)
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2))
    assert_unchanged(state)
    print(f"Prepared isolated protected-correction experiment at {runtime}; T3.22 checkpoints protected")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="/home/wqzheng/project_transformer")
    parser.add_argument("--encoder-package", default="/home/wqzheng/T3_17_encoder_upgrade")
    parser.add_argument("--baseline-run", default=DEFAULT_BASELINE)
    parser.add_argument("--previous-run",default=DEFAULT_PREVIOUS)
    args = parser.parse_args()
    prepare(args.project,args.encoder_package,args.baseline_run,args.previous_run)


if __name__ == "__main__": main()
