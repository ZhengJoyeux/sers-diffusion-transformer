"""Run the fixed-condition-mean 2D residual diffusion pipeline."""

import argparse
import runpy
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "fixed_mean_residual"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "generate", "postprocess", "evaluate", "evaluate-native",
        "prepare", "train", "check", "selftest",
    ))
    args = parser.parse_args(sys.argv[1:2])
    entry, prefix = {
        "generate": ("generate_ridge0.py", []),
        "postprocess": ("pca_diversity_trial.py", []),
        "evaluate": ("evaluate_sers126.py", ["--project", str(root)]),
        "evaluate-native": ("run.py", ["evaluate"]),
    }.get(args.command, ("run.py", [args.command]))
    sys.path.insert(0, str(root))
    sys.argv = [str(root / entry), *prefix, *sys.argv[2:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
