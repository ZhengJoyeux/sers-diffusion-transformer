"""Unified entry point for the current image recognizer and retained experiments."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXTENSIONS = ROOT / 'extensions'


def build_command(stage, project, forwarded):
    project = Path(project).expanduser().resolve()
    outputs = project / 'outputs'
    image = EXTENSIONS / 'T3_32_Image_Recognition'
    correction = EXTENSIONS / 'T3_33_Ternary_Correction'
    image_output = outputs / 't3_32_line_image_seed2026'
    correction_output = outputs / 't3_33_ternary_correction_seed2026'
    shared = ['--correction-package', str(correction)]
    if stage == 'image':
        script = image / 'run_image.py'
        options = ['--project', str(project), '--output-directory', str(image_output),
                   '--image-cache', str(image_output / 'image_cache'),
                   '--anchor-checkpoint', str(outputs / 't3_30_feature_finetune_combined_seed2026/train/concentration/checkpoints/TEB_only/best_accepted.pt')]
    elif stage == 'correction':
        script = correction / 'run_correction.py'
        options = ['--project', str(project), '--image-package', str(image),
                   '--baseline', str(image_output / 'train/checkpoints/best_ordinal.pt'),
                   '--image-cache', str(image_output / 'image_cache'),
                   '--output-directory', str(correction_output)]
    elif stage == 'diagnosis':
        script = EXTENSIONS / 'T3_34_Frozen_Diagnosis/run_diagnosis.py'
        options = shared + ['--run-directory', str(correction_output),
                           '--output-directory', str(outputs / 't3_34_frozen_diagnosis_seed2026')]
    elif stage == 'heads':
        script = EXTENSIONS / 'T3_35_Ternary_Head_Finetune/run_heads.py'
        options = shared + ['--teacher-run', str(correction_output),
                           '--output-directory', str(outputs / 't3_35_ternary_heads_seed2026')]
    elif stage == 'fusion':
        script = EXTENSIONS / 'T3_36_Ternary_Fusion_Finetune/run_fusion.py'
        options = shared + ['--project', str(project), '--image-package', str(image),
                            '--teacher-run', str(correction_output),
                            '--image-cache', str(image_output / 'image_cache'),
                            '--output-directory', str(outputs / 't3_36_ternary_fusion_seed2026')]
    else:
        raise ValueError('Unknown recognition stage: ' + stage)
    if not script.is_file():
        raise FileNotFoundError(script)
    # User options are last, so explicit checkpoint/cache/output paths take precedence.
    return [sys.executable, str(script), *options, *forwarded]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('image','correction','diagnosis','heads','fusion'), default='correction')
    p.add_argument('--project', type=Path, default=ROOT,
                   help='Dataset.py/data/outputs root; use the existing server project to reuse its artifacts')
    p.add_argument('--dry-run', action='store_true', help='Show resolved arguments without loading data or weights')
    args, rest = p.parse_known_args()
    command = build_command(args.stage, args.project, rest)
    if args.dry_run:
        import json
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == '__main__':
    main()
