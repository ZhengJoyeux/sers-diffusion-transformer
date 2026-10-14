"""Pack reviewable reports; exclude spectra, model weights and optimizer states."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--suite', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
suite = args.suite.resolve(strict=True)
if args.output.exists():
    raise FileExistsError(args.output)
files = sorted(p for p in suite.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.log'))
with ZipFile(args.output, 'x', ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, suite.name + '/' + path.relative_to(suite).as_posix())
print(f'Packed {len(files)} report files: {args.output}')
