"""Validate the incremental correction patch without touching existing code."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'checksums.json').read_text())
errors=[]
for name,expected in manifest.items():
    path=root/name
    if not path.is_file():errors.append(name+': missing')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append(name+': SHA256 differs')
extra={p.name for p in root.glob('*.py')}-{name for name in manifest if name.endswith('.py')}
errors.extend(name+': unexpected Python file' for name in sorted(extra))
if errors:raise SystemExit('\n'.join(errors))
print(f'T3.33 PATCH CHECKSUMS: PASS ({len(manifest)} files)')
