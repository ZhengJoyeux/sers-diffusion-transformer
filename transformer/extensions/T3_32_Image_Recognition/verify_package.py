"""Validate exactly the delivered patch files; never modify the existing project."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'checksums.json').read_text())
errors=[]
expected_python={name for name in manifest if name.endswith('.py')}
extra_python={p.name for p in root.glob('*.py')}-expected_python
errors.extend(name+': unexpected Python file' for name in sorted(extra_python))
for name, expected in manifest.items():
    path=root/name
    if not path.is_file(): errors.append(name+': missing')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected: errors.append(name+': SHA256 differs')
if errors: raise SystemExit('\n'.join(errors))
print(f'PATCH CHECKSUMS: PASS ({len(manifest)} files)')
