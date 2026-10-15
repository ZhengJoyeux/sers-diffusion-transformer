"""Check the delivered package without touching project files."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
expected=json.loads((root/'checksums.json').read_text())
bad=[name for name,digest in expected.items() if not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
if bad:
    raise SystemExit('Package check FAILED: '+str(bad))
print('Package check PASS: '+str(len(expected))+' files')
