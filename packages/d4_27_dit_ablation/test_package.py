import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil

package=Path(__file__).parent
manifest=json.loads((package/'file_hashes.json').read_text())
base=package/'test_fixtures/originals'
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory)
 for file,spec in manifest.items():
  if spec['before']:
   dest=root/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/file,dest)
 command=[sys.executable,str(package/'apply_patch.py'),'--project',str(root)]
 subprocess.run(command+['--check-only'],check=True)
 subprocess.run(command,check=True)
 subprocess.run(command,check=True)
 backups=list((root/'patch_backups').iterdir());assert len(backups)==1
 for file,spec in manifest.items():
  if spec['before']:assert (backups[0]/file).read_bytes()==(base/file).read_bytes()
 first='src/model_builder.py';(root/first).write_text('# independent user edit\n')
 before={f:(root/f).read_bytes() for f in manifest}
 result=subprocess.run(command,capture_output=True,text=True);assert result.returncode!=0
 assert before=={f:(root/f).read_bytes() for f in manifest}
 print('PACKAGE CHECKS PASSED: preflight, install, backup, idempotence, no mutation on drift')
