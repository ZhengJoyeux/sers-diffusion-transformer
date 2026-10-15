"""Hash-guarded install. Back up every overwritten file, never touch outputs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='/home/wqzheng/project')
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    project = Path(a.project).expanduser().resolve()
    package = Path(__file__).parent
    manifest = json.loads((package / 'file_hashes.json').read_text())
    pending = []
    for relative, hashes in manifest.items():
        target = project / relative
        actual = digest(target)
        if actual == hashes['after']:
            continue
        if actual != hashes['before']:
            raise ValueError(f'文件与D26+T24基础版本不同，未修改任何文件：{target}\n实际sha256={actual}\n期望={hashes["before"]}')
        if digest(package / 'payload' / relative) != hashes['after']:
            raise ValueError(f'corrupted payload: {relative}')
        pending.append(relative)
    print('Preflight passed:', len(pending), 'files to install')
    if a.check_only or not pending:
        return
    backup = project / 'patch_backups' / ('d4_27_dit_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True)
    for relative in pending:
        target = project / relative
        if target.exists():
            old = backup / relative; old.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(target, old)
    (backup / 'installation.json').write_text(json.dumps({'files': pending, 'manifest': manifest}, indent=2))
    try:
        for relative in pending:
            target = project / relative; target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.d427_tmp')
            shutil.copy2(package / 'payload' / relative, temp); temp.replace(target)
    except BaseException:
        for relative in pending:
            old = backup / relative; target = project / relative
            if old.exists(): shutil.copy2(old, target)
            elif manifest[relative]['before'] is None and digest(target) == manifest[relative]['after']: target.unlink()
        raise
    print('Installed. Backup:', backup)


if __name__ == '__main__':
    main()
