"""Hash-checked, backed-up, idempotent application; no config/checkpoint edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.d4_26_hybrid_')
    os.close(fd)
    try:
        shutil.copy2(source, name)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    args = parser.parse_args()
    root = args.project.expanduser().resolve()
    package = Path(__file__).resolve().parent
    specs = json.loads((package / 'file_hashes.json').read_text())
    prerequisites = json.loads((package / 'prerequisites.json').read_text())
    for relative, expected in prerequisites.items():
        target = root / relative
        if not target.is_file() or digest(target) != expected:
            raise SystemExit(f'先前阶段追踪/终点保护/联合采样源码版本不符；未修改任何文件: {relative}')
    pending = []
    for relative, hashes in specs.items():
        source, target = package / 'payload' / relative, root / relative
        if digest(source) != hashes['after']:
            raise SystemExit(f'包内文件校验失败: {relative}')
        compile(source.read_bytes(), str(source), 'exec')
        existing = digest(target) if target.is_file() else None
        if existing == hashes['after']:
            continue
        if existing != hashes['before']:
            raise SystemExit(f'服务器代码与提交源码不一致，未修改任何文件: {relative}')
        if target.exists() and not target.is_file():
            raise SystemExit(f'目标不是文件: {target}')
        pending.append((relative, source, target, existing is not None))
    if not pending:
        print('APPLY PASS (already applied)')
        return
    backup = root / 'outputs/upgrades' / ('d4_26_hybrid_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir(parents=True, exist_ok=False)
    records = []
    for relative, source, target, existed in pending:
        if existed:
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        records.append({'path': relative, 'existed': existed})
    (backup / 'restore_manifest.json').write_text(json.dumps(records, indent=2))
    written = []
    try:
        for relative, source, target, existed in pending:
            atomic_copy(source, target)
            written.append((relative, target, existed))
            if digest(target) != specs[relative]['after']:
                raise RuntimeError(f'写入校验失败: {relative}')
    except Exception:
        for relative, target, existed in reversed(written):
            if existed:
                atomic_copy(backup / relative, target)
            else:
                target.unlink(missing_ok=True)
        raise
    print(f'APPLY PASS\nbackup: {backup}\n修改文件数: {len(pending)}')


if __name__ == '__main__':
    main()
