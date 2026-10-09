import argparse
import json
from pathlib import Path
from apply_fix import digest, atomic_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--backup', required=True, type=Path)
    args = parser.parse_args()
    project, backup = args.project.resolve(), args.backup.resolve()
    if not backup.is_relative_to(project / 'outputs/upgrades'):
        raise SystemExit('backup必须在项目outputs/upgrades内。')
    specs = json.loads((Path(__file__).parent / 'file_hashes.json').read_text())
    records = json.loads((backup / 'restore_manifest.json').read_text())
    for record in records:
        relative = record['path']
        if relative not in specs:
            raise SystemExit('备份清单路径不属于本包。')
        current = project / relative
        if not current.is_file() or digest(current) != specs[relative]['after']:
            raise SystemExit(f'代码已变化，停止回滚，未修改任何文件: {relative}')
        if record['existed'] and digest(backup / relative) != specs[relative]['before']:
            raise SystemExit(f'备份校验失败: {relative}')
    for record in records:
        target = project / record['path']
        if record['existed']:
            atomic_copy(backup / record['path'], target)
        else:
            target.unlink()
    print('ROLLBACK PASS')


if __name__ == '__main__':
    main()
