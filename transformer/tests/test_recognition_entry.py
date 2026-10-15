"""Verify repository-relative dispatch without loading datasets or checkpoints."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'run_recognition.py'
spec = importlib.util.spec_from_file_location('recognition_entry_under_test', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def test_every_stage_resolves_its_delivered_script(self):
        for stage in ('image','correction','diagnosis','heads','fusion'):
            with self.subTest(stage=stage):
                command = entry.build_command(stage, entry.ROOT, [])
                self.assertTrue(Path(command[1]).is_file())
                self.assertTrue(Path(command[1]).is_relative_to(entry.EXTENSIONS))

    def test_existing_project_keeps_existing_cache_and_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            command = entry.build_command('correction', Path(folder), [])
            self.assertEqual(command[command.index('--project')+1], str(Path(folder).resolve()))
            baseline = command[command.index('--baseline')+1]
            self.assertTrue(baseline.startswith(str(Path(folder).resolve() / 'outputs')))

    def test_explicit_options_remain_last(self):
        forwarded = ['--output-directory','/custom/result','--device','cpu']
        command = entry.build_command('fusion', entry.ROOT, forwarded)
        self.assertEqual(command[-len(forwarded):], forwarded)

    def test_unknown_stage_fails(self):
        with self.assertRaises(ValueError):
            entry.build_command('unknown', entry.ROOT, [])


if __name__ == '__main__':
    unittest.main()
