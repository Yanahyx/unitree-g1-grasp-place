"""No robot connection or motion: verify recording and interactive labels."""
import json
import os
import io
import threading
import types
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import test_grasp_20 as runner


class RecordingChecks(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'Full runner uses Linux flock on the robot')
    def test_twenty_trial_loop_including_automatic_errors(self):
        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as tmp:
                sdk = Path(tmp)
                source = sdk / 'example/g1/real_arm_grasp/g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb.py'
                source.parent.mkdir(parents=True)
                source.write_text('# never executed\n')
                output = sdk / 'output'
                def simulate(cmd, sdk, folder, timeout):
                    failed = folder.name in ('trial_002', 'trial_007')
                    r = {'exit_code': 4 if failed else 0, 'total_seconds': .1,
                         'status': 'error' if failed else 'awaiting_feedback', 'positions': {}}
                    if failed:
                        r.update(success=0, label_source='automatic_error')
                    runner.save_json(folder / 'result.json', r)
                    return r, False
                state = types.SimpleNamespace(record_lock=threading.Lock())
                server = types.SimpleNamespace(shutdown=lambda: None, server_close=lambda: None)
                with patch.object(sys, 'argv', ['test', '--sdk-dir', str(sdk), '--output-root', str(output)]), \
                     patch.object(sys.stdin, 'isatty', return_value=True), \
                     patch.object(runner, 'other_actions', return_value=[]), \
                     patch.object(runner, 'make_detector', return_value=(state, server, 'http://127.0.0.1:1/detect')), \
                     patch.object(runner, 'run_action', side_effect=simulate) as actions, \
                     patch('builtins.input', side_effect=lambda prompt: '1' if '是否成功' in prompt else ''), \
                     patch('sys.stdout', new=io.StringIO()):
                    runner.main()
                sessions = list(output.glob('session_*'))
                rows = json.loads((sessions[0] / 'results.json').read_text())
                self.assertEqual(actions.call_count, 20)
                self.assertEqual(len(rows), 20)
                self.assertEqual(sum(r['success'] for r in rows), 18)
                self.assertEqual(sum(r['label_source'] == 'automatic_error' for r in rows), 2)
        finally:
            os.chdir(str(previous_cwd))

    def test_latest_motion_parameters_and_speech_disabled(self):
        cmd = runner.command(Path('/sdk'), 'http://127.0.0.1:1234/detect')
        for key, value in {'--pre-duration': '2.0', '--hand-hold': '1.5',
                           '--hold-after-open': '0.2', '--vision-fallback-source': 'none'}.items():
            self.assertEqual(cmd[cmd.index(key) + 1], value)
        self.assertIn('--no-announce', cmd)
        self.assertNotIn('--announce-text', cmd)

    def test_invalid_feedback_reprompts(self):
        with patch('builtins.input', side_effect=['yes', '', '1']):
            self.assertEqual(runner.ask_success(), 1)
        with patch('builtins.input', return_value='0'):
            self.assertEqual(runner.ask_success(), 0)

    def test_process_error_auto_zero_and_position_export(self):
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as tmp:
            session = Path(tmp)
            folder = session / 'trial_001'
            folder.mkdir()
            cmd = [sys.executable, '-c', "print('vision_point_pelvis_m: [ 0.4 -0.1 0.02]'); raise SystemExit(4)"]
            result, interrupted = runner.run_action(cmd, session, folder, 10)
            self.assertFalse(interrupted)
            self.assertEqual(result['success'], 0)
            self.assertEqual(result['exit_code'], 4)
            self.assertEqual(result['positions']['vision_point_pelvis_m'], [.4, -.1, .02])
            rows = runner.summarize(session)
            self.assertEqual(len(rows), 1)
            self.assertIn('0.4,-0.1,0.02', (session / 'results.csv').read_text(encoding='utf-8-sig'))

    def test_normal_exit_requires_human_label_and_keeps_pending(self):
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).parent)) as tmp:
            folder = Path(tmp)
            result, interrupted = runner.run_action([sys.executable, '-c', 'print("done")'], folder, folder, 10)
            self.assertEqual(result['status'], 'awaiting_feedback')
            self.assertNotIn('success', result)
            self.assertEqual(json.loads((folder / 'result.json').read_text())['status'], 'awaiting_feedback')


if __name__ == '__main__':
    unittest.main()
