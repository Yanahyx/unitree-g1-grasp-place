#!/usr/bin/env python3
"""Interactive G1 grasp trials. Run on the robot; --check never moves it."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def command(sdk, endpoint):
    return [sys.executable, '-u', str(sdk / 'example/g1/real_arm_grasp/g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb.py'),
            'eth1', '--use-vision', '--detector-server-url', endpoint,
            '--detector-retries', '0', '--vision-fallback-source', 'none',
            '--object', 'bottle', '--vision-xy-offset', '-0.06', '0.0',
            '--side-standoff-y', '0.0', '--pre-duration', '2.0', '--return-duration', '3.0',
            '--hold-before-open', '0', '--hold-after-open', '0.2',
            '--hold-before-grasp', '0', '--hold-after-grasp', '0.3',
            '--hand-speed', '300', '--hand-force', '800', '--hand-hold', '1.5',
            '--thumb-inward-delay', '0.3', '--thumb-rotate-speed', '500',
            '--thumb-rotate-force', '2000', '--thumb-rotate-step', '40',
            '--thumb-rotate-hold', '0.03', '--thumb-close-tolerance', '120',
            '--feedback-force-delta', '40', '--feedback-current-delta', '35',
            '--feedback-min-phase', '0.85', '--close-at-approach-end',
            '--no-announce', '--execute', '--yes']


def other_actions():
    rows = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
    return [r for r in rows if 'python' in r and
            ('grasp_sequence' in r or 'place_sequence' in r)]


def make_detector(sdk, rgbd_url):
    import cv2
    import numpy as np
    source = sdk / 'example/g1/grasp/g1_yolo_world_detection_server.py'
    spec = importlib.util.spec_from_file_location('existing_grasp_detector', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RecordingDetector(module.DetectorState):
        trial_dir = None

        def capture_frame(self):
            frame = super().capture_frame()
            self.last_capture = frame
            if self.trial_dir is not None:
                stem = self.trial_dir / ('detection_%02d' % self.request_count)
                if not cv2.imwrite(str(stem) + '_rgb.png', frame['color_bgr']):
                    raise RuntimeError('Cannot save detection RGB')
                np.savez_compressed(str(stem) + '_rgbd.npz', **{
                    k: v for k, v in frame.items() if v is not None})
            return frame

        def detect(self, object_name, conf, imgsz):
            # HTTPServer serves one request at a time. Main changes trial_dir
            # only between action processes and waits on this lock.
            with self.record_lock:
                stem = self.trial_dir / ('detection_%02d' % (self.request_count + 1))
                try:
                    result = super().detect(object_name, conf, imgsz)
                    result['recorded_rgb_file'] = stem.name + '_rgb.png'
                    save_json(Path(str(stem) + '.json'), result)
                    annotated = self.last_capture['color_bgr'].copy()
                    box = result['detection']['bbox_xyxy']
                    x1, y1, x2, y2 = (round(v) for v in box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, 'bottle %.3f' % result['detection']['confidence'],
                                (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                    if not cv2.imwrite(str(stem) + '_annotated.png', annotated):
                        raise RuntimeError('Cannot save detection overlay')
                    return result
                except Exception as exc:
                    save_json(Path(str(stem) + '_error.json'), {'error': repr(exc), 'time': now()})
                    raise

    args = argparse.Namespace(model=str(sdk / 'yolov8s-world.pt'), object='bottle',
                              source='http-rgbd', fallback_source='none',
                              rgbd_npz_url=rgbd_url, http_timeout=10., serial='243322071448',
                              no_warmup_predict=True, conf=.10, imgsz=640,
                              width=640, height=480, fps=30, camera_warmup_frames=30,
                              frame_timeout_ms=5000)
    state = RecordingDetector(args)
    state.record_lock = threading.Lock()
    # A dynamically assigned loopback port avoids changing any existing service.
    from http.server import HTTPServer
    server = HTTPServer(('127.0.0.1', 0), module.make_handler(state, args.conf, args.imgsz))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return state, server, 'http://127.0.0.1:%d/detect' % server.server_port


def run_action(cmd, sdk, folder, timeout):
    result = {'started_at': now(), 'command': cmd, 'status': 'running'}
    save_json(folder / 'result.json', result)
    start = time.monotonic()
    interrupted = False
    child = None
    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        with (folder / 'run.log').open('w', encoding='utf-8', buffering=1) as log:
            child = subprocess.Popen(cmd, cwd=str(sdk), env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            result['pid'] = child.pid
            save_json(folder / 'result.json', result)
            with (folder / 'run.log').open(encoding='utf-8', errors='replace') as reader:
                while child.poll() is None:
                    part = reader.read()
                    if part:
                        print(part, end='', flush=True)
                    if time.monotonic() - start > timeout:
                        print('\n超时：发送中断，等待脚本释放手臂控制；不自动重试。', flush=True)
                        result['error'] = 'action_timeout'
                        child.send_signal(signal.SIGINT)
                        child.wait()  # existing script finally releases arm control
                        break
                    time.sleep(.1)
                print(reader.read(), end='', flush=True)
            result['exit_code'] = child.returncode
    except KeyboardInterrupt:
        interrupted = True
        result['error'] = 'operator_interrupted'
        if child is not None and child.poll() is None:
            print('\n正在中断动作并等待手臂控制释放，请勿重复启动。', flush=True)
            child.send_signal(signal.SIGINT)
            child.wait()
        result['exit_code'] = child.returncode if child else None
    except Exception as exc:
        result['error'] = repr(exc)
        result['exit_code'] = None
    finally:
        result['total_seconds'] = time.monotonic() - start
        result['ended_at'] = now()
        log = (folder / 'run.log').read_text(encoding='utf-8', errors='replace') if (folder / 'run.log').exists() else ''
        result['positions'] = {}
        for field in ['vision_point_camera_m', 'vision_point_pelvis_m', 'final_grasp_target']:
            match = re.search(r'^' + field + r': \[([^\]]+)\]', log, re.M)
            if match:
                result['positions'][field] = [float(v) for v in match.group(1).split()]
        result['feedback_warnings'] = [line for line in log.splitlines() if line.startswith('Warning:')]
        result['completion_tts_api_ok'] = 'TTS success announced:' in log
        result['status'] = 'awaiting_feedback' if result.get('exit_code') == 0 and not result.get('error') else 'error'
        if result['status'] == 'error':
            result.update(success=0, label_source='automatic_error')
        save_json(folder / 'result.json', result)
    return result, interrupted


def ask_success():
    while True:
        text = input('本次抓取是否成功？1=成功，0=失败（先摆好下一次瓶子，输入后立即开始下一次）：').strip()
        if text in ('0', '1'):
            return int(text)
        print('请输入 1 或 0。')


def summarize(session):
    rows = []
    for folder in sorted(session.glob('trial_*')):
        path = folder / 'result.json'
        if path.exists():
            r = json.loads(path.read_text(encoding='utf-8'))
            r['trial_id'] = folder.name
            rows.append(r)
    save_json(session / 'results.json', rows)
    fields = ['trial_id', 'success', 'label_source', 'status', 'exit_code',
              'total_seconds', 'pixel_x', 'pixel_y', 'camera_x_m', 'camera_y_m',
              'camera_z_m', 'pelvis_x_m', 'pelvis_y_m', 'pelvis_z_m', 'error']
    with (session / 'results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {key: r.get(key, '') for key in fields}
            folder = session / r['trial_id']
            detections = sorted(folder.glob('detection_*.json'))
            detections = [p for p in detections if not p.name.endswith('_error.json')]
            if detections:
                det = json.loads(detections[-1].read_text(encoding='utf-8'))
                xy = det['detection']['center_xy']
                row.update(pixel_x=xy[0], pixel_y=xy[1])
                for axis, value in zip('xyz', det['point_camera_m']):
                    row['camera_' + axis + '_m'] = value
            for axis, value in zip('xyz', r.get('positions', {}).get('vision_point_pelvis_m', [])):
                row['pelvis_' + axis + '_m'] = value
            writer.writerow(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--sdk-dir', type=Path, default=Path('/home/unitree/unitree_sdk2_python'))
    p.add_argument('--output-root', type=Path, default=Path('/home/unitree/grasp_trials/formal'))
    p.add_argument('--rgbd-url', default='http://127.0.0.1:18082/rgbd.npz')
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--check', action='store_true', help='Save one real detection; never launch robot actions')
    a = p.parse_args()
    if a.trials < 1 or a.timeout <= 0:
        p.error('trials and timeout must be positive')
    if not a.check and not sys.stdin.isatty():
        p.error('请在交互终端运行，不能使用 nohup 或输入重定向。')
    sdk = a.sdk_dir.resolve()
    os.chdir(str(sdk))
    # Prevent two copies of this runner from taking turns commanding the robot.
    import fcntl
    a.output_root.mkdir(parents=True, exist_ok=True)
    lock = Path('/tmp/g1_interactive_grasp_trials.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another test runner is already active')
    if other_actions():
        raise RuntimeError('检测到其他抓取或放置脚本，停止启动测试。')
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    session = a.output_root / (('check_' if a.check else 'session_') + stamp)
    session.mkdir()
    print('保存目录：', session, flush=True)
    print('加载本轮专用检测模型（整个测试只加载一次）；不会修改现有检测服务。', flush=True)
    state, server, endpoint = make_detector(sdk, a.rgbd_url)
    try:
        cmd = command(sdk, endpoint)
        source = Path(cmd[2])
        config = {'created_at': now(), 'trial_count': a.trials, 'command': cmd,
                  'script_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'timing_definition': 'Action subprocess launch to exit including initialization, vision, motion and hand waits; speech disabled; excludes model startup, user input and recording file transfer.',
                  'rgbd_url': a.rgbd_url, 'mode': 'check' if a.check else 'formal',
                  'detection_source': 'Same existing DetectorState implementation, dedicated model instance; exact captured image saved before inference; no local camera fallback.'}
        save_json(session / 'session.json', config)
        if a.check:
            state.trial_dir = session
            state.detect('bottle', .10, 640)
            print('CHECK_OK：检测原图/标注图/坐标已保存；没有执行机器人动作。')
            return
        input('共 %d 次。先放好瓶子并保持运动空间清空，按回车开始第 1 次（Ctrl+C 退出）：' % a.trials)
        for number in range(1, a.trials + 1):
            if other_actions():
                raise RuntimeError('发现其他动作脚本，停止本轮测试。')
            folder = session / ('trial_%03d' % number)
            folder.mkdir()
            with state.record_lock:
                state.trial_dir = folder
            print('\n===== 第 %d/%d 次 =====' % (number, a.trials), flush=True)
            result, interrupted = run_action(cmd, sdk, folder, a.timeout)
            result['trial_id'] = folder.name
            save_json(folder / 'result.json', result)
            summarize(session)
            if interrupted:
                print('本轮已中止，记录已保存。')
                break
            print('\n本次脚本总耗时：%.2f 秒' % result['total_seconds'])
            if result['status'] == 'awaiting_feedback':
                result.update(success=ask_success(), label_source='user', status='labelled')
                save_json(folder / 'result.json', result)
                summarize(session)
            else:
                print('运行错误，已自动记为 0。详情见', folder / 'run.log')
                if number < a.trials:
                    input('请检查现场并摆好下一次瓶子，按回车继续（Ctrl+C 结束）：')
        rows = summarize(session)
        labelled = [r for r in rows if r.get('success') in (0, 1)]
        wins = sum(r['success'] for r in labelled)
        print('\n已记录 %d 次，成功 %d 次，成功率 %.1f%%。' % (len(labelled), wins, 100 * wins / len(labelled) if labelled else 0))
        print('结果目录：', session)
    finally:
        server.shutdown()
        server.server_close()
        summarize(session)
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\n已退出，已完成的记录保留；未反馈的成功脚本标记为 awaiting_feedback。')
