"""Windows-only robosuite SAC demo runner. Benchmark sources remain untouched.

Run this under the EXISTING Windows robosuite .venv. ROS 2 / rclpy are NOT
required on Windows. JSON/UDP carries state to the WSL2 ROS 2 bridge.
Service reset requests run at policy-step boundaries, never inside DLS/OSC.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

PROTOCOL = "panda_bridge_v1"


def phase_from_info(info, terminated=False, truncated=False):
    if terminated:
        return 'success' if info.get('is_success') else 'done'
    if truncated:
        return 'timeout'
    if info.get('place_streak', 0) > 0:
        return 'placing'
    if info.get('has_transferred'):
        return 'transferred'
    if info.get('has_lifted'):
        return 'lifting'
    if info.get('grasped'):
        return 'grasping'
    return 'approaching'


def snapshot(env, obs, action, info, *, seed, run_id, phase, clock_offset):
    import mujoco

    assert len(obs) == 46, 'Unexpected observation shape'
    assert len(action) == 4, 'Unexpected action shape'
    # MjData.xquat stores body-world quaternion in MuJoCo [w,x,y,z] order.
    def quat(body_id):
        return [float(v) for v in env.data.xquat[body_id]]

    joint_names = []
    for addr in env.qpos_ids:
        ids = np.where(np.asarray(env.model.jnt_qposadr) == int(addr))[0]
        if len(ids) != 1:
            raise RuntimeError('Cannot map qpos address to unique joint ID')
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, int(ids[0]))
        joint_names.append(name or f'joint_{int(ids[0])}')

    cube_pos = [float(v) for v in env.data.xpos[env.cube]]
    # Use one monotonically increasing ROS time across episode resets; raw MuJoCo
    # time restarts at its 1.5s settled initialization on every reset.
    return {
        'protocol': PROTOCOL, 'kind': 'state', 'run_id': run_id,
        'seed': int(seed), 'controller': env.controller, 'phase': phase,
        'step': int(env.step_count),
        'sim_time_s': float(env.data.time),
        'clock_s': float(clock_offset + env.data.time),
        'joint_names': joint_names,
        'joint_q': [float(v) for v in env.data.qpos[env.qpos_ids]],
        'joint_dq': [float(v) for v in env.data.qvel[env.dof_ids]],
        'hand_xyz': [float(v) for v in env.data.xpos[env.hand]],
        'hand_quat_wxyz': quat(env.hand),
        'cube_xyz': cube_pos,
        'cube_quat_wxyz': quat(env.cube),
        'target_xyz': [float(env.target_xy[0]), float(env.target_xy[1]),
                       float(env.table_z + env.s4.CUBE_HALF_WIDTH)],
        'cube_side_m': 2.0 * float(env.s4.CUBE_HALF_WIDTH),
        'observation': [float(v) for v in obs],
        'sac_action': [float(v) for v in action],
        'gripper_opening_m': float(sum(env.data.qpos[i] for i in env.finger_qpos_ids)),
        'original_is_success': bool(info.get('is_success', False)),
        'has_lifted': bool(info.get('has_lifted', False)),
        'has_transferred': bool(info.get('has_transferred', False)),
        'cube_xy_error_mm': float(info.get('cube_xy_error_mm', 0.0)),
    }


class Telemetry:
    def __init__(self, host, port, token):
        self.peer = (host, int(port))
        self.token = token
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 0))
        self.sock.setblocking(False)
        print(f'[UDP] Windows local port={self.sock.getsockname()[1]} -> WSL {host}:{port}')

    def send(self, packet):
        packet = dict(packet, token=self.token)
        wire = json.dumps(packet, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(wire) >= 16000:
            raise RuntimeError('Snapshot exceeds UDP packet safety limit')
        self.sock.sendto(wire, self.peer)

    def receive_reset(self):
        reset_id = None
        while True:
            try:
                raw, peer = self.sock.recvfrom(8192)
            except BlockingIOError:
                break
            if peer != self.peer:
                continue
            try:
                msg = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                continue
            if msg.get('protocol') != PROTOCOL or msg.get('token') != self.token:
                continue
            if msg.get('kind') == 'command' and msg.get('command') == 'reset':
                reset_id = str(msg.get('request_id', ''))[:64]
        return reset_id

    def close(self):
        self.sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--controller', choices=('dls', 'osc'), default='dls')
    ap.add_argument('--seed', type=int, default=42000)
    ap.add_argument('--wsl-host', required=True, help='WSL2 IPv4 from wsl hostname -I')
    ap.add_argument('--port', type=int, default=15555)
    ap.add_argument('--token', default='local_demo_only', help='Same token on both processes')
    ap.add_argument('--viewer', action='store_true', help='MuJoCo native window on Windows')
    ap.add_argument('--autostart', action='store_true', help='Start first episode immediately')
    ap.add_argument('--max-episode-steps', type=int, default=600)
    ap.add_argument('--idle-hz', type=float, default=2.0)
    ap.add_argument('--realtime', action='store_true', help='Sleep to approximately match 0.04s simulation steps')
    args = ap.parse_args()
    if not args.model.is_file() or not (1 <= args.port <= 65535) or args.idle_hz <= 0:
        ap.error('Check --model, --port and --idle-hz')
    from envs.panda_dual_controller import PandaDualControllerEnv
    from legacy_sac_checkpoint import load_legacy_sac

    env = PandaDualControllerEnv(controller=args.controller, task='pickplace',
                                 max_episode_steps=args.max_episode_steps,
                                 render_mode='human' if args.viewer else None)
    comm = Telemetry(args.wsl_host, args.port, args.token)
    try:
        model = load_legacy_sac(args.model, env)
        run_id = 0
        offset = 0.0
        last_clock = 0.0
        obs, info = env.reset(seed=args.seed)
        action = np.zeros(4, dtype=np.float32)
        active = bool(args.autostart)
        phase = 'ready'
        last_idle_publish = 0.0
        print('[READY] Run /panda/reset_episode Service in WSL to start or restart demo.')
        while True:
            request_id = comm.receive_reset()
            if request_id is not None:
                # Make /clock strictly monotonically increasing across resets.
                before = last_clock
                obs, info = env.reset(seed=args.seed)
                offset = max(offset, before + env.control_dt - float(env.data.time))
                run_id += 1
                action = np.zeros(4, dtype=np.float32)
                phase = 'approaching'
                active = True
                comm.send({'protocol': PROTOCOL, 'kind': 'event',
                           'event': 'reset_applied', 'request_id': request_id,
                           'run_id': run_id})
                print(f'[RESET] request_id={request_id} run={run_id} seed={args.seed}')
            if active:
                begin = time.monotonic()
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                phase = phase_from_info(info, terminated, truncated)
                last_clock = offset + float(env.data.time)
                comm.send(snapshot(env, obs, action, info, seed=args.seed,
                                   run_id=run_id, phase=phase, clock_offset=offset))
                if args.viewer:
                    env.render()
                if terminated or truncated:
                    print(f'[END] run={run_id} phase={phase} steps={env.step_count} '
                          f'original_success={info["is_success"]}')
                    active = False
                if args.realtime:
                    delay = float(env.control_dt) - (time.monotonic() - begin)
                    if delay > 0:
                        time.sleep(delay)
            else:
                now = time.monotonic()
                if now - last_idle_publish >= 1.0 / args.idle_hz:
                    last_clock = max(last_clock, offset + float(env.data.time))
                    comm.send(snapshot(env, obs, action, info, seed=args.seed,
                                       run_id=run_id, phase=phase, clock_offset=offset))
                    last_idle_publish = now
                    if args.viewer:
                        env.render()
                time.sleep(0.015)
    except KeyboardInterrupt:
        print('[STOP] Ctrl+C')
    finally:
        comm.close()
        env.close()


if __name__ == '__main__':
    main()
