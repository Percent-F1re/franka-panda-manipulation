"""stdlib + numpy only: checks UDP request roundtrip without ROS2/MuJoCo."""
from __future__ import annotations

import json
import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ros2_demo_runner import PROTOCOL, Telemetry, phase_from_info


class UDPTests(unittest.TestCase):
    def test_telemetry_request(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(('127.0.0.1', 0))
        rx.settimeout(1)
        client = Telemetry('127.0.0.1', rx.getsockname()[1], 'token_for_test')
        try:
            client.send({'protocol': PROTOCOL, 'kind': 'state', 'step': 1})
            raw, sender = rx.recvfrom(8192)
            payload = json.loads(raw)
            self.assertEqual(payload['protocol'], PROTOCOL)
            self.assertEqual(payload['token'], 'token_for_test')
            request = {'protocol': PROTOCOL, 'token': 'token_for_test', 'kind': 'command',
                       'command': 'reset', 'request_id': 'req0'}
            rx.sendto(json.dumps(request).encode('utf-8'), sender)
            import time
            for _ in range(50):
                found = client.receive_reset()
                if found is not None:
                    break
                time.sleep(0.005)
            self.assertEqual(found, 'req0')
        finally:
            rx.close()
            client.close()

    def test_wrong_token_ignored(self):
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(('127.0.0.1', 0))
        rx.settimeout(1)
        client = Telemetry('127.0.0.1', rx.getsockname()[1], 'correct')
        try:
            client.send({'protocol': PROTOCOL, 'kind': 'state'})
            _, sender = rx.recvfrom(8192)
            rx.sendto(json.dumps({'protocol': PROTOCOL, 'kind': 'command',
                        'command': 'reset', 'request_id': 'attacker',
                        'token': 'incorrect'}).encode('utf-8'), sender)
            import time
            time.sleep(.02)
            self.assertIsNone(client.receive_reset())
        finally:
            rx.close()
            client.close()

    def test_phase(self):
        self.assertEqual(phase_from_info({'grasped': False}), 'approaching')
        self.assertEqual(phase_from_info({'has_lifted': True}), 'lifting')
        self.assertEqual(phase_from_info({'has_transferred': True}), 'transferred')
        self.assertEqual(phase_from_info({'is_success': True}, terminated=True), 'success')


if __name__ == '__main__':
    unittest.main()
