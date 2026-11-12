"""Offline tests: stdlib only, ROS 2 not needed."""
import importlib.util
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / 'ros2_ws' / 'src' /
          'panda_ros2_bridge' / 'panda_ros2_bridge' / 'joint_mapping.py')
spec = importlib.util.spec_from_file_location('joint_mapping', SOURCE)
mapping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapping)


def test_menagerie_7_to_9():
    names, q, dq = mapping.translate_joint_state(
        [f'joint{i}' for i in range(1, 8)], [0.1 * i for i in range(7)], [0] * 7, 0.06)
    assert names[:7] == [f'panda_joint{i}' for i in range(1, 8)]
    assert names[7:] == ['panda_finger_joint1', 'panda_finger_joint2']
    assert q[7:] == [0.03, 0.03]
    assert dq == [0] * 9


def test_replay_default_gripper_and_clamp():
    data = ([f'joint{i}' for i in range(1, 8)], [0] * 7, [0] * 7)
    assert mapping.translate_joint_state(*data)[1][-2:] == [0.02, 0.02]
    assert mapping.translate_joint_state(*data, opening_m=0.3)[1][-2:] == [0.04, 0.04]
    assert mapping.translate_joint_state(*data, opening_m=-3)[1][-2:] == [0.0, 0.0]


def test_errors():
    try:
        mapping.translate_joint_state(['joint1'], [0], [0])
    except ValueError:
        pass
    else:
        raise AssertionError('Expected missing joint rejection')


if __name__ == '__main__':
    test_menagerie_7_to_9()
    test_replay_default_gripper_and_clamp()
    test_errors()
    print('PASS: joint name adaptation, gripper limits, missing joint validation')
