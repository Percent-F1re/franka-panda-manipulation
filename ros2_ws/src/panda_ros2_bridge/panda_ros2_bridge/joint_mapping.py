"""Pure mapping from Menagerie joint names to Panda URDF names.

Does not import ROS, MuJoCo or robosuite; can be tested offline.
"""

from math import isfinite

ARM = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGERS = ("panda_finger_joint1", "panda_finger_joint2")


def translate_joint_state(names, positions, velocities, opening_m=0.04):
    """Return nine URDF joints in a deterministic order.

    Supports Menagerie joint1..joint7 and already-prefixed panda_joint1..7.
    opening_m means total fingertip spread; max physical opening is 0.08m.
    """
    if not (len(names) == len(positions) == len(velocities)):
        raise ValueError("JointState names, position, velocity lengths differ")
    lookup = {str(n): (float(q), float(dq))
              for n, q, dq in zip(names, positions, velocities)}
    q = []
    dq = []
    for i in range(1, 8):
        raw = lookup.get(f"joint{i}", lookup.get(f"panda_joint{i}"))
        if raw is None:
            raise ValueError(f"Missing arm joint joint{i} / panda_joint{i}")
        if not all(isfinite(v) for v in raw):
            raise ValueError(f"Non-finite joint{i} state")
        q.append(raw[0])
        dq.append(raw[1])
    opening = float(opening_m)
    if not isfinite(opening):
        raise ValueError("Non-finite gripper opening")
    finger = max(0.0, min(0.04, opening / 2.0))
    return list(ARM + FINGERS), q + [finger, finger], dq + [0.0, 0.0]
