"""Both controller arms run the SAME original Menagerie Panda physical model.

Add only two massless MuJoCo sites to derived XML files (never overwrite original
user XML): a hand-body reference for robosuite's actual OSC_POSE implementation,
and a static reference site required by robosuite OSC's velocity subtraction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import xml.etree.ElementTree as ET

from envs.robosuite_menagerie_core import LegacyMenageriePandaRobosuite, MODEL_DIR

SOURCE_ROBOT = MODEL_DIR / 'panda_pickplace.xml'
SOURCE_SCENE = MODEL_DIR / 'scene_pickplace_stage3.xml'
DUAL_ROBOT = MODEL_DIR / 'panda_pickplace_dual_controllers.xml'
DUAL_SCENE = MODEL_DIR / 'scene_pickplace_dual_controllers.xml'
EEF_SITE = 'dual_osc_hand_ref'
CENTER_SITE = 'hand_center'
_LOCK = threading.Lock()


def _xml_bytes(root: ET.Element) -> bytes:
    # Formatting-only XML changes are permitted; only sites are added/renamed.
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def _write_if_changed(path: Path, contents: bytes) -> None:
    if not path.exists() or path.read_bytes() != contents:
        path.write_bytes(contents)


def ensure_dual_scene() -> Path:
    if not SOURCE_ROBOT.is_file() or not SOURCE_SCENE.is_file():
        raise FileNotFoundError(
            f'Original Menagerie MJCF files missing: {SOURCE_ROBOT}, {SOURCE_SCENE}. '
            'Install legacy_sac_robosuite_exact_v1.zip first.'
        )
    with _LOCK:
        robot_root = ET.parse(SOURCE_ROBOT).getroot()
        hands = robot_root.findall(".//body[@name='hand']")
        if len(hands) != 1:
            raise RuntimeError(f'Expected exactly one original hand body; found {len(hands)}')
        if robot_root.find(f".//site[@name='{EEF_SITE}']") is not None:
            raise RuntimeError(f'Original MJCF already defines {EEF_SITE}; choose another site name')
        ET.SubElement(hands[0], 'site', {
            'name': EEF_SITE, 'pos': '0 0 0', 'size': '0.001',
            'type': 'sphere', 'rgba': '0 0 0 0',
        })
        scene_root = ET.parse(SOURCE_SCENE).getroot()
        includes = scene_root.findall('include')
        if len(includes) != 1 or includes[0].get('file') != SOURCE_ROBOT.name:
            raise RuntimeError('Expected Stage3 scene to include original panda_pickplace.xml once')
        includes[0].set('file', DUAL_ROBOT.name)
        world = scene_root.find('worldbody')
        if world is None:
            raise RuntimeError('Original scene is missing worldbody')
        if scene_root.find(f".//site[@name='{CENTER_SITE}']") is not None:
            raise RuntimeError(f'Original MJCF already defines {CENTER_SITE}')
        # Required by robosuite 1.5.2 osc.py velocity calculation when
        # part_name='hand', naming_prefix=''. World-fixed => zero velocity.
        ET.SubElement(world, 'site', {
            'name': CENTER_SITE, 'pos': '0 0 0', 'size': '0.001',
            'type': 'sphere', 'rgba': '0 0 0 0',
        })
        _write_if_changed(DUAL_ROBOT, _xml_bytes(robot_root))
        _write_if_changed(DUAL_SCENE, _xml_bytes(scene_root))
        return DUAL_SCENE


class DualSceneMenagerieRobosuite(LegacyMenageriePandaRobosuite):
    """Original robosuite engine; only two non-colliding reference sites added."""

    def __init__(self, *, control_period_s=0.04, **kwargs):
        scene_path = ensure_dual_scene()
        super().__init__(scene_path=scene_path, control_period_s=control_period_s, **kwargs)
