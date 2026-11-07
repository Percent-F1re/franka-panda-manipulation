"""Read-only load of the user's original 46D/4D Stable-Baselines3 SAC ZIP."""
from __future__ import annotations
import io
from pathlib import Path
import zipfile

from numpy_pickle_compat import enable_numpy_bitgenerator_pickle_compat


def verify_checkpoint(path: str | Path) -> tuple[int, int]:
    import torch
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'Original 46D PickPlace model not found: {p}')
    with zipfile.ZipFile(p) as z:
        if 'policy.pth' not in z.namelist():
            raise ValueError(f'Not an SB3 SAC ZIP (policy.pth missing): {p}')
        # Only use a checkpoint you own/trust: SB3 ZIP metadata contains pickle.
        weights = torch.load(io.BytesIO(z.read('policy.pth')), map_location='cpu', weights_only=True)
    a = weights.get('actor.latent_pi.0.weight')
    b = weights.get('actor.mu.weight')
    if a is None or b is None:
        raise ValueError('Original SAC actor tensors missing from policy.pth')
    dims = (int(a.shape[1]), int(b.shape[0]))
    if dims != (46, 4):
        raise ValueError(f'Expected ORIGINAL 46D/4D SAC, found {dims}. Do not use staged 39D policies.')
    print('[Legacy checkpoint] Original actor weights verified: 46D -> 4D', flush=True)
    return dims


def load_legacy_sac(path: str | Path, env, device='cpu'):
    from stable_baselines3 import SAC
    verify_checkpoint(path)
    if tuple(env.observation_space.shape) != (46,) or tuple(env.action_space.shape) != (4,):
        raise ValueError('Original SAC environment must expose 46D/4D interfaces')
    enable_numpy_bitgenerator_pickle_compat()
    model = SAC.load(str(Path(path).expanduser().resolve()), env=env, device=device,
                     custom_objects={
                         'observation_space': env.observation_space,
                         'action_space': env.action_space,
                     })
    if tuple(model.policy.observation_space.shape) != (46,):
        raise ValueError('Loaded SAC policy has incompatible observation space')
    print('[Legacy checkpoint] Loaded untouched original weights', flush=True)
    return model
