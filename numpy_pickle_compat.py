"""Compatibility support for ORIGINAL SAC checkpoints saved with NumPy 2.x.

Use only when loading a checkpoint that you trust. A Stable-Baselines3 checkpoint
contains pickled Python objects, so loading untrusted checkpoints can execute code.

The source checkpoint is never modified. This is a narrow process-local alias
for NumPy 2.x pickle GLOBAL module paths when running under NumPy 1.26.x.
It also fixes older NumPy's PCG64-class BitGenerator pickle constructor.

Public entry point keeps the name expected by the previous migration evaluator:
    enable_numpy_bitgenerator_pickle_compat()
"""
from __future__ import annotations

import importlib
import sys
from typing import Iterable

# Submodules actually present in NumPy 1.26's numpy.core. Keep this list
# comprehensive for the common ndarray/dtype/ufunc pickle references. Do NOT
# alias random._pickle to core; that is a separate existing NumPy module.
_CORE_SUBMODULES = (
    "numeric", "multiarray", "_multiarray_umath", "umath", "numerictypes",
    "_dtype", "_dtype_ctypes", "_internal", "_methods", "_exceptions",
    "_ufunc_config", "fromnumeric", "shape_base", "function_base",
    "arrayprint", "getlimits", "records", "memmap", "einsumfunc",
    "overrides", "defchararray", "machar", "_add_newdocs", "_add_newdocs_scalars",
)


def enable_numpy2_core_pickle_compat(*, verbose: bool = True) -> tuple[str, ...]:
    """Make NumPy-2 private pickle module names importable under NumPy 1.x.

    Existing real NumPy 2 modules are never overwritten. In NumPy 1.x, aliases
    are installed once in ``sys.modules`` and remain for this Python process.
    This does NOT replace NumPy, upgrade it, modify the checkpoint, or change
    array values; it only resolves paths stored by Python pickle.
    """
    import numpy as np

    try:
        # In NumPy 2.x the package is genuine: leave it entirely untouched.
        importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError as err:
        # Only repair the missing numpy._core namespace, not broken dependencies.
        if err.name not in ("numpy._core", "numpy._core.numeric"):
            raise
    else:
        if verbose:
            print(f"[Legacy checkpoint] numpy._core is native (NumPy {np.__version__})", flush=True)
        return ()

    package = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", package)
    # Some pickle loading paths may first resolve `numpy._core` as an attribute.
    if not hasattr(np, "_core"):
        setattr(np, "_core", package)

    installed = []
    for short_name in _CORE_SUBMODULES:
        name = "numpy._core." + short_name
        if name in sys.modules:
            continue
        try:
            old_module = importlib.import_module("numpy.core." + short_name)
        except (ModuleNotFoundError, ImportError, AttributeError):
            # Some internals vary by NumPy 1.x patch level. They are optional.
            continue
        sys.modules[name] = old_module
        installed.append(name)
        # Ensure ordinary attribute-based imports also work where relevant.
        if not hasattr(package, short_name):
            try:
                setattr(package, short_name, old_module)
            except (AttributeError, TypeError):
                pass
    if "numpy._core.numeric" not in sys.modules:
        raise RuntimeError(
            "Cannot register numpy._core.numeric compatibility alias; "
            "installed NumPy core module is incomplete."
        )
    if verbose:
        print(
            "[Legacy checkpoint] NumPy 2.x pickle module paths mapped to "
            f"NumPy {np.__version__} numpy.core ({len(installed)} submodules)",
            flush=True,
        )
    return tuple(installed)


def enable_numpy_bitgenerator_pickle_compat(*, verbose: bool = True) -> None:
    """Fix PCG64-class pickles *and* NumPy-2 private module paths.

    This is the same public function imported by the existing
    ``eval_legacy_sac_robosuite.py``. No other project source needs changing.
    Call before ``SAC.load``.
    """
    import numpy as np
    import numpy.random._pickle as np_pickle
    from numpy.random.bit_generator import BitGenerator

    # Register the private-module aliases first: NumPy may unpickle array
    # dtype/state objects before invoking the BitGenerator constructor.
    enable_numpy2_core_pickle_compat(verbose=verbose)

    current_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if current_ctor is None:
        raise RuntimeError("NumPy random pickle constructor not found")
    if getattr(current_ctor, "_legacy_class_compatible", False):
        if verbose:
            print("[Legacy checkpoint] BitGenerator class/string constructor already compatible", flush=True)
        return

    def compatible_bit_generator_ctor(bit_generator="MT19937"):
        if isinstance(bit_generator, type):
            try:
                if issubclass(bit_generator, BitGenerator):
                    return bit_generator()
            except TypeError:
                pass
        return current_ctor(bit_generator)

    # NumPy stores this constructor as an importable module-global in RNG
    # pickle payloads. Keep its __module__ pointing to the actual patched
    # NumPy module, otherwise future pickle.dumps(Generator) would fail.
    compatible_bit_generator_ctor.__module__ = np_pickle.__name__
    compatible_bit_generator_ctor.__name__ = "__bit_generator_ctor"
    compatible_bit_generator_ctor.__qualname__ = "__bit_generator_ctor"
    compatible_bit_generator_ctor._legacy_class_compatible = True
    np_pickle.__bit_generator_ctor = compatible_bit_generator_ctor

    # The generator helper may have captured the old constructor as a default.
    # Keep both helpers consistent with the new entry point.
    for helper_name in ("__generator_ctor", "__randomstate_ctor"):
        helper = getattr(np_pickle, helper_name, None)
        defaults = getattr(helper, "__defaults__", None) if helper else None
        if defaults:
            helper.__defaults__ = tuple(
                compatible_bit_generator_ctor if value is current_ctor else value
                for value in defaults
            )

    bit_generators = getattr(np_pickle, "BitGenerators", None)
    if isinstance(bit_generators, dict):
        for cls in tuple(bit_generators.values()):
            if isinstance(cls, type):
                bit_generators.setdefault(cls, cls)

    if verbose:
        print(
            "[Legacy checkpoint] PCG64 BitGenerator class/string pickle "
            f"compatibility enabled (NumPy {np.__version__})",
            flush=True,
        )
