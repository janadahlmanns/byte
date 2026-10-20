"""The production path's own compile-cache folder (plan Step 18b).

Inductor's FX graph cache reuses a compiled module for the same graph whatever batch
width it was first compiled at, and that width sets the kernels' block configs: a
module compiled at full width ran 2.2-2.4x slower (Step 16). Since Step 18a every graph
the production path compiles is first compiled at COMPILE_WARMUP_WIDTH, but modules
compiled before that sit in the machines' existing caches and would keep being reused.
So the production path (`make_tensor_evaluator`) keeps its modules in a versioned
subfolder of the usual cache folder: the old modules are never seen, and everything in
the new folder was compiled through the warm-up, so reusing it is safe and free.

Bump COMPILE_CACHE_VERSION whenever the compile policy changes (warm-up width, inductor
settings). Tests and diagnostics, which compile at other widths, keep the default folder.

Deliberately torch-free at import: simulate/run_selected.py loads this file directly to
report the folder's state without importing torch.
"""

import getpass
import os
import re
import tempfile

COMPILE_CACHE_VERSION = "byte-w512-v1"
ENV = "TORCHINDUCTOR_CACHE_DIR"


def _torch_default_cache_dir() -> str:
    """torch._inductor.runtime.cache_dir_utils.default_cache_dir(), reproduced (torch
    2.13): <tempdir>/torchinductor_<user>."""
    try:
        user = getpass.getuser()
    except (KeyError, ModuleNotFoundError, OSError):
        getuid = getattr(os, "getuid", None)
        user = f"uid_{getuid()}" if callable(getuid) else "unknown_user"
    return os.path.join(tempfile.gettempdir(),
                        "torchinductor_" + re.sub(r'[\\/:*?"<>|]', "_", user))


def base_cache_dir() -> str:
    """The cache folder torch would use without Step 18b: $TORCHINDUCTOR_CACHE_DIR if set
    (Docker: /cache/inductor), otherwise torch's default. If the variable already points
    at the versioned folder (this process switched before), its parent."""
    d = os.environ.get(ENV)
    if d is None:
        return _torch_default_cache_dir()
    d = os.path.abspath(d)
    return os.path.dirname(d) if os.path.basename(d) == COMPILE_CACHE_VERSION else d


def versioned_cache_dir() -> str:
    return os.path.join(base_cache_dir(), COMPILE_CACHE_VERSION)


def use_versioned_compile_cache() -> str:
    """Point this process's inductor cache at the versioned folder; returns it.

    torch reads the variable at every use (cache_dir_utils.cache_dir()), so this must
    run before the process's first compile. If something was compiled already, the
    process may hold modules from another folder: that raises instead of mixing."""
    target = versioned_cache_dir()
    current = os.environ.get(ENV)
    if current is not None and os.path.abspath(current) == target:
        return target
    import torch._dynamo.utils as du     # only here: the module stays torch-free to import
    if int(du.counters["stats"]["unique_graphs"]) > 0:
        raise RuntimeError(
            f"[ERROR] this process has already compiled {du.counters['stats']['unique_graphs']} "
            f"graph(s) with the compile cache at {current or _torch_default_cache_dir()}; "
            f"the production evaluator needs its own folder {target} from the first compile "
            f"(plan Step 18b). Build the compiled evaluator before anything else compiles, or "
            f"in its own process.")
    os.environ[ENV] = target
    os.makedirs(target, exist_ok=True)
    return target
