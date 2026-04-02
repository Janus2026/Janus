import importlib.util
import os
import sys
import sysconfig


def _get_python_version_tag() -> str:
    # returns "310", "311", ...
    return sysconfig.get_python_version().replace(".", "")


def _find_export_so_path() -> str:
    pkg_dir = os.path.dirname(__file__)
    pyver = _get_python_version_tag()

    # Preferred, exact tags we build for today.
    candidates = [
        os.path.join(pkg_dir, f"janus_export.cpython-{pyver}-x86_64-linux-gnu.so"),
        os.path.join(pkg_dir, f"janus_export.cpython-{pyver}-aarch64-linux-gnu.so"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)

    # Fallback: accept any janus_export*.so that got packaged (tag may differ).
    for fname in os.listdir(pkg_dir):
        if fname.startswith("janus_export") and fname.endswith(".so"):
            return os.path.abspath(os.path.join(pkg_dir, fname))

    raise ImportError(
        f"cannot find janus_export shared library under {pkg_dir!r}. "
        f"Expected one of: {candidates!r}"
    )


_export_so_path = _find_export_so_path()
_spec = importlib.util.spec_from_file_location("janus_export", _export_so_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"failed to create import spec for janus_export: {_export_so_path}")

# Make `import janus_export` work for submodules (pybind/*) by loading and
# registering it before importing any modules that depend on it.
janus_export = importlib.util.module_from_spec(_spec)
sys.modules["janus_export"] = janus_export
_spec.loader.exec_module(janus_export)

from janus.pybind.embedding import Embedding
from janus.pybind.llm import LLM
try:
    from janus.pybind.vlm import VLM
except Exception:
    VLM = None
from janus.pybind.args import ArgumentParser
from janus.pybind.params import SamplingParams, BeamSearchParams, PoolingParams
from janus_export import (
    LLMMaster,
    VLMMaster,
    Options,
    RequestParams,
    RequestOutput,
    Usage,
    SequenceOutput,
    Status,
    StatusCode,
    MMType,
    MMData,
)

__all__ = [
    "ArgumentParser",
    "Embedding",
    "LLM",
    "LLMMaster",
    "VLM",
    "VLMMaster",
    "Options",
    "SamplingParams",
    "BeamSearchParams",
    "PoolingParams",
    "RequestParams",
    "RequestOutput",
    "Usage",
    "SequenceOutput",
    "Status",
    "StatusCode",
]
