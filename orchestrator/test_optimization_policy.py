from pathlib import Path

from orchestrator.optimization_policy import production_kernel_violations


CUDA_MARKER = '''
CUDA_SOURCE = r"extern \\\"C\\\" __global__ void kernel() {}"
from cuda.bindings import nvrtc
'''


def _violations(tmp_path: Path, source: str) -> list[str]:
    (tmp_path / "kernel.py").write_text(source)
    return production_kernel_violations(
        tmp_path,
        "Cuda",
        dependency_reviewer=lambda _workspace, _framework, _signals: [],
    )


def test_ctypes_abi_types_are_allowed_for_cuda_launch_plumbing(tmp_path: Path) -> None:
    source = CUDA_MARKER + '''
import ctypes
values = (1, 2)
types = (ctypes.c_void_p, ctypes.c_int)
'''
    assert not _violations(tmp_path, source)


def test_ctypes_dynamic_library_loaders_remain_forbidden(tmp_path: Path) -> None:
    for source in (
        CUDA_MARKER + '\nimport ctypes\nlib = ctypes.CDLL("libcompute.so")\n',
        CUDA_MARKER + '\nimport ctypes as ct\nlib = ct.cdll.LoadLibrary("libcompute.so")\n',
        CUDA_MARKER + '\nfrom ctypes import CDLL as load\nlib = load("libcompute.so")\n',
        CUDA_MARKER + '\nimport ctypes\napi = ctypes.pythonapi\n',
    ):
        violations = _violations(tmp_path, source)
        assert any("dynamic external-code loading" in item for item in violations)
