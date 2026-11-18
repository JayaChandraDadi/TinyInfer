"""Python bindings for the TinyInfer C++ engine."""

from ._tinyinfer import (  # noqa: F401
    Session,
    gemm,
    get_num_threads,
    set_conv_algo,
    set_gemm_kernel,
    set_num_threads,
    simd_available,
    supported_ops,
)

__all__ = [
    "Session",
    "gemm",
    "get_num_threads",
    "set_conv_algo",
    "set_gemm_kernel",
    "set_num_threads",
    "simd_available",
    "supported_ops",
]
