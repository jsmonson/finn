"""
FINN Plugin System

Provides registration and discovery for FINN transforms, kernels, and backends,
enabling integration with the unified QONNX/BrainSmith plugin system.
"""

from .adapters import transform, kernel, backend
from .registry import FinnPluginRegistry, get_finn_registry

__all__ = [
    'transform',
    'kernel',
    'backend',
    'FinnPluginRegistry',
    'get_finn_registry'
]