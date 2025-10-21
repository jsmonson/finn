# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Brainsmith integration for FINN.

This module provides component discovery for Brainsmith plugin system.
FINN components (kernels, backends, steps) are automatically discovered and
returned as metadata for registration by Brainsmith.

Entry point: brainsmith.plugins -> finn = finn.brainsmith_integration:register_all

Note: This module has ZERO dependencies on brainsmith. It only discovers and
returns FINN components - registration is handled by brainsmith.
"""

import importlib
import inspect
from pathlib import Path


def register_all():
    """Discover all FINN components and return metadata for registration.

    This is the entry point function called by Brainsmith's plugin discovery.
    FINN has no dependency on brainsmith - this just returns component data.

    Returns:
        Dict with keys 'kernels', 'backends', 'steps', each containing lists
        of component metadata dicts.

    Example:
        >>> components = register_all()
        >>> len(components['kernels'])
        36
        >>> components['kernels'][0]
        {'name': 'MVAU', 'class': <class 'MVAU'>, 'op_type': 'MVAU'}
    """
    return {
        'kernels': _discover_kernels(),
        'backends': _discover_backends(),
        'steps': _discover_steps()
    }


def _discover_kernels():
    """Auto-discover FINN kernel components.

    Scans finn.custom_op.fpgadataflow for HWCustomOp subclasses.

    Returns:
        List of kernel metadata dicts, each containing:
        - name: Kernel name for registration
        - class: The kernel class object
        - op_type: ONNX operation type
    """
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    import finn.custom_op.fpgadataflow as fpga

    kernels = []

    # Modules to exclude from kernel discovery
    exclude = {
        '__init__.py', 'hwcustomop.py', 'hlsbackend.py',
        'rtlbackend.py', 'templates.py', 'streamingdataflowpartition.py'
    }

    fpga_dir = Path(fpga.__file__).parent

    for py_file in sorted(fpga_dir.glob('*.py')):
        if py_file.name in exclude:
            continue

        module_name = f'finn.custom_op.fpgadataflow.{py_file.stem}'
        try:
            module = importlib.import_module(module_name)

            # Find HWCustomOp subclasses
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (cls.__module__ == module.__name__ and
                    issubclass(cls, HWCustomOp) and
                    cls is not HWCustomOp):

                    # Use op_type as name if available
                    kernel_name = getattr(cls, 'op_type', name)

                    kernels.append({
                        'name': kernel_name,
                        'class': cls,
                        'op_type': kernel_name
                    })

        except Exception:
            # Skip modules that fail to import
            pass

    return kernels


def _discover_backends():
    """Auto-discover FINN backend components.

    Scans finn.custom_op.fpgadataflow.hls and .rtl for backend classes.

    Returns:
        List of backend metadata dicts, each containing:
        - name: Backend name for registration
        - class: The backend class object
        - target_kernel: Full kernel name (e.g., 'finn:MVAU')
        - language: 'hls' or 'rtl'
    """
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
    from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
    import finn.custom_op.fpgadataflow.hls as hls_module
    import finn.custom_op.fpgadataflow.rtl as rtl_module

    backends = []

    # Scan HLS backends
    hls_dir = Path(hls_module.__file__).parent
    for py_file in sorted(hls_dir.glob('*.py')):
        if py_file.name == '__init__.py':
            continue

        module_name = f'finn.custom_op.fpgadataflow.hls.{py_file.stem}'
        try:
            module = importlib.import_module(module_name)

            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (cls.__module__ == module.__name__ and
                    issubclass(cls, HLSBackend) and
                    cls is not HLSBackend):

                    target_kernel = _infer_target_kernel(cls, 'hls')
                    backends.append({
                        'name': name,
                        'class': cls,
                        'target_kernel': target_kernel,
                        'language': 'hls'
                    })

        except Exception:
            pass

    # Scan RTL backends
    rtl_dir = Path(rtl_module.__file__).parent
    for py_file in sorted(rtl_dir.glob('*.py')):
        if py_file.name == '__init__.py':
            continue

        module_name = f'finn.custom_op.fpgadataflow.rtl.{py_file.stem}'
        try:
            module = importlib.import_module(module_name)

            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (cls.__module__ == module.__name__ and
                    issubclass(cls, RTLBackend) and
                    cls is not RTLBackend):

                    target_kernel = _infer_target_kernel(cls, 'rtl')
                    backends.append({
                        'name': name,
                        'class': cls,
                        'target_kernel': target_kernel,
                        'language': 'rtl'
                    })

        except Exception:
            pass

    return backends


def _discover_steps():
    """Auto-discover FINN builder step functions.

    Scans finn.builder.build_dataflow_steps for functions starting with 'step_'.

    Returns:
        List of step metadata dicts, each containing:
        - name: Step name for registration (without 'step_' prefix)
        - func: The step function object
    """
    from finn.builder import build_dataflow_steps

    steps = []

    # Find all step_* functions
    for name, func in inspect.getmembers(build_dataflow_steps, inspect.isfunction):
        if name.startswith('step_') and func.__module__ == build_dataflow_steps.__name__:
            # Remove 'step_' prefix for registration name
            step_name = name[5:]
            steps.append({
                'name': step_name,
                'func': func
            })

    return steps


def _infer_target_kernel(backend_class, language):
    """Infer target kernel from backend's parent class hierarchy.

    Args:
        backend_class: The backend class to analyze
        language: 'hls' or 'rtl'

    Returns:
        Full kernel name with source prefix (e.g., 'finn:MVAU')

    Strategy:
        1. Check parent classes for kernel (non-Backend class from fpgadataflow)
        2. If not found, parse backend name (remove _hls/_rtl suffix)
    """
    # Strategy 1: Find kernel in parent classes
    for base in backend_class.__bases__:
        base_module = getattr(base, '__module__', '')
        base_name = getattr(base, '__name__', '')

        if (base_module.startswith('finn.custom_op.fpgadataflow') and
            not base_name.endswith('Backend') and
            base_name not in ('HWCustomOp', 'CustomOp')):
            return f'finn:{base_name}'

    # Strategy 2: Parse backend class name
    name = backend_class.__name__
    if name.endswith(f'_{language}'):
        kernel_name = name[:-len(f'_{language}')]
        return f'finn:{kernel_name}'

    # Fallback: use backend name as-is
    return f'finn:{name}'
