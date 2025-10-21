"""Brainsmith integration for FINN.

Entry point: brainsmith.plugins -> finn = finn.brainsmith_integration:register_all

Strategy:
- Auto-discover components following standard patterns
- Manually register exceptions and cross-module associations
- Zero dependencies on brainsmith (returns metadata only)

Components:
- Steps: Auto-discovered from finn.builder.build_dataflow_steps
- Kernels: Auto-discovered from fpgadataflow/*.py + manual special cases
- Backends: Auto-discovered from hls/ and rtl/ directories
- Infer Transforms: Manual mapping (cross-module association)
"""

import importlib
import inspect
from pathlib import Path


def register_all():
    """Return FINN components for Brainsmith registration.

    This is the entry point function called by Brainsmith's plugin discovery.
    FINN has no dependency on brainsmith - this just returns component data.

    Returns:
        Dict with keys 'kernels', 'backends', 'steps', each containing
        lists of component metadata dicts.

    Example:
        >>> components = register_all()
        >>> 'steps' in components and 'kernels' in components
        True
        >>> all(isinstance(components[k], list) for k in components)
        True
    """
    return {
        'kernels': _register_kernels(),
        'backends': _register_backends(),
        'steps': _discover_steps()
    }


# ============================================================================
# KERNELS: Hybrid auto-discovery + manual enrichment
# ============================================================================

def _register_kernels():
    """Register FINN kernels - hybrid approach.

    Strategy:
    1. Auto-discover: Regular kernels from fpgadataflow/*.py
    2. Manual add: Special kernels (combined kernel+backend)
    3. Manual enrich: Add infer_transform metadata where applicable

    Returns ~36 kernels total.
    """
    kernels = _discover_regular_kernels()  # ~33 kernels
    kernels.extend(_register_special_kernels())  # +3 special
    _add_infer_transforms(kernels)  # Enrich ~10 with transforms

    return kernels


def _discover_regular_kernels():
    """Auto-discover standard kernel classes from fpgadataflow/*.py.

    Pattern: HWCustomOp subclasses in finn.custom_op.fpgadataflow module.

    Returns:
        List of kernel dicts: [{'name': str, 'class': type}, ...]
    """
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
    import finn.custom_op.fpgadataflow as fpga

    kernels = []

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

            # Find all HWCustomOp subclasses defined in this module
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (cls.__module__ == module.__name__ and
                    issubclass(cls, HWCustomOp) and
                    cls is not HWCustomOp):

                    kernels.append({
                        'name': name,
                        'class': cls
                    })

        except Exception:
            pass

    return kernels


def _register_special_kernels():
    """Manually register special kernels that don't follow standard pattern.

    These are combined kernel+backend implementations that live in hls/
    instead of fpgadataflow/. They break the discovery pattern.

    Returns:
        List of kernel dicts for special cases.
    """
    from finn.custom_op.fpgadataflow.hls.checksum_hls import CheckSum_hls
    from finn.custom_op.fpgadataflow.hls.iodma_hls import IODMA_hls
    from finn.custom_op.fpgadataflow.hls.tlastmarker_hls import TLastMarker_hls

    return [
        {'name': 'CheckSum', 'class': CheckSum_hls},
        {'name': 'IODMA', 'class': IODMA_hls},
        {'name': 'TLastMarker', 'class': TLastMarker_hls},
    ]


def _add_infer_transforms(kernels):
    """Enrich kernel metadata with infer_transform classes.

    Infer transforms are in finn.transformation.fpgadataflow.convert_to_hw_layers,
    a separate module from kernels. No automatic way to link them - requires
    explicit mapping.

    Only ~10 out of 36 kernels support inference transformations.

    Args:
        kernels: List of kernel dicts to enrich (modified in-place)
    """
    # Import available infer transforms
    try:
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferQuantizedMatrixVectorActivation,
            InferVectorVectorActivation,
            InferThresholdingLayer,
            InferChannelwiseLinearLayer,
        )

        # Explicit mapping: kernel name -> transform class
        # Add more entries as FINN exposes new infer transforms
        TRANSFORM_MAP = {
            'MVAU': InferQuantizedMatrixVectorActivation,
            'VVAU': InferVectorVectorActivation,
            'Thresholding': InferThresholdingLayer,
            'ChannelwiseOp': InferChannelwiseLinearLayer,
        }

        # Enrich discovered kernels with transforms
        for kernel in kernels:
            if kernel['name'] in TRANSFORM_MAP:
                kernel['infer_transform'] = TRANSFORM_MAP[kernel['name']]

    except ImportError:
        pass


# ============================================================================
# BACKENDS: Full auto-discovery
# ============================================================================

def _register_backends():
    """Auto-discover all FINN backends from hls/ and rtl/ directories.

    Pattern:
    - HLS: finn.custom_op.fpgadataflow.hls.*_hls.py contains HLSBackend subclasses
    - RTL: finn.custom_op.fpgadataflow.rtl.*_rtl.py contains RTLBackend subclasses
    - Target kernel: Remove _hls or _rtl suffix from class name

    Returns ~40+ backends (mix of HLS and RTL implementations).
    """
    backends = []
    backends.extend(_discover_hls_backends())
    backends.extend(_discover_rtl_backends())
    return backends


def _discover_hls_backends():
    """Auto-discover HLS backend implementations.

    Returns:
        List of backend dicts with name, class, target_kernel, language.
    """
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
    import finn.custom_op.fpgadataflow.hls as hls_module

    backends = []
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

                    target = name[:-4] if name.endswith('_hls') else name

                    backends.append({
                        'name': name,
                        'class': cls,
                        'target_kernel': f'finn:{target}',
                        'language': 'hls'
                    })

        except Exception:
            pass

    return backends


def _discover_rtl_backends():
    """Auto-discover RTL backend implementations.

    Returns:
        List of backend dicts with name, class, target_kernel, language.
    """
    from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
    import finn.custom_op.fpgadataflow.rtl as rtl_module

    backends = []
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

                    target = name[:-4] if name.endswith('_rtl') else name

                    backends.append({
                        'name': name,
                        'class': cls,
                        'target_kernel': f'finn:{target}',
                        'language': 'rtl'
                    })

        except Exception:
            pass

    return backends


# ============================================================================
# STEPS: Full auto-discovery (unchanged)
# ============================================================================

def _discover_steps():
    """Auto-discover FINN builder step functions.

    Pattern: Functions named step_* in finn.builder.build_dataflow_steps

    Returns:
        List of step dicts with name and func.
    """
    from finn.builder import build_dataflow_steps

    steps = []

    for name, func in inspect.getmembers(build_dataflow_steps, inspect.isfunction):
        if name.startswith('step_') and func.__module__ == build_dataflow_steps.__name__:
            step_name = name[5:]  # Remove 'step_' prefix
            steps.append({
                'name': step_name,
                'func': func
            })

    return steps
