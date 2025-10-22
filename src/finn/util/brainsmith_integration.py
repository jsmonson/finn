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

# Kernels that are intentionally NOT registered (no infer transforms)
# These kernels are either:
# 1. Infrastructure: Inserted by build pipeline or manually placed
# 2. Sub-components: Created by other transforms, not from ONNX directly

INFRASTRUCTURE_KERNELS = {
    'StreamingFIFO',              # Inserted by InsertFIFO/SetFIFODepths
    'StreamingDataWidthConverter', # AXI width alignment
    'TLastMarker',                # AXI stream utility
    'CheckSum',                   # Data verification
    'IODMA',                      # DMA interface, manually placed
}

SUBCOMPONENT_KERNELS = {
    # Created by other infer transforms
    'FMPadding',        # Created by InferConvInpGen when padding needed
    'FMPadding_Pixel',  # Variant of FMPadding
    # Already hardware targets (output of InferElementwiseBinaryOperation)
    'ElementwiseAdd', 'ElementwiseSub', 'ElementwiseMul', 'ElementwiseDiv',
    'ElementwiseAnd', 'ElementwiseOr', 'ElementwiseXor',
    'ElementwiseBitwiseAnd', 'ElementwiseBitwiseOr', 'ElementwiseBitwiseXor',
    'ElementwiseBitShift',
    'ElementwiseEqual', 'ElementwiseGreater', 'ElementwiseGreaterOrEqual',
    'ElementwiseLess', 'ElementwiseLessOrEqual',
    # Note: ElementwiseBinaryOperation has its own infer transform, registered separately
}


def _register_kernels():
    """Register FINN kernels - only those with infer transforms.

    Strategy:
    1. Auto-discover kernels from fpgadataflow/*.py
    2. Filter out infrastructure and sub-component kernels
    3. Enrich with infer_transform metadata

    Returns 16 kernels (all with infer transforms).
    23 kernels excluded (see INFRASTRUCTURE_KERNELS and SUBCOMPONENT_KERNELS).
    """
    kernels = _discover_regular_kernels()  # Discovers and filters
    _add_infer_transforms(kernels)  # All kernels get transforms

    return kernels


def _discover_regular_kernels():
    """Auto-discover standard kernel classes from fpgadataflow/*.py.

    Pattern: HWCustomOp subclasses in finn.custom_op.fpgadataflow module.
    Filters out infrastructure and sub-component kernels.

    Returns:
        List of kernel dicts: [{'name': str, 'class': type}, ...]
        Only kernels with infer transforms are included.
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

                    # Skip infrastructure and sub-component kernels
                    if name in INFRASTRUCTURE_KERNELS or name in SUBCOMPONENT_KERNELS:
                        continue

                    kernels.append({
                        'name': name,
                        'class': cls
                    })

        except Exception:
            pass

    return kernels


def _add_infer_transforms(kernels):
    """Enrich kernel metadata with infer_transform classes.

    Infer transforms are in finn.transformation.fpgadataflow.convert_to_hw_layers,
    a separate module from kernels. No automatic way to link them - requires
    explicit mapping.

    All 16 registered kernels receive transforms via TRANSFORM_MAP below.
    (23 kernels excluded during discovery - see ignore lists at top)

    Args:
        kernels: List of kernel dicts to enrich (modified in-place)
    """
    # Import available infer transforms
    try:
        from finn.transformation.fpgadataflow.convert_to_hw_layers import (
            InferAddStreamsLayer,
            InferBinaryMatrixVectorActivation,
            InferChannelwiseLinearLayer,
            InferConcatLayer,
            InferConvInpGen,
            InferDuplicateStreamsLayer,
            InferElementwiseBinaryOperation,
            InferGlobalAccPoolLayer,
            InferLabelSelectLayer,
            InferLookupLayer,
            InferPool,
            InferQuantizedMatrixVectorActivation,
            InferSplitLayer,
            InferStreamingEltwise,
            InferThresholdingLayer,
            InferUpsample,
            InferVectorVectorActivation,
        )

        # Explicit mapping: kernel name -> transform class
        # Maps each hardware kernel to its corresponding inference transformation
        TRANSFORM_MAP = {
            'AddStreams': InferAddStreamsLayer,
            'ChannelwiseOp': InferChannelwiseLinearLayer,
            'ConvolutionInputGenerator': InferConvInpGen,
            'DuplicateStreams': InferDuplicateStreamsLayer,
            'ElementwiseBinaryOperation': InferElementwiseBinaryOperation,
            'GlobalAccPool': InferGlobalAccPoolLayer,
            'LabelSelect': InferLabelSelectLayer,
            'Lookup': InferLookupLayer,
            'MVAU': InferQuantizedMatrixVectorActivation,
            'Pool': InferPool,
            'StreamingConcat': InferConcatLayer,
            'StreamingEltwise': InferStreamingEltwise,
            'StreamingSplit': InferSplitLayer,
            'Thresholding': InferThresholdingLayer,
            'UpsampleNearestNeighbour': InferUpsample,
            'VVAU': InferVectorVectorActivation,
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
