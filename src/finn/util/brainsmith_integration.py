############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

"""Brainsmith integration for FINN.

Entry point: brainsmith.plugins -> finn = finn.brainsmith_integration:register_all

Strategy:
- Static declaration of all components (fast discovery ~1ms)
- Zero dependencies on brainsmith (returns metadata only)
- Lazy loading: components imported only when accessed

Components:
- Steps: 19 build pipeline steps (static list)
- Kernels: 18 hardware kernels (16 computational + 2 infrastructure)
- Backends: 27 implementations (17 HLS + 10 RTL, static list)
- Infer Transforms: Manual mapping (lazy metadata)

Infrastructure Kernels:
Infrastructure kernels (DuplicateStreams, StreamingFIFO, StreamingDataWidthConverter)
are marked with is_infrastructure=True and are filtered out of InferKernelList
when kernel_classes=None. They are inserted by topology transforms (InsertFIFO,
InsertDWC, InsertDuplicateStreams) rather than pattern matching.

Legacy Components:
CheckSum_hls, TLastMarker_hls, IODMA_hls are legacy FINN backend-only components
(no base kernel class) and are not registered in Brainsmith.

Maintenance:
When adding new FINN components, add them to the static lists in
_register_kernels(), _register_backends(), and _discover_steps().
"""


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
        "kernels": _register_kernels(),
        "backends": _register_backends(),
        "steps": _discover_steps(),
    }


# ============================================================================
# KERNELS: Hybrid auto-discovery + manual enrichment
# ============================================================================

# Kernels that are intentionally NOT registered (sub-components only)
# These are created by other transforms, not from ONNX directly
# Note: Infrastructure kernels are now registered with is_infrastructure=True

SUBCOMPONENT_KERNELS = {
    # Created by other infer transforms
    "FMPadding",  # Created by InferConvInpGen when padding needed
    "FMPadding_Pixel",  # Variant of FMPadding
    # Note: Specialized elementwise operations (ElementwiseAdd, etc.) are now registered
    # as full kernels for backend specialization support
}


def _register_kernels():
    """Register FINN kernels - static list for performance.

    Returns 18 kernels:
    - 16 computational kernels (with infer transforms)
    - 2 infrastructure kernels (marked with is_infrastructure=True)

    Infrastructure kernels are inserted by topology transforms (InsertFIFO,
    InsertDWC, etc.) rather than pattern matching, and are filtered out of
    InferKernelList when kernel_classes=None.

    Note: CheckSum, TLastMarker, IODMA are legacy FINN backend-only components
    (no base kernel class) and are not registered in Brainsmith.

    Note: This is a static list for fast discovery (~1ms).
    When adding new kernels to FINN, add them here manually.
    """
    return [
        {
            "name": "AddStreams",
            "module": "finn.custom_op.fpgadataflow.addstreams",
            "class_name": "AddStreams",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferAddStreamsLayer",
            },
        },
        {
            "name": "ChannelwiseOp",
            "module": "finn.custom_op.fpgadataflow.channelwise_op",
            "class_name": "ChannelwiseOp",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferChannelwiseLinearLayer",
            },
        },
        {
            "name": "StreamingConcat",
            "module": "finn.custom_op.fpgadataflow.concat",
            "class_name": "StreamingConcat",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferConcatLayer",
            },
        },
        {
            "name": "ConvolutionInputGenerator",
            "module": "finn.custom_op.fpgadataflow.convolutioninputgenerator",
            "class_name": "ConvolutionInputGenerator",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferConvInpGen",
            },
        },
        {
            "name": "DuplicateStreams",
            "module": "finn.custom_op.fpgadataflow.duplicatestreams",
            "class_name": "DuplicateStreams",
            "is_infrastructure": True,  # Inserted by topology transforms (InsertDuplicateStreams)
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferDuplicateStreamsLayer",
            },
        },
        {
            "name": "ElementwiseBinaryOperation",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseBinaryOperation",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferElementwiseBinaryOperation",
            },
        },
        # Specialized elementwise operations (output of InferElementwiseBinaryOperation)
        {
            "name": "ElementwiseAdd",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseAdd",
        },
        {
            "name": "ElementwiseSub",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseSub",
        },
        {
            "name": "ElementwiseMul",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseMul",
        },
        {
            "name": "ElementwiseDiv",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseDiv",
        },
        {
            "name": "ElementwiseAnd",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseAnd",
        },
        {
            "name": "ElementwiseOr",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseOr",
        },
        {
            "name": "ElementwiseXor",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseXor",
        },
        {
            "name": "ElementwiseEqual",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseEqual",
        },
        {
            "name": "ElementwiseLess",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseLess",
        },
        {
            "name": "ElementwiseLessOrEqual",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseLessOrEqual",
        },
        {
            "name": "ElementwiseGreater",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseGreater",
        },
        {
            "name": "ElementwiseGreaterOrEqual",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseGreaterOrEqual",
        },
        {
            "name": "ElementwiseBitwiseAnd",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseBitwiseAnd",
        },
        {
            "name": "ElementwiseBitwiseOr",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseBitwiseOr",
        },
        {
            "name": "ElementwiseBitwiseXor",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseBitwiseXor",
        },
        {
            "name": "ElementwiseBitShift",
            "module": "finn.custom_op.fpgadataflow.elementwise_binary",
            "class_name": "ElementwiseBitShift",
        },
        {
            "name": "GlobalAccPool",
            "module": "finn.custom_op.fpgadataflow.globalaccpool",
            "class_name": "GlobalAccPool",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferGlobalAccPoolLayer",
            },
        },
        {
            "name": "LabelSelect",
            "module": "finn.custom_op.fpgadataflow.labelselect",
            "class_name": "LabelSelect",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferLabelSelectLayer",
            },
        },
        {
            "name": "Lookup",
            "module": "finn.custom_op.fpgadataflow.lookup",
            "class_name": "Lookup",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferLookupLayer",
            },
        },
        {
            "name": "MVAU",
            "module": "finn.custom_op.fpgadataflow.matrixvectoractivation",
            "class_name": "MVAU",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferQuantizedMatrixVectorActivation",
            },
        },
        {
            "name": "Pool",
            "module": "finn.custom_op.fpgadataflow.pool",
            "class_name": "Pool",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferPool",
            },
        },
        {
            "name": "StreamingSplit",
            "module": "finn.custom_op.fpgadataflow.split",
            "class_name": "StreamingSplit",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferSplitLayer",
            },
        },
        {
            "name": "StreamingEltwise",
            "module": "finn.custom_op.fpgadataflow.streamingeltwise",
            "class_name": "StreamingEltwise",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferStreamingEltwise",
            },
        },
        {
            "name": "Thresholding",
            "module": "finn.custom_op.fpgadataflow.thresholding",
            "class_name": "Thresholding",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferThresholdingLayer",
            },
        },
        {
            "name": "UpsampleNearestNeighbour",
            "module": "finn.custom_op.fpgadataflow.upsampler",
            "class_name": "UpsampleNearestNeighbour",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferUpsample",
            },
        },
        {
            "name": "VVAU",
            "module": "finn.custom_op.fpgadataflow.vectorvectoractivation",
            "class_name": "VVAU",
            "infer_transform": {
                "module": "finn.transformation.fpgadataflow.convert_to_hw_layers",
                "class_name": "InferVectorVectorActivation",
            },
        },
        # Infrastructure kernels (inserted by topology transforms, not pattern matching)
        {
            "name": "StreamingFIFO",
            "module": "finn.custom_op.fpgadataflow.streamingfifo",
            "class_name": "StreamingFIFO",
            "is_infrastructure": True,  # Inserted by InsertFIFO/InsertAndSetFIFODepths
        },
        {
            "name": "StreamingDataWidthConverter",
            "module": "finn.custom_op.fpgadataflow.streamingdatawidthconverter",
            "class_name": "StreamingDataWidthConverter",
            "is_infrastructure": True,  # Inserted by InsertDWC (stream width mismatch correction)
        },
        # Note: CheckSum, TLastMarker, IODMA are legacy FINN backend-only components
        # without base kernel classes, so they cannot be registered here.
    ]


# ============================================================================
# BACKENDS: Static list for performance
# ============================================================================


def _register_backends():
    """Register FINN backends - static list for performance.

    Returns 27 backends:
    - 17 HLS implementations (16 computational + 1 infrastructure)
    - 10 RTL implementations (7 computational + 3 infrastructure)

    Infrastructure backends are for kernels inserted by topology transforms
    (StreamingFIFO, StreamingDataWidthConverter).

    Note: CheckSum_hls, TLastMarker_hls, IODMA_hls are legacy backend-only
    components and are not registered (no base kernel class).

    Note: This is a static list for fast discovery (~1ms).
    When adding new backends to FINN, add them here manually.
    """
    return [
        # HLS Backends
        {
            "name": "AddStreams_hls",
            "module": "finn.custom_op.fpgadataflow.hls.addstreams_hls",
            "class_name": "AddStreams_hls",
            "target_kernel": "finn:AddStreams",
            "language": "hls",
        },
        {
            "name": "ChannelwiseOp_hls",
            "module": "finn.custom_op.fpgadataflow.hls.channelwise_op_hls",
            "class_name": "ChannelwiseOp_hls",
            "target_kernel": "finn:ChannelwiseOp",
            "language": "hls",
        },
        {
            "name": "StreamingConcat_hls",
            "module": "finn.custom_op.fpgadataflow.hls.concat_hls",
            "class_name": "StreamingConcat_hls",
            "target_kernel": "finn:StreamingConcat",
            "language": "hls",
        },
        {
            "name": "DuplicateStreams_hls",
            "module": "finn.custom_op.fpgadataflow.hls.duplicatestreams_hls",
            "class_name": "DuplicateStreams_hls",
            "target_kernel": "finn:DuplicateStreams",
            "language": "hls",
        },
        {
            "name": "ElementwiseBinaryOperation_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseBinaryOperation_hls",
            "target_kernel": "finn:ElementwiseBinaryOperation",
            "language": "hls",
        },
        # Specialized elementwise backends
        {
            "name": "ElementwiseAdd_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseAdd_hls",
            "target_kernel": "finn:ElementwiseAdd",
            "language": "hls",
        },
        {
            "name": "ElementwiseSub_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseSub_hls",
            "target_kernel": "finn:ElementwiseSub",
            "language": "hls",
        },
        {
            "name": "ElementwiseMul_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseMul_hls",
            "target_kernel": "finn:ElementwiseMul",
            "language": "hls",
        },
        {
            "name": "ElementwiseDiv_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseDiv_hls",
            "target_kernel": "finn:ElementwiseDiv",
            "language": "hls",
        },
        {
            "name": "ElementwiseAnd_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseAnd_hls",
            "target_kernel": "finn:ElementwiseAnd",
            "language": "hls",
        },
        {
            "name": "ElementwiseOr_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseOr_hls",
            "target_kernel": "finn:ElementwiseOr",
            "language": "hls",
        },
        {
            "name": "ElementwiseXor_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseXor_hls",
            "target_kernel": "finn:ElementwiseXor",
            "language": "hls",
        },
        {
            "name": "ElementwiseEqual_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseEqual_hls",
            "target_kernel": "finn:ElementwiseEqual",
            "language": "hls",
        },
        {
            "name": "ElementwiseLess_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseLess_hls",
            "target_kernel": "finn:ElementwiseLess",
            "language": "hls",
        },
        {
            "name": "ElementwiseLessOrEqual_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseLessOrEqual_hls",
            "target_kernel": "finn:ElementwiseLessOrEqual",
            "language": "hls",
        },
        {
            "name": "ElementwiseGreater_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseGreater_hls",
            "target_kernel": "finn:ElementwiseGreater",
            "language": "hls",
        },
        {
            "name": "ElementwiseGreaterOrEqual_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseGreaterOrEqual_hls",
            "target_kernel": "finn:ElementwiseGreaterOrEqual",
            "language": "hls",
        },
        {
            "name": "ElementwiseBitwiseAnd_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseBitwiseAnd_hls",
            "target_kernel": "finn:ElementwiseBitwiseAnd",
            "language": "hls",
        },
        {
            "name": "ElementwiseBitwiseOr_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseBitwiseOr_hls",
            "target_kernel": "finn:ElementwiseBitwiseOr",
            "language": "hls",
        },
        {
            "name": "ElementwiseBitwiseXor_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseBitwiseXor_hls",
            "target_kernel": "finn:ElementwiseBitwiseXor",
            "language": "hls",
        },
        {
            "name": "ElementwiseBitShift_hls",
            "module": "finn.custom_op.fpgadataflow.hls.elementwise_binary_hls",
            "class_name": "ElementwiseBitShift_hls",
            "target_kernel": "finn:ElementwiseBitShift",
            "language": "hls",
        },
        {
            "name": "GlobalAccPool_hls",
            "module": "finn.custom_op.fpgadataflow.hls.globalaccpool_hls",
            "class_name": "GlobalAccPool_hls",
            "target_kernel": "finn:GlobalAccPool",
            "language": "hls",
        },
        {
            "name": "LabelSelect_hls",
            "module": "finn.custom_op.fpgadataflow.hls.labelselect_hls",
            "class_name": "LabelSelect_hls",
            "target_kernel": "finn:LabelSelect",
            "language": "hls",
        },
        {
            "name": "Lookup_hls",
            "module": "finn.custom_op.fpgadataflow.hls.lookup_hls",
            "class_name": "Lookup_hls",
            "target_kernel": "finn:Lookup",
            "language": "hls",
        },
        {
            "name": "MVAU_hls",
            "module": "finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls",
            "class_name": "MVAU_hls",
            "target_kernel": "finn:MVAU",
            "language": "hls",
        },
        {
            "name": "Pool_hls",
            "module": "finn.custom_op.fpgadataflow.hls.pool_hls",
            "class_name": "Pool_hls",
            "target_kernel": "finn:Pool",
            "language": "hls",
        },
        {
            "name": "StreamingSplit_hls",
            "module": "finn.custom_op.fpgadataflow.hls.split_hls",
            "class_name": "StreamingSplit_hls",
            "target_kernel": "finn:StreamingSplit",
            "language": "hls",
        },
        {
            "name": "StreamingEltwise_hls",
            "module": "finn.custom_op.fpgadataflow.hls.streamingeltwise_hls",
            "class_name": "StreamingEltwise_hls",
            "target_kernel": "finn:StreamingEltwise",
            "language": "hls",
        },
        {
            "name": "Thresholding_hls",
            "module": "finn.custom_op.fpgadataflow.hls.thresholding_hls",
            "class_name": "Thresholding_hls",
            "target_kernel": "finn:Thresholding",
            "language": "hls",
        },
        {
            "name": "UpsampleNearestNeighbour_hls",
            "module": "finn.custom_op.fpgadataflow.hls.upsampler_hls",
            "class_name": "UpsampleNearestNeighbour_hls",
            "target_kernel": "finn:UpsampleNearestNeighbour",
            "language": "hls",
        },
        {
            "name": "VVAU_hls",
            "module": "finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls",
            "class_name": "VVAU_hls",
            "target_kernel": "finn:VVAU",
            "language": "hls",
        },
        # RTL Backends
        {
            "name": "ConvolutionInputGenerator_rtl",
            "module": "finn.custom_op.fpgadataflow.rtl.convolutioninputgenerator_rtl",
            "class_name": "ConvolutionInputGenerator_rtl",
            "target_kernel": "finn:ConvolutionInputGenerator",
            "language": "rtl",
        },
        {
            "name": "MVAU_rtl",
            "module": "finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl",
            "class_name": "MVAU_rtl",
            "target_kernel": "finn:MVAU",
            "language": "rtl",
        },
        {
            "name": "Thresholding_rtl",
            "module": "finn.custom_op.fpgadataflow.rtl.thresholding_rtl",
            "class_name": "Thresholding_rtl",
            "target_kernel": "finn:Thresholding",
            "language": "rtl",
        },
        {
            "name": "VVAU_rtl",
            "module": "finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl",
            "class_name": "VVAU_rtl",
            "target_kernel": "finn:VVAU",
            "language": "rtl",
        },
        # Infrastructure kernel backends
        {
            "name": "StreamingFIFO_rtl",
            "module": "finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl",
            "class_name": "StreamingFIFO_rtl",
            "target_kernel": "finn:StreamingFIFO",
            "language": "rtl",
        },
        {
            "name": "StreamingDataWidthConverter_hls",
            "module": "finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls",
            "class_name": "StreamingDataWidthConverter_hls",
            "target_kernel": "finn:StreamingDataWidthConverter",
            "language": "hls",
        },
        {
            "name": "StreamingDataWidthConverter_rtl",
            "module": "finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl",
            "class_name": "StreamingDataWidthConverter_rtl",
            "target_kernel": "finn:StreamingDataWidthConverter",
            "language": "rtl",
        },
        # Note: CheckSum_hls, TLastMarker_hls, IODMA_hls are legacy backend-only
        # components without base kernel classes, so they're not registered.
    ]


# ============================================================================
# STEPS: Static list for performance
# ============================================================================


def _discover_steps():
    """Register FINN builder step functions - static list for performance.

    Returns 19 build pipeline steps.

    Note: This is a static list for fast discovery (~1ms).
    When adding new steps to FINN, add them here manually.
    """
    return [
        {
            "name": "qonnx_to_finn",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_qonnx_to_finn",
        },
        {
            "name": "tidy_up",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_tidy_up",
        },
        {
            "name": "streamline",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_streamline",
        },
        {
            "name": "convert_to_hw",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_convert_to_hw",
        },
        {
            "name": "create_dataflow_partition",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_create_dataflow_partition",
        },
        {
            "name": "specialize_layers",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_specialize_layers",
        },
        {
            "name": "target_fps_parallelization",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_target_fps_parallelization",
        },
        {
            "name": "apply_folding_config",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_apply_folding_config",
        },
        {
            "name": "generate_estimate_reports",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_generate_estimate_reports",
        },
        {
            "name": "minimize_bit_width",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_minimize_bit_width",
        },
        {
            "name": "hw_codegen",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_hw_codegen",
        },
        {
            "name": "hw_ipgen",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_hw_ipgen",
        },
        {
            "name": "set_fifo_depths",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_set_fifo_depths",
        },
        {
            "name": "create_stitched_ip",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_create_stitched_ip",
        },
        {
            "name": "measure_rtlsim_performance",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_measure_rtlsim_performance",
        },
        {
            "name": "make_driver",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_make_driver",
        },
        {
            "name": "out_of_context_synthesis",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_out_of_context_synthesis",
        },
        {
            "name": "synthesize_bitfile",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_synthesize_bitfile",
        },
        {
            "name": "deployment_package",
            "module": "finn.builder.build_dataflow_steps",
            "func_name": "step_deployment_package",
        },
    ]
