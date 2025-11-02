############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import tempfile
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from onnx import TensorProto, helper

# Import the classes we're testing
from finn.transformation.fpgadataflow.specialize_kernel import SpecializeKernel
from finn.builder.build_dataflow_steps import step_specialize_layers
from finn.builder.build_dataflow_config import DataflowBuildConfig

# Import kernel classes for testing
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import StreamingDataWidthConverter
from finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls import (
    StreamingDataWidthConverter_hls,
)
from finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl import (
    StreamingDataWidthConverter_rtl,
)
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend


def make_test_mvau_model(backend="fpgadataflow"):
    """Create a simple MVAU model for testing."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 4])

    # Create MVAU node with backend attribute
    mvau_node = helper.make_node(
        "MVAU",
        ["inp", "weights"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend=backend,
        MW=4,
        MH=4,
        SIMD=4,
        PE=4,
        inputDataType="INT4",
        weightDataType="INT4",
        outputDataType="INT4",
        ActVal=0,
        noActivation=1,
        binaryXnorMode=0,
        numInputVectors=[1],
    )

    # Create weight tensor
    weights = gen_finn_dt_tensor(DataType["INT4"], (4, 4))
    weight_tensor = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        weights.shape,
        weights.flatten().astype(float).tolist(),
    )

    graph = helper.make_graph(
        nodes=[mvau_node],
        name="test_mvau",
        inputs=[inp],
        outputs=[outp],
        value_info=[],
        initializer=[weight_tensor],
    )

    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)
    return model


def make_test_dwc_model(in_width, out_width, backend="fpgadataflow"):
    """Create a simple DWC model for testing."""
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, in_width])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, out_width])

    dwc_node = helper.make_node(
        "StreamingDataWidthConverter",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend=backend,
        inWidth=in_width,
        outWidth=out_width,
        dataType="INT8",
        numInputVectors=[1],
    )

    graph = helper.make_graph(
        nodes=[dwc_node],
        name="test_dwc",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)
    return model


def test_class_metadata_extraction():
    """Test that we can extract correct metadata from variant classes."""
    print("\n=== Test 1: Class Metadata Extraction ===")

    # Test MVAU_hls
    assert MVAU_hls.__name__ == "MVAU_hls"
    assert "hls" in MVAU_hls.__module__
    print(f"✓ MVAU_hls: name={MVAU_hls.__name__}, module={MVAU_hls.__module__}")

    # Test MVAU_rtl
    assert MVAU_rtl.__name__ == "MVAU_rtl"
    assert "rtl" in MVAU_rtl.__module__
    print(f"✓ MVAU_rtl: name={MVAU_rtl.__name__}, module={MVAU_rtl.__module__}")

    # Test backend suffix extraction
    for variant_class in [MVAU_hls, MVAU_rtl]:
        variant_name = variant_class.__name__
        if "_" in variant_name:
            backend_style = variant_name.split("_")[-1]
            assert backend_style in ["hls", "rtl"]
            print(f"✓ Extracted backend style '{backend_style}' from {variant_name}")

    print("✓ All metadata extraction tests passed")


def test_mvau_rtl_specialization():
    """Test MVAU specialization to RTL (should succeed with INT4 weights)."""
    print("\n=== Test 2: MVAU RTL Specialization (should succeed) ===")

    model = make_test_mvau_model(backend="fpgadataflow")

    # Apply specialization with RTL first
    transform = SpecializeKernel(
        kernel_class=MVAU,
        backend_variants=[MVAU_rtl, MVAU_hls],
        fpgapart="xczu3eg-sbva484-1-e",  # Zynq UltraScale+ with DSP48E2
    )

    model_transformed, modified = transform.apply(model)

    assert modified, "Model should have been modified"

    # Check the node was specialized to RTL
    node = model_transformed.graph.node[0]
    assert node.op_type == "MVAU_rtl", f"Expected MVAU_rtl, got {node.op_type}"

    # Check backend attribute
    from qonnx.util.basic import get_by_name
    backend_attr = get_by_name(node.attribute, "backend")
    backend_value = backend_attr.s.decode("UTF-8")
    assert backend_value == "rtl", f"Expected backend='rtl', got '{backend_value}'"

    print(f"✓ MVAU specialized to RTL: op_type={node.op_type}, backend={backend_value}")
    print("✓ RTL specialization test passed")


def test_mvau_hls_fallback():
    """Test MVAU fallback to HLS (when RTL constraints not met)."""
    print("\n=== Test 3: MVAU HLS Fallback ===")

    # Create MVAU with 1-bit weights (RTL doesn't support < 2 bits)
    model = make_test_mvau_model(backend="fpgadataflow")

    # Modify to use 1-bit weights (should force HLS fallback)
    node = model.graph.node[0]
    from qonnx.util.basic import get_by_name
    wdt_attr = get_by_name(node.attribute, "weightDataType")
    wdt_attr.s = "BIPOLAR".encode("UTF-8")

    # Apply specialization
    transform = SpecializeKernel(
        kernel_class=MVAU,
        backend_variants=[MVAU_rtl, MVAU_hls],  # Try RTL first
        fpgapart="xczu3eg-sbva484-1-e",
    )

    model_transformed, modified = transform.apply(model)

    assert modified, "Model should have been modified"

    # Check the node was specialized to HLS (fallback from RTL)
    node = model_transformed.graph.node[0]
    assert node.op_type == "MVAU_hls", f"Expected MVAU_hls fallback, got {node.op_type}"

    backend_attr = get_by_name(node.attribute, "backend")
    backend_value = backend_attr.s.decode("UTF-8")
    assert backend_value == "hls", f"Expected backend='hls', got '{backend_value}'"

    print(f"✓ MVAU fell back to HLS: op_type={node.op_type}, backend={backend_value}")
    print("✓ HLS fallback test passed")


def test_dwc_rtl_specialization():
    """Test DWC specialization to RTL (with integer width ratio)."""
    print("\n=== Test 4: DWC RTL Specialization ===")

    # Create DWC with integer width ratio (64 -> 32)
    model = make_test_dwc_model(in_width=64, out_width=32, backend="fpgadataflow")

    transform = SpecializeKernel(
        kernel_class=StreamingDataWidthConverter,
        backend_variants=[StreamingDataWidthConverter_rtl, StreamingDataWidthConverter_hls],
        fpgapart="xczu3eg-sbva484-1-e",
    )

    model_transformed, modified = transform.apply(model)

    assert modified, "Model should have been modified"

    node = model_transformed.graph.node[0]
    assert node.op_type == "StreamingDataWidthConverter_rtl"

    from qonnx.util.basic import get_by_name
    backend_attr = get_by_name(node.attribute, "backend")
    backend_value = backend_attr.s.decode("UTF-8")
    assert backend_value == "rtl"

    print(f"✓ DWC specialized to RTL: op_type={node.op_type}, backend={backend_value}")
    print("✓ DWC RTL specialization test passed")


def test_kernel_name_filtering():
    """Test that only matching kernel names are processed."""
    print("\n=== Test 5: Kernel Name Filtering ===")

    # Create model with MVAU
    model = make_test_mvau_model(backend="fpgadataflow")

    # Try to specialize with DWC classes (should not match)
    transform = SpecializeKernel(
        kernel_class=StreamingDataWidthConverter,  # Wrong kernel class
        backend_variants=[StreamingDataWidthConverter_rtl],
        fpgapart="xczu3eg-sbva484-1-e",
    )

    model_transformed, modified = transform.apply(model)

    assert not modified, "Model should NOT have been modified (wrong kernel class)"

    # Node should remain unchanged
    node = model_transformed.graph.node[0]
    assert node.op_type == "MVAU", "Original op_type should be preserved"

    from qonnx.util.basic import get_by_name
    backend_attr = get_by_name(node.attribute, "backend")
    backend_value = backend_attr.s.decode("UTF-8")
    assert backend_value == "fpgadataflow", "Original backend should be preserved"

    print(f"✓ MVAU node not modified by DWC transform: op_type={node.op_type}")
    print("✓ Kernel name filtering test passed")


def test_build_step_integration():
    """Test step_specialize_layers with class-based config."""
    print("\n=== Test 6: Build Step Integration ===")

    # Create model with MVAU
    model = make_test_mvau_model(backend="fpgadataflow")

    # Create config with class-based kernel_selections
    with tempfile.TemporaryDirectory() as tmpdir:
        from finn.builder.build_dataflow_config import DataflowOutputType

        cfg = DataflowBuildConfig(
            output_dir=tmpdir,
            board="ZCU104",
            synth_clk_period_ns=5.0,
            generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
            kernel_selections=[
                (MVAU, [MVAU_rtl, MVAU_hls]),
            ],
        )

        # Apply the build step
        model_transformed = step_specialize_layers(model, cfg)

        # Check specialization occurred
        node = model_transformed.graph.node[0]
        assert node.op_type == "MVAU_rtl", f"Expected MVAU_rtl, got {node.op_type}"

        from qonnx.util.basic import get_by_name
        backend_attr = get_by_name(node.attribute, "backend")
        backend_value = backend_attr.s.decode("UTF-8")
        assert backend_value == "rtl"

        print(f"✓ Build step applied: op_type={node.op_type}, backend={backend_value}")
        print("✓ Build step integration test passed")


def test_already_specialized_nodes():
    """Test that already-specialized nodes are skipped."""
    print("\n=== Test 7: Skip Already-Specialized Nodes ===")

    # Create model with already-specialized node
    model = make_test_mvau_model(backend="hls")
    model.graph.node[0].op_type = "MVAU_hls"

    transform = SpecializeKernel(
        kernel_class=MVAU,
        backend_variants=[MVAU_rtl, MVAU_hls],
        fpgapart="xczu3eg-sbva484-1-e",
    )

    model_transformed, modified = transform.apply(model)

    assert not modified, "Already-specialized node should not be re-specialized"

    node = model_transformed.graph.node[0]
    assert node.op_type == "MVAU_hls", "Op type should remain MVAU_hls"

    from qonnx.util.basic import get_by_name
    backend_attr = get_by_name(node.attribute, "backend")
    backend_value = backend_attr.s.decode("UTF-8")
    assert backend_value == "hls", "Backend should remain 'hls'"

    print(f"✓ Already-specialized node skipped: op_type={node.op_type}, backend={backend_value}")
    print("✓ Skip already-specialized test passed")


def test_sequential_specialization():
    """Test sequential specialization: explicit then automatic."""
    print("\n=== Test 8: Sequential Specialization ===")

    # Create model with MVAU and DWC
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 64])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 32])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 32])

    # MVAU node
    mvau_node = helper.make_node(
        "MVAU",
        ["inp", "weights"],
        ["mid"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        MW=64,
        MH=32,
        SIMD=64,
        PE=32,
        inputDataType="INT4",
        weightDataType="INT4",
        outputDataType="INT4",
        ActVal=0,
        noActivation=1,
        binaryXnorMode=0,
        numInputVectors=[1],
    )

    # DWC node
    dwc_node = helper.make_node(
        "StreamingDataWidthConverter",
        ["mid"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inWidth=32,
        outWidth=32,
        dataType="INT8",
        shape=[1, 32],
        numInputVectors=[1],
    )

    weights = gen_finn_dt_tensor(DataType["INT4"], (64, 32))
    weight_tensor = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        weights.shape,
        weights.flatten().astype(float).tolist(),
    )

    graph = helper.make_graph(
        nodes=[mvau_node, dwc_node],
        name="test_sequential",
        inputs=[inp],
        outputs=[outp],
        value_info=[mid],
        initializer=[weight_tensor],
    )

    model = qonnx_make_model(graph, producer_name="test")
    model = ModelWrapper(model)

    # Test sequential specialization
    with tempfile.TemporaryDirectory() as tmpdir:
        from finn.builder.build_dataflow_config import DataflowOutputType

        # Specify explicit priority for MVAU only
        # DWC should be specialized automatically
        cfg = DataflowBuildConfig(
            output_dir=tmpdir,
            board="ZCU104",
            synth_clk_period_ns=5.0,
            generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
            kernel_selections=[
                (MVAU, [MVAU_rtl, MVAU_hls]),  # Explicit for MVAU
                # DWC not specified - should use automatic
            ],
        )

        model_transformed = step_specialize_layers(model, cfg)

        # Check MVAU specialized via explicit (RTL)
        mvau_node = model_transformed.graph.node[0]
        assert mvau_node.op_type == "MVAU_rtl", f"Expected MVAU_rtl, got {mvau_node.op_type}"
        print(f"✓ MVAU specialized via explicit selection: {mvau_node.op_type}")

        # Check DWC specialized via automatic (should be RTL due to integer width ratio)
        dwc_node = model_transformed.graph.node[1]
        assert dwc_node.op_type == "StreamingDataWidthConverter_rtl", f"Expected DWC_rtl, got {dwc_node.op_type}"
        print(f"✓ DWC specialized via automatic selection: {dwc_node.op_type}")

    print("✓ Sequential specialization test passed")


def test_backend_style_detection():
    """Test backend style detection with various naming patterns."""
    print("\n=== Test 9: Backend Style Detection ===")

    from finn.transformation.fpgadataflow.specialize_kernel import SpecializeKernel

    # Create mock classes for testing (these won't be in registry, but we're just testing detection)
    class MVAU_rtl_v2(MVAU, RTLBackend):
        """RTL variant with version suffix."""
        pass

    class CustomKernel_optimized(MVAU, HLSBackend):
        """HLS variant without standard suffix."""
        pass

    # Test suffix detection
    assert MVAU_rtl.__name__.endswith("_rtl"), "Standard RTL naming"
    assert MVAU_hls.__name__.endswith("_hls"), "Standard HLS naming"

    # Test multi-underscore handling - should use inheritance, not split("_")[-1]
    print(f"✓ MVAU_rtl_v2 class name: {MVAU_rtl_v2.__name__}")
    assert issubclass(MVAU_rtl_v2, RTLBackend), "MVAU_rtl_v2 should inherit RTLBackend"

    print(f"✓ CustomKernel_optimized class name: {CustomKernel_optimized.__name__}")
    assert issubclass(CustomKernel_optimized, HLSBackend), "CustomKernel_optimized should inherit HLSBackend"

    print("✓ Backend style detection test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Class-Based Kernel Specialization")
    print("="*60)

    try:
        test_class_metadata_extraction()
        test_mvau_rtl_specialization()
        test_mvau_hls_fallback()
        test_dwc_rtl_specialization()
        test_kernel_name_filtering()
        test_build_step_integration()
        test_already_specialized_nodes()
        test_sequential_specialization()
        test_backend_style_detection()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
