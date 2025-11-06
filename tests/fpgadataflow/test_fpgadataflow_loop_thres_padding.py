import pytest
from finn.util.basic import make_build_dir
import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow as build
import numpy as np
import onnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

verif_steps = [
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]

fpga_part = "xcvc1902-vsva2197-2MP-e-S"
clk_ns = 5

def generate_random_threshold_values(data_type, num_input_channels, num_steps):
    """Generate random threshold values for thresholding layers."""
    if data_type.is_integer():
        return np.random.randint(
            data_type.min(),
            data_type.max() + 1,
            (num_input_channels, num_steps),
        ).astype(np.float32)
    else:
        return (np.random.randn(num_input_channels, num_steps) * 1000).astype(
            data_type.to_numpy_dt()
        )


def create_tensor_info(name, shape, proto=TensorProto.FLOAT):
    """Create tensor value info for ONNX graph."""
    return helper.make_tensor_value_info(name, proto, shape)


def create_node(node_type, inputs, outputs, name, extra_params={}):
    """Create an ONNX node with FINN-specific attributes."""
    base_params = {
        "domain": "finn.custom_op.fpgadataflow.rtl"
        if "rtl" in node_type
        else "finn.custom_op.fpgadataflow.hls",
        "backend": "fpgadataflow",
        "name": name,
    }
    return helper.make_node(node_type, inputs, outputs, **{**base_params, **extra_params})


def make_transformer_loop_model(dtype=DataType["INT8"], name_suffix=""):
    """
    Create a transformer-style loop body with:
    - Input: 1x1x1x384 (FINN requires 4D tensors)
    - MVAU: 384x1536 weights (INT8 -> INT32)
    - Thresholding: 1536x255 weights (INT32 -> INT8)
    - MVAU: 1536x384 weights (INT8 -> INT32)
    - Final Thresholding: 384x255 weights (INT32 -> INT8)
    - Output: 1x1x1x384 (INT8)
    """

    # Dimensions - using FINN conventions
    d_model = 384  # mw in FINN terms
    d_ff = 1536    # mh in FINN terms
    num_thresh_steps = dtype.get_num_possible_values() - 1  # Use valid threshold steps

    # Generate weight tensors
    W1 = gen_finn_dt_tensor(dtype, (d_model, d_ff))  # 384x1536 for first MVAU
    W2 = gen_finn_dt_tensor(dtype, (d_ff, d_model))  # 1536x384 for second MVAU

    # Generate threshold values (1536 channels, proper num_steps)
    T1 = np.sort(
        generate_random_threshold_values(dtype, d_ff, num_thresh_steps), axis=1
    )

    # Generate threshold values for final output (384 channels, proper num_steps)
    T2 = np.sort(
        generate_random_threshold_values(dtype, d_model, num_thresh_steps), axis=1
    )

    # Define tensor shapes - FINN uses 4D tensors [batch, height, width, channels]
    input_shape = [1, 1, 1, d_model]  # 1x1x1x384
    hidden_shape = [1, 1, 1, d_ff]    # 1x1x1x1536
    output_shape = [1, 1, 1, d_model] # 1x1x1x384

    # Create tensor infos
    tensor_infos = {
        f"input{name_suffix}": create_tensor_info(f"input{name_suffix}", input_shape),
        f"weights1{name_suffix}": create_tensor_info(f"weights1{name_suffix}", [d_model, d_ff]),
        f"weights2{name_suffix}": create_tensor_info(f"weights2{name_suffix}", [d_ff, d_model]),
        f"thresh1{name_suffix}": create_tensor_info(f"thresh1{name_suffix}", [d_ff, num_thresh_steps]),
        f"thresh2{name_suffix}": create_tensor_info(f"thresh2{name_suffix}", [d_model, num_thresh_steps]),
    }

    # Create nodes
    nodes = [
        # First MVAU: 384 -> 1536
        create_node(
            "MVAU_rtl",
            [f"input{name_suffix}", f"weights1{name_suffix}"],
            [f"mvau1_out{name_suffix}"],
            f"MVAU_rtl_1{name_suffix}",
            {
                "MW": d_model,      # 384
                "MH": d_ff,         # 1536
                "SIMD": 8,          # Parallelism factor for input
                "PE": 16,           # Parallelism factor for output
                "inputDataType": dtype.name,
                "weightDataType": dtype.name,
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),

        # Thresholding: 1536 channels with 255 threshold steps
        create_node(
            "Thresholding_rtl",
            [f"mvau1_out{name_suffix}", f"thresh1{name_suffix}"],
            [f"thresh1_out{name_suffix}"],
            f"Thresholding_rtl_1{name_suffix}",
            {
                "NumChannels": d_ff,        # 1536
                "PE": 16,                   # Parallelism factor
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": num_thresh_steps,  # 255
            },
        ),

        # Second MVAU: 1536 -> 384
        create_node(
            "MVAU_rtl",
            [f"thresh1_out{name_suffix}", f"weights2{name_suffix}"],
            [f"mvau2_out{name_suffix}"],
            f"MVAU_rtl_2{name_suffix}",
            {
                "MW": d_ff,         # 1536
                "MH": d_model,      # 384
                "SIMD": 16,         # Parallelism factor for input
                "PE": 8,            # Parallelism factor for output
                "inputDataType": dtype.name,
                "weightDataType": dtype.name,
                "outputDataType": "INT32",  # Output as INT32 for thresholding
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),

        # Final Thresholding: 384 channels, INT32 -> INT8
        create_node(
            "Thresholding_rtl",
            [f"mvau2_out{name_suffix}", f"thresh2{name_suffix}"],
            [f"output{name_suffix}"],
            f"Thresholding_rtl_2{name_suffix}",
            {
                "NumChannels": d_model,     # 384
                "PE": 8,                    # Parallelism factor
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,  # Final output as INT8
                "ActVal": int(dtype.min()),
                "numSteps": num_thresh_steps,  # 255
            },
        ),
    ]

    # Create the loop body graph
    loop_body = helper.make_graph(
        nodes=nodes,
        name=f"transformer_loop_graph{name_suffix}",
        inputs=[
            tensor_infos[f"input{name_suffix}"],
            tensor_infos[f"thresh1{name_suffix}"],
            tensor_infos[f"thresh2{name_suffix}"],
        ],
        outputs=[
            create_tensor_info(f"output{name_suffix}", output_shape)  # Final output as INT8
        ],
        value_info=[
            create_tensor_info(f"mvau1_out{name_suffix}", hidden_shape, TensorProto.INT32),
            create_tensor_info(f"thresh1_out{name_suffix}", hidden_shape),
            create_tensor_info(f"mvau2_out{name_suffix}", output_shape, TensorProto.INT32),
        ],
    )

    # Create model wrapper
    loop_body_model = qonnx_make_model(
        loop_body, producer_name=f"transformer-loop-body{name_suffix}"
    )
    loop_body_model = ModelWrapper(loop_body_model)

    # Set initializers
    loop_body_model.set_initializer(f"weights1{name_suffix}", W1)
    loop_body_model.set_initializer(f"weights2{name_suffix}", W2)
    loop_body_model.set_initializer(f"thresh1{name_suffix}", T1)
    loop_body_model.set_initializer(f"thresh2{name_suffix}", T2)

    # Set tensor datatypes
    loop_body_model.set_tensor_datatype(f"weights1{name_suffix}", dtype)
    loop_body_model.set_tensor_datatype(f"weights2{name_suffix}", dtype)
    loop_body_model.set_tensor_datatype(f"thresh1{name_suffix}", dtype)
    loop_body_model.set_tensor_datatype(f"thresh2{name_suffix}", dtype)
    loop_body_model.set_tensor_datatype(f"input{name_suffix}", dtype)
    loop_body_model.set_tensor_datatype(f"output{name_suffix}", dtype)

    return loop_body_model


def create_chained_transformer_loop_bodies(num_copies, dtype=DataType["INT8"]):
    """Create multiple instances of the transformer loop body and chain them together."""
    loop_body_models = []

    # Create multiple instances of the loop body with unique name_suffix
    for i in range(num_copies):
        name_suffix = f"_{i}"
        loop_body_model = make_transformer_loop_model(
            dtype=dtype,
            name_suffix=name_suffix,
        )

        # Add metadata for hierarchy identification (similar to test_fpgadataflow_finnloop)
        for node in loop_body_model.graph.node:
            node.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="pkg.torch.onnx.name_scopes",
                    value=f"['', 'transformer.layers.{num_copies-(i+1)}']"
                )
            )
            node.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="pkg.torch.onnx.class_hierarchy",
                    value=f"['TransformerModule', '{node.name}']"
                )
            )

        loop_body_models.append(loop_body_model)

    return loop_body_models


def test_transformer_loop_model():
    """Test the transformer-style loop model creation and execution with chained models."""

    # Create 3 chained transformer loop bodies and merge them (similar to test_fpgadataflow_finnloop)
    num_layers = 3
    loop_body_models = create_chained_transformer_loop_bodies(num_layers, dtype=DataType["INT8"])

    # Start with the first model and merge the others
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    # Apply cleanup transformations
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Prepare for cppsim execution
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))

    # Generate test input: 1x1x1x384 (FINN 4D tensor format)
    x = gen_finn_dt_tensor(DataType["INT8"], (1, 1, 1, 384))
    io_dict = {model.graph.input[0].name: x}

    # Execute model
    y_dict = oxe.execute_onnx(model, io_dict)
    y_ref = y_dict[model.graph.output[0].name]

    # Verify output shape
    assert y_ref.shape == (1, 1, 1, 384), f"Expected output shape (1, 1, 1, 384), got {y_ref.shape}"


    # Save the model
    model.save("transformer_loop_model_chained.onnx")

    print(f"Chained model created successfully with {num_layers} layers!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_ref.shape}")
    print(f"Model saved as 'transformer_loop_model_chained.onnx'")

    return model, x, y_ref


def test_transformer_loop_with_rolling():
    """Test the transformer loop model with loop extraction and rolling."""

    # Create multiple loop bodies and chain them using the helper function
    num_layers = 3
    loop_body_models = create_chained_transformer_loop_bodies(num_layers, dtype=DataType["INT8"])

    # Start with the first model and merge the others
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    # Apply transformations
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Prepare for cppsim
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))

    # Test execution
    x = gen_finn_dt_tensor(DataType["INT8"], (1, 1, 1, 384))
    io_dict = {model.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model, io_dict)
    y_ref = y_dict[model.graph.output[0].name]

    print(f"Loop rolling test completed!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_ref.shape}")

    tmp_output_dir = make_build_dir("build_mlo")

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    # Save model before loop transformations
    model.save(tmp_output_dir + "/transformer_model_before_loop_rolling.onnx")
    steps = [
        # "step_qonnx_to_finn",
        # "step_tidy_up",
        # "step_streamline",
        # "step_convert_to_hw",
        "step_create_dataflow_partition",
        # "step_specialize_layers",
        "step_loop_rolling",
        # "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_make_driver",
        "step_deployment_package",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        loop_body_hierarchy=[["", "transformer.layers.0"]],
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        verify_save_rtlsim_waveforms=True,
        # stitched_ip_gen_dcp=True,
        generate_outputs=[
            # build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            # build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/transformer_model_before_loop_rolling.onnx", cfg)


    # # Apply loop extraction (this would extract loops based on hierarchy)
    # # Note: This requires the model to have the right structure for loop extraction
    # try:
    #     loop_extraction = LoopExtraction(hierarchy_list=[["", "transformer.layers.0"]])
    #     model = model.transform(loop_extraction)

    #     # Apply loop rolling
    #     model = model.transform(LoopRolling(loop_extraction.loop_body_template))

    #     print("Loop extraction and rolling applied successfully!")
    #     model.save("transformer_model_after_loop_rolling.onnx")

    # except Exception as e:
    #     print(f"Loop transformation not applicable yet: {e}")
    #     print("This is expected as the model structure needs to be set up properly for loop extraction.")

    return model, x, y_ref


def test_single_transformer_loop_model():
    """Test a single transformer loop model (not chained)."""

    # Create single model
    model = make_transformer_loop_model(dtype=DataType["INT8"])

    # Apply cleanup transformations
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Prepare for cppsim execution
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))

    # Generate test input: 1x1x1x384 (FINN 4D tensor format)
    x = gen_finn_dt_tensor(DataType["INT8"], (1, 1, 1, 384))
    io_dict = {model.graph.input[0].name: x}

    # Execute model
    y_dict = oxe.execute_onnx(model, io_dict)
    y_ref = y_dict[model.graph.output[0].name]

    # Verify output shape
    assert y_ref.shape == (1, 1, 1, 384), f"Expected output shape (1, 1, 1, 384), got {y_ref.shape}"

    # Save the model
    model.save("transformer_loop_model_single.onnx")

    print(f"Single model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y_ref.shape}")
    print(f"Model saved as 'transformer_loop_model_single.onnx'")

    return model, x, y_ref


if __name__ == "__main__":
    # Run the single model test first
    #print("Testing single transformer loop model...")
    #model_single, x_single, y_single = test_single_transformer_loop_model()

    #print("\n" + "="*50 + "\n")

    # Run the chained model test
    #print("Testing chained transformer loop model...")
    #model, x, y = test_transformer_loop_model()

    #print("\n" + "="*50 + "\n")

    # Run the loop rolling test
    print("Testing transformer loop model with rolling...")
    model_rolled, x_rolled, y_rolled = test_transformer_loop_with_rolling()
