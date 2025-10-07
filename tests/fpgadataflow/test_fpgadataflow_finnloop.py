import pytest

import numpy as np
import onnx
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.loop_rolling import LoopExtraction, LoopRolling
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def generate_random_threshold_values(data_type, num_input_channels, num_steps):
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
    return helper.make_tensor_value_info(name, proto, shape)


def create_threshold(name, shape):
    return create_tensor_info(name, shape)


def create_node(node_type, inputs, outputs, name, extra_params={}):
    base_params = {
        "domain": "finn.custom_op.fpgadataflow.rtl"
        if "rtl" in node_type
        else "finn.custom_op.fpgadataflow.hls",
        "backend": "fpgadataflow",
        "numInputVectors": list((1, 3, 3)),
        "inFIFODepths": [2, 2],
        "name": name,
    }
    return helper.make_node(node_type, inputs, outputs, **{**base_params, **extra_params})


def make_loop_modelwrapper(
    mw,
    mh,
    dtype=DataType["INT8"],
    elemwise_optype="ElementwiseMul_hls",
    rhs_shape=[1],
    eltw_param_dtype="INT8",
    name_suffix="",
):
    elemwise_output_dtype = (
        DataType["FLOAT32"] if eltw_param_dtype == "FLOAT32" else DataType["INT32"]
    )
    thresholding_input_dtype = (
        DataType["FLOAT32"] if eltw_param_dtype == "FLOAT32" else DataType["INT32"]
    )

    # Tensor Initialization with Separate Names
    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    W1 = gen_finn_dt_tensor(dtype, (mw, mh))
    W2 = gen_finn_dt_tensor(dtype, (mh, mh))
    T0 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T1 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    T2 = np.sort(
        generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1), axis=1
    )
    
    T3_dtype = DataType["FLOAT32"] if eltw_param_dtype == "FLOAT32" else dtype
    T3 = np.sort(
        generate_random_threshold_values(
            T3_dtype,
            1,
            dtype.get_num_possible_values() - 1,
        ),
        axis=1,
    )
    EltwParam = gen_finn_dt_tensor(DataType[eltw_param_dtype], rhs_shape)

    tensor_shapes = {
        f"ifm{name_suffix}": [1, 3, 3, mw],
        f"weights{name_suffix}": [mw, mh],
        f"weights2{name_suffix}": [mh, mh],
    }
    output_shapes = {f"mm{name_suffix}": [1, 3, 3, mh], f"ofm{name_suffix}": (1, 3, 3, mh)}

    tensor_infos = {k: create_tensor_info(k, v) for k, v in tensor_shapes.items()}
    thresholds = [
        create_threshold(f"thresh{i}{name_suffix}", (1, dtype.get_num_possible_values() - 1))
        for i in range(4)
    ]

    nodes = [
        create_node(
            "DuplicateStreams_hls",
            [f"ifm{name_suffix}"],
            [f"ifm_1{name_suffix}", f"ifm_2{name_suffix}"],
            f"DuplicateStreams_hls_0{name_suffix}",
            {
                "NumChannels": mh,
                "NumOutputStreams": 2,
                "PE": 1,
                "inputDataType": dtype.name,
                "outFIFODepths": [2, 2],
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"ifm_1{name_suffix}", f"weights0{name_suffix}"],
            [f"mm0_out{name_suffix}"],
            f"MVAU_rtl_0{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": 1,
                "PE": 1,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm0_out{name_suffix}", f"thresh0{name_suffix}"],
            [f"mt0_out{name_suffix}"],
            f"Thresholding_rtl_0{name_suffix}",
            {
                "NumChannels": mh,
                "PE": 1,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"mt0_out{name_suffix}", f"weights1{name_suffix}"],
            [f"mm1_out{name_suffix}"],
            f"MVAU_rtl_1{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": 1,
                "PE": 1,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm1_out{name_suffix}", f"thresh1{name_suffix}"],
            [f"mt1_out{name_suffix}"],
            f"Thresholding_rtl_1{name_suffix}",
            {
                "NumChannels": mh,
                "PE": 1,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "MVAU_rtl",
            [f"ifm_2{name_suffix}", f"weights2{name_suffix}"],
            [f"mm2_out{name_suffix}"],
            f"MVAU_rtl_2{name_suffix}",
            {
                "MW": mw,
                "MH": mh,
                "SIMD": 1,
                "PE": 1,
                "inputDataType": "INT8",
                "weightDataType": "INT8",
                "outputDataType": "INT32",
                "ActVal": 0,
                "binaryXnorMode": 0,
                "noActivation": 1,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"mm2_out{name_suffix}", f"thresh2{name_suffix}"],
            [f"mt2_out{name_suffix}"],
            f"Thresholding_rtl_2{name_suffix}",
            {
                "NumChannels": mh,
                "PE": 1,
                "inputDataType": "INT32",
                "weightDataType": "INT33",
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
                "numSteps": dtype.get_num_possible_values() - 1,
            },
        ),
        create_node(
            "AddStreams_hls",
            [f"mt2_out{name_suffix}", f"mt1_out{name_suffix}"],
            [f"ofm{name_suffix}"],
            f"AddStreams_hls_0{name_suffix}",
            {"NumChannels": mh, "PE": 1, "inputDataTypes": [dtype.name, dtype.name]},
        ),
        create_node(
            elemwise_optype,
            [f"ofm{name_suffix}", f"mul_param{name_suffix}"],
            [f"ofm_ew{name_suffix}"],
            f"ElementwiseOp_hls_0{name_suffix}",
            {
                "lhs_shape": [1, 3, 3, mh],
                "rhs_shape": rhs_shape,
                "out_shape": [1, 3, 3, mh],
                "lhs_dtype": "INT9",
                "rhs_dtype": eltw_param_dtype,
                "out_dtype": elemwise_output_dtype.name,
            },
        ),
        create_node(
            "Thresholding_rtl",
            [f"ofm_ew{name_suffix}", f"thresh3{name_suffix}"],
            [f"ofm_final{name_suffix}"],
            f"Thresholding_rtl4{name_suffix}",
            {
                "NumChannels": mh,
                "PE": 1,
                "numSteps": dtype.get_num_possible_values() - 1,
                "inputDataType": thresholding_input_dtype.name,
                "weightDataType": thresholding_input_dtype.name,
                "outputDataType": dtype.name,
                "ActVal": int(dtype.min()),
            },
        ),
    ]

    loop_body = helper.make_graph(
        nodes=nodes,
        name=f"matmul_graph{name_suffix}",
        inputs=[tensor_infos[f"ifm{name_suffix}"]] + thresholds,
        outputs=[create_tensor_info(f"ofm_final{name_suffix}", output_shapes[f"ofm{name_suffix}"])],
        value_info=[
            create_tensor_info(name, output_shapes[f"mm{name_suffix}"])
            for name in [
                f"mm0_out{name_suffix}",
                f"mm1_out{name_suffix}",
                f"mm2_out{name_suffix}",
                f"ifm_1{name_suffix}",
                f"ifm_2{name_suffix}",
            ]
        ]
        + [
            create_tensor_info(name, output_shapes[f"ofm{name_suffix}"])
            for name in [
                f"mt0_out{name_suffix}",
                f"mt1_out{name_suffix}",
                f"mt2_out{name_suffix}",
                f"ofm{name_suffix}",
                f"ofm_ew{name_suffix}",
            ]
        ],
    )

    loop_body_model = qonnx_make_model(loop_body, producer_name=f"loop-body-model{name_suffix}")
    loop_body_model = ModelWrapper(loop_body_model)

    # Set initializers using generated values
    loop_body_model.set_initializer(f"weights0{name_suffix}", W0)
    loop_body_model.set_initializer(f"weights1{name_suffix}", W1)
    loop_body_model.set_initializer(f"weights2{name_suffix}", W2)
    loop_body_model.set_initializer(f"thresh0{name_suffix}", T0)
    loop_body_model.set_initializer(f"thresh1{name_suffix}", T1)
    loop_body_model.set_initializer(f"thresh2{name_suffix}", T2)
    loop_body_model.set_initializer(f"thresh3{name_suffix}", T3)
    loop_body_model.set_initializer(f"mul_param{name_suffix}", EltwParam)

    # Set tensor datatypes
    tensors = [
        f"weights0{name_suffix}",
        f"weights1{name_suffix}",
        f"weights2{name_suffix}",
        f"thresh0{name_suffix}",
        f"thresh1{name_suffix}",
        f"thresh2{name_suffix}",
        f"ifm{name_suffix}",
        f"ofm_final{name_suffix}",
    ]
    for tensor in tensors:
        loop_body_model.set_tensor_datatype(tensor, dtype)

    loop_body_model.set_tensor_datatype(f"thresh3{name_suffix}", T3_dtype)
    loop_body_model.set_tensor_datatype(f"mul_param{name_suffix}", DataType[eltw_param_dtype])

    return loop_body_model


def create_chained_loop_bodies(
    mw, mh, num_copies, elemwise_optype="ElementwiseMul_hls", rhs_shape=[1], eltw_param_dtype="INT8"
):
    loop_body_models = []

    # Create multiple instances of the loop body with unique name_suffix
    for i in range(num_copies):
        name_suffix = f"_{i}"
        loop_body_model = make_loop_modelwrapper(
            mw=mw,
            mh=mh,
            dtype=DataType["INT8"],
            elemwise_optype=elemwise_optype,
            rhs_shape=rhs_shape,
            eltw_param_dtype=eltw_param_dtype,
            name_suffix=name_suffix,
        )
        for node in loop_body_model.graph.node:
            node.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="pkg.torch.onnx.name_scopes", value=f"['', 'layers.{i}']"
                )
            )
            node.metadata_props.append(
                onnx.StringStringEntryProto(
                    key="pkg.torch.onnx.class_hierarchy", value=f"['TestModule', '{node.name}']"
                )
            )

        loop_body_models.append(loop_body_model)

    return loop_body_models


# dimensions
@pytest.mark.parametrize("dim", [16])
# iteration count, number of models chained together
@pytest.mark.parametrize("iteration", [1, 3])
# elementwise operation
@pytest.mark.parametrize("elemwise_optype", ["ElementwiseMul_hls", "ElementwiseAdd_hls"])
# elementwise shape
@pytest.mark.parametrize("rhs_shape", [[1], [16]])
# eltwise param dtype
@pytest.mark.parametrize("eltw_param_dtype", ["INT8", "FLOAT32"])
@pytest.mark.fpgataflow
def test_fpgadataflow_finnloop(dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype):
    loop_body_models = create_chained_loop_bodies(
        dim, dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype
    )
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        model = model.transform(MergeONNXModels(m))

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # cppsim preparation
    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))
    model.save("fpgadataflow_finnloop.onnx")
    # generate reference io pair
    x = gen_finn_dt_tensor(DataType["INT8"], (1, 3, 3, dim))
    io_dict = {model.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model, io_dict)
    y_ref = y_dict[model.graph.output[0].name]

    # loop extraction and rolling
    loop_extraction = LoopExtraction(hierarchy_list=["", "layers.0"])
    model = model.transform(loop_extraction)

    assert (
        len(model.get_nodes_by_op_type("fn_loop-body")) == iteration
    ), "Loop extraction did not find expected number of loop bodies"

    model = model.transform(LoopRolling(loop_extraction.loop_body_template))

    y_dict = oxe.execute_onnx(model, io_dict)
    y_prod = y_dict[model.graph.output[0].name]
    assert (y_prod == y_ref).all()
