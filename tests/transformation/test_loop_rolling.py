import onnx
import onnxscript
import torch
import pytest
import onnxruntime as ort
import qonnx.core.modelwrapper
import qonnx.util.basic as util
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

import brevitas.onnx as bo

from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import ConvertDivToMul
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from finn.transformation.fpgadataflow.loop_rolling import LoopRolling, LoopExtraction
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int8Bias
from brevitas.nn import QuantLinear



class SimpleSubModule(torch.nn.Module):
    def __init__(self, in_features, out_features, mul_val=200):
        super(SimpleSubModule, self).__init__()
        #self.mul_val = torch.tensor([mul_val])
        self.linear = QuantLinear(in_features, out_features, bias=True,
                                  weight_quant=Int8WeightPerTensorFloat,
                                  input_quant=Int8ActPerTensorFloat,
                                  bias_quant=Int8Bias)

    def forward(self, x):
        #return self.mul_val * self.linear(x)
        return self.linear(x)


# Simple Torch Module with parameterizable number of linear layers
class SimpleModule(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=4, mul_val=200, output_size=None):
        super(SimpleModule, self).__init__()
        self.mul_val = mul_val

        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()

        # Create the linear layers
        for i in range(num_layers):
            self.layers.append(SimpleSubModule(input_size, hidden_size, mul_val=self.mul_val))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# export the model to ONNX format using dynamo
def export_model_to_onnx(input_size=10, hidden_size=20, num_layers=4, output_size=None):
    #model = SimpleSubModule(input_size, hidden_size, mul_val=150)
    model = SimpleModule(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, output_size=output_size, mul_val=150)
    model.eval()
    import numpy as np
    # Create a dummy input tensor
    dummy_input = torch.randn(input_size, hidden_size)

    # Export the model to ONNX format
    onnx_path = f"simple_module_{num_layers}layers.onnx"
    with torch.no_grad():
        bo.export_qonnx(
            model,
            (dummy_input),
            onnx_path,
            do_constant_folding=True,
            input_names=['x'],
            opset_version=18,
            dynamo=True,
            optimize=True
        )
    #torch.onnx.export(model, dummy_input, onnx_path, opset_version=18, dynamo=True)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Add finn_datatype Quantization Annotation for all tensors
    #onnx_ir = onnxscript.ir.serde.deserialize_model(onnx_model)
    # for node in onnx_ir.graph._nodes:
    #     for tensor in node.inputs + node.outputs:
    #         if "finn_datatype" not in tensor.meta.get("quant_parameter_tensor_names", {}):
    #             tensor.meta["quant_parameter_tensor_names"] = {"finn_datatype": "FLOAT32"}

    # for node in onnx_ir.graph._nodes:
    #     if 'pkg.torch.onnx.name_scopes' not in node.metadata_props:
    #         if node.name.startswith("/"):
    #             node.metadata_props['pkg.torch.onnx.name_scopes'] = str(node.name.split("/"))
    #     if 'pkg.torch.onnx.class_hierarchy' not in node.metadata_props:
    #         if node.name.startswith("/"):
    #             node.metadata_props['pkg.torch.onnx.class_hierarchy'] = str(node.name.split("/"))

    #onnx_model = onnxscript.ir.serde.serialize_model(onnx_ir)
    #onnx.save(onnx_model, onnx_path)

    # Load the ONNX model to verify
    print(f"Model exported successfully to {onnx_path}")
    print(f"Model has {num_layers} layers with input size {input_size}")
    return onnx_path

def check_tensor_shape(model_wrapper, name, expected_shape):
    actual_shape = model_wrapper.get_tensor_shape(name)
    assert actual_shape == expected_shape, f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}"


def test_finn_loop():
    input_size = 20
    hidden_size = 20
    num_layers = 6
    output_size = None

    onnx_path = export_model_to_onnx(input_size, hidden_size, num_layers, output_size)

    model_wrapper = qonnx.core.modelwrapper.ModelWrapper(onnx_path)


    model_wrapper.save('before_loop_extraction_metadata.onnx')
    m_input_dt = model_wrapper.get_tensor_datatype(model_wrapper.model.graph.input[0].name)
    m_output_dt = model_wrapper.get_tensor_datatype(model_wrapper.model.graph.output[0].name)

    model_wrapper = model_wrapper.transform(FoldConstants())
    model_wrapper = model_wrapper.transform(ConvertDivToMul())
    model_wrapper.save('before_loop_extraction_div_to_mul.onnx')
    model_wrapper = model_wrapper.transform(ConvertQONNXtoFINN())
    import finn.transformation.streamline as absorb
    import finn.transformation.streamline.reorder as reorder
    from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
    from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
    model_wrapper = model_wrapper.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model_wrapper = model_wrapper.transform(absorb.AbsorbAddIntoMultiThreshold())
    model_wrapper = model_wrapper.transform(absorb.AbsorbMulIntoMultiThreshold())
    model_wrapper = model_wrapper.transform(RoundAndClipThresholds())
    model_wrapper = model_wrapper.transform(reorder.MoveOpPastFork(["Mul"]))
    model_wrapper = model_wrapper.transform(reorder.MoveScalarMulPastMatMul())
    model_wrapper = model_wrapper.transform(reorder.MoveScalarLinearPastInvariants())
    model_wrapper = model_wrapper.transform(absorb.AbsorbMulIntoMultiThreshold())
    model_wrapper = model_wrapper.transform(absorb.AbsorbAddIntoMultiThreshold())
    model_wrapper = model_wrapper.transform(CollapseRepeatedMul())
    model_wrapper = model_wrapper.transform(InferDataTypes(allow_scaledint_dtypes=False))
    model_wrapper = model_wrapper.transform(GiveUniqueNodeNames())
    model_wrapper = model_wrapper.transform(SortGraph())
    model_wrapper.save('before_loop_extraction_convert_qonnx_to_finn.onnx')
    #model_wrapper = model_wrapper.transform(GemmToMatMul())
    model_wrapper = model_wrapper.transform(ConvertDivToMul())
    model_wrapper = model_wrapper.transform(to_hw.InferElementwiseBinaryOperation())
    model_wrapper.save('before_loop_extraction_gemm_to_matmul.onnx')
    model_wrapper = model_wrapper.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model_wrapper = model_wrapper.transform(FoldConstants())

    model_wrapper.save('before_loop_extraction.onnx')
    loop_extraction = LoopExtraction(hierarchy_list=['', 'layers.0'])
    model_wrapper = model_wrapper.transform(loop_extraction)
    model_wrapper.save('after_loop_extraction.onnx')

    # should be one loop-body node per layer
    assert len(model_wrapper.model.graph.node) == num_layers + 2, "Loop extraction did not find expected number of loop bodies"

    model_wrapper = model_wrapper.transform(LoopRolling(loop_extraction.loop_body_template))
    assert len(model_wrapper.model.graph.node) == 1, "Should Roll into a Single FinnLoop Node"
    loop_node = model_wrapper.model.graph.node[0]

    assert loop_node.op_type == "FINNLoop", "Node should be op_type FinnLoop"

    assert util.get_by_name(loop_node.attribute, "iteration").i == num_layers
    assert util.get_by_name(loop_node.attribute, "backend").s.decode('utf-8') == "fpgadataflow"
    assert util.get_by_name(loop_node.attribute, "inputDataType").s.decode('utf-8') == m_input_dt
    assert util.get_by_name(loop_node.attribute, "outputDataType").s.decode('utf-8') == m_output_dt
    model_wrapper.save('rolled.onnx')
    # Check tensor shapes by name since loop rolling may reorder inputs
    check_tensor_shape(model_wrapper, 'x', [input_size, hidden_size]) # activation input shape should remain the same
    check_tensor_shape(model_wrapper, 'linear_5', [input_size, hidden_size]) # activation output shape should remain the same
    model_wrapper.get_tensor_shape(loop_node.input[1])[0] == num_layers # loop iteration count should match number of layers
    model_wrapper.get_tensor_shape(loop_node.input[2])[0] == num_layers # loop condition count should match number of layers

    loop_body_wrapper = model_wrapper.make_subgraph_modelwrapper(util.get_by_name(loop_node.attribute, "body").g )

    for node in loop_body_wrapper.model.graph.node:
        if node.op_type == "MatMul" or node.op_type == "ElementWiseAdd":
            mlo_attr  = util.get_by_name(node.attribute, "mlo_max_iter")
            assert mlo_attr is not None, f"{node.op_type} node in loop body should have mlo_max_iter attribute"
            assert mlo_attr.i == num_layers, "Loop body max iteration count should match number of layers"

    print("assertions checked, now executing original and rolled model to compare results")
    # execute original model and rolled model using onnx runtime
    # ensure results match
    #ort_sess = ort.InferenceSession(onnx_path)

    # Create an input tensor
    dummy_input = gen_finn_dt_tensor(DataType["FLOAT32"], [hidden_size, input_size])

    simple_model = qonnx.core.modelwrapper.ModelWrapper('before_loop_extraction.onnx')
    simple_model = simple_model.transform(SetExecMode("cppsim"))
    simple_model = simple_model.transform(InferShapes())
    ort_inputs = {'x': dummy_input}
    simple_model.save("simple_model_cppsim_preexec.onnx")
    ort_outputs = oxe.execute_onnx(simple_model, ort_inputs)
    #ort_outputs = ort_sess.run(None, ort_inputs)

    model_wrapper = model_wrapper.transform(SetExecMode("cppsim"))
    finn_outputs = oxe.execute_onnx(model_wrapper, ort_inputs)

    assert (ort_outputs['linear_5'] == finn_outputs['linear_5']).all()

