import onnx
import onnxscript
import torch 
import pytest
import qonnx.core.modelwrapper

from finn.transformation.fpgadataflow.loop_rolling import LoopRolling, LoopExtraction


class SimpleSubModule(torch.nn.Module):
    def __init__(self, in_features, out_features, mul_val=200):
        super(SimpleSubModule, self).__init__()
        self.mul_val = mul_val
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        return self.mul_val * self.linear(x)



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
    model = SimpleModule(input_size=input_size, hidden_size=hidden_size, 
                        num_layers=num_layers, output_size=output_size, mul_val=150)
    model.eval()
    # Create a dummy input tensor
    dummy_input = torch.randn(hidden_size, input_size)
    
    # Export the model to ONNX format
    onnx_path = f"simple_module_{num_layers}layers.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=18, dynamo=True)

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Add finn_datatype Quantization Annotation for all tensors
    onnx_ir = onnxscript.ir.serde.deserialize_model(onnx_model)
    for node in onnx_ir.graph._nodes:
        for tensor in node.inputs + node.outputs:
            if "finn_datatype" not in tensor.meta.get("quant_parameter_tensor_names", {}):
                tensor.meta["quant_parameter_tensor_names"] = {"finn_datatype": "FLOAT32"}
    onnx_model = onnxscript.ir.serde.serialize_model(onnx_ir)
    onnx.save(onnx_model, onnx_path)

    # Load the ONNX model to verify    
    print(f"Model exported successfully to {onnx_path}")
    print(f"Model has {num_layers} layers with input size {input_size}")
    return onnx_path



def test_export_model():
    input_size = 20
    hidden_size = 20
    num_layers = 6
    output_size = None
    
    onnx_path = export_model_to_onnx(input_size, hidden_size, num_layers, output_size)
    
    model_wrapper = qonnx.core.modelwrapper.ModelWrapper(onnx_path)
    
    loop_extraction = LoopExtraction(hierarchy_list=['', 'layers.0'])   
    model_wrapper = model_wrapper.transform(loop_extraction)
    model_wrapper.save("output_with_loop_extraction.onnx")
    model_wrapper = model_wrapper.transform(LoopRolling(loop_extraction.loop_body_template))

    model_wrapper.save("output_with_loop_rolling.onnx")    
    
    # Check the number of nodes in the graph
    #assert len(onnx_model.graph.node) == num_layers, "Number of layers in ONNX model does not match expected"
    
    # Check input and output sizes
    #assert onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value == input_size, "Input size mismatch"
    #assert onnx_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value == (hidden_size + (num_layers - 1) * 10), "Output size mismatch"