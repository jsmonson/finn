import torch
from brevitas.export import export_qonnx
from brevitas.nn import QuantLinear
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat

class SimpleSubModule(torch.nn.Module):
    def __init__(self, in_features, out_features, mul_val=200):
        super(SimpleSubModule, self).__init__()
        self.mul_val = torch.tensor([mul_val])
        self.linear = QuantLinear(in_features, out_features, bias=True,
                                  weight_quant=Int8WeightPerTensorFloat,
                                  input_quant=Int8ActPerTensorFloat)

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
            in_features = input_size if i == 0 else hidden_size
            out_features = hidden_size if i != (num_layers-1) or output_size is None else hidden_size
            self.layers.append(SimpleSubModule(in_features, out_features, mul_val=self.mul_val))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

batch_size = 2
input_size = 10
model = SimpleModule(input_size=input_size)
x = torch.rand((batch_size,input_size))
model(x) # Initialise scale factors
model.eval()
export_qonnx(model, x, "simple_model.onnx", dynamo=True)
