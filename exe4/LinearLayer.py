"""
    Implement customized Linear Layer, with the help of torch
    There are some comments about how gradient is computed w.r.t each item (like input or weight)
    Author: Qianyue He
    Date: 2022.10.7
"""

import math
import torch
from torch import nn
from torch.autograd import Function

from torch.nn import init
from torch.nn import functional as F

class LinearLayerCore(Function):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input @ weight.T
        if bias is not None:
            output += bias.unsqueeze(dim = 0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # calculate gradient with respect to input (in problem set 4, we found that this term is used in the layer before)
        # You can refer to my homework (Problem set 4, equation (11) to find out)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight

        # w.r.t to weight, it is interesting to node that, instead of concatenating
        # grad are added together (different grad from different input (rows), are added)
        # Easy to know with some deduction 
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.T @ input
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim = 0)

        return grad_input, grad_weight, grad_bias

"""
    This module takes the form of nn.Linear
"""
class LinearLayer(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    """
        This function is copied from torch.nn.Linear (as this is just a initialization method)
    """
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return LinearLayerCore.apply(input, self.weight, self.bias)

if __name__ == "__main__":
    layer1 = LinearLayer(4, 8)
    # layer2 = LinearLayer(8, 8)
    data = torch.normal(0, 1, (16, 4))
    tmp = F.relu(layer1(data))
    # tmp = F.relu(layer2(tmp))

    tmp.backward(torch.ones_like(tmp))

    for param in layer1.parameters():
        print(param.grad.shape)