import torch.nn as nn
from typing import List, Tuple
import functools
import spconv.pytorch as spconv
from .sparse_module import *
from typing import Dict
import torch.nn.functional as F

class SmartTreeXX(nn.Module):
    def __init__(
            self,
            *,
            u_channels: List[int], 
            mlp_hidden_layers: List[int],
            conv_block_reps: int = 1
    ) -> None:
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                3, u_channels[0], kernel_size=3, padding=1, bias=False, indice_key="subm1"
            )
        )

        self.unet = UBlock(
            nPlanes=u_channels,
            norm_fn=norm_fn,
            block_reps=conv_block_reps,
            block=ResidualBlock,
            indice_key_id=1
        )

        self.output_layer = spconv.SparseSequential(
            norm_fn(u_channels[0]),
            nn.ReLU()
        )

        self.offset_linear = MLP(
            n_planes=[u_channels[0]] + mlp_hidden_layers + [3],
            norm_fn=norm_fn,
            activation_fn=nn.ReLU,    
            bias=True
        )

        self.regression_linear = MLP(
            n_planes=[u_channels[0]] + mlp_hidden_layers + [1], 
            norm_fn=norm_fn,
            activation_fn=nn.ReLU,    
            bias=True
        )
        
        self.apply(self.set_bn_init)
        return
    
    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
    
    def forward(self, input: spconv.SparseConvTensor) -> Dict[str, torch.Tensor]:
        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)

        offset = self.offset_linear(output).features

        directions: torch.Tensor
        distances: torch.Tensor

        regression = self.regression_linear(output).features
        directions = F.normalize(offset)
        distances = F.softplus(regression, beta=1.0, threshold=20.0)
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html
        return {"direction": directions, "distance": distances}