import torch
import torch.nn as nn

from collections import OrderedDict

import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)

        output = output.replace_feature(output.features + self.i_branch(identity).features)

        return output


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),                                   
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),                                             
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    @staticmethod
    def safe_to_downsample(conv_block, sparse_tensor):
        for layer in conv_block:
            if isinstance(layer, (spconv.SparseConv3d, spconv.SubMConv3d)):
                ksize = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                dilation = layer.dilation
                shape = sparse_tensor.spatial_shape

                min_required = [(k - 1) * dil + 1 for k, dil in zip(ksize, dilation)]
                if not all(s >= m for s, m in zip(shape, min_required)):
                    return False

                coords = sparse_tensor.indices[:, 1:]  # [N, 3] z, y, x
                survived = []
                for dim in range(3):
                    out_coord = (coords[:, dim] + padding[dim]
                        - (ksize[dim] - 1) * dilation[dim] // 2) // stride[dim]
                    survived.append((out_coord >= 0) & (out_coord < 
                        (shape[dim] + 2*padding[dim] - dilation[dim]*(ksize[dim]-1) - 1)//stride[dim] + 1))

                mask = survived[0] & survived[1] & survived[2]
                return mask.any().item() 

        return True 

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        
        if (
            len(self.nPlanes) > 1 
            and all(d > 2 for d in output.spatial_shape) # Stop network deepening before the dimension disappears
            and UBlock.safe_to_downsample(self.conv, output)
        ): 
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output_features = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(output_features)

            output = self.blocks_tail(output)

        return output

class MLP(SparseModule):
    def __init__(
        self,
        n_planes,
        norm_fn,
        activation_fn=None,
        bias=False,
    ):
        super().__init__()

        self.sequence = spconv.SparseSequential()

        for i in range(len(n_planes) - 2):
            self.sequence.add(
                nn.Linear(
                    n_planes[i],
                    n_planes[i + 1],
                    bias=bias,
                )
            )
            self.sequence.add(norm_fn(n_planes[i + 1]))
            self.sequence.add(activation_fn())

        self.sequence.add(
            nn.Linear(
                n_planes[-2],
                n_planes[-1],
                bias=bias,
            )
        )

    def forward(self, input):
        return self.sequence(input)
