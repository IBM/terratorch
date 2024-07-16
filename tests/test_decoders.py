import unittest
import pytest 

import torch
from torch import nn

from terratorch.models.decoders.upernet_decoder import ConvModule
from terratorch.models.decoders import FCNDecoder

class TestConvModule(unittest.TestCase):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 64
        self.kernel_size = 3
        self.padding = 1
        self.inplace = True
        self.batch_size = 8
        self.input_shape = (self.batch_size, self.in_channels, 256, 256)

        self.module = ConvModule(
            self.in_channels, self.out_channels, self.kernel_size, self.padding, self.inplace
        )

        self.input = torch.rand(self.input_shape)

    def test_forward(self):
        output = self.module(self.input)
        self.assertEqual(output.shape, self.input_shape[:1] + (self.out_channels,) + output.shape[2:])

    def test_conv_weight_shape(self):
        self.assertEqual(self.module.conv.weight.shape, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

    def test_norm_weight_shape(self):
        self.assertEqual(self.module.norm.weight.shape, (self.out_channels,))

class TestFCNDecoder(unittest.TestCase):

    def test_fcn_decoder(self):
        # create inputs
        batch_size = 32
        height = 32
        width = 32
        num_channels = 3
        embed_dim = [64, 128, 256]
        num_convs = 4
        in_index = -1

        # create model
        decoder = FCNDecoder(
            embed_dim=embed_dim,
            channels=num_channels,
            num_convs=num_convs,
            in_index=in_index
        )

        # create input tensor
        x = torch.rand((batch_size, embed_dim[in_index], height, width))
        # get output shape
        out = decoder([None, x])
        out_shape = out.shape

        # check output shape
        self.assertEqual(out_shape, (batch_size, num_channels, (2**num_convs)*height, (2**num_convs)*width))