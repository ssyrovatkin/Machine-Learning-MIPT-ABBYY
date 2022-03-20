import random
import os

import pytest
import torch
import torch.nn as nn

import SciDevNN.SDmodules as SDModules


class TestConv:
    @pytest.mark.parametrize("output_channels, kernel_size, stride, padding, input_channels, batch_size",
                             [(3, 1, 1, 0, 3, 1),
                              (3, 3, 1, 0, 3, 1),
                              (5, 3, 1, 0, 3, 3),
                              (5, 3, 2, 0, 3, 3),
                              (5, 3, 2, 5, 3, 3),
                              (3, 3, 2, 5, 5, 3),
                              (3, 7, 1, 5, 5, 3)
                             ])
    def test_conv_forward(self, output_channels, kernel_size, stride, padding, input_channels, batch_size):
        test_input = torch.rand((batch_size, input_channels, 10, 10), dtype=torch.float32)
        torch_conv = nn.Conv2d(input_channels, output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        weight = torch.rand((output_channels, input_channels, kernel_size, kernel_size), dtype=torch.float32)
        bias = torch.rand(output_channels, dtype=torch.float32)
        torch_conv.weight.data = weight
        torch_conv.bias.data = bias

        scidev_conv = SDModules.Conv2D(input_channels, output_channels, kernel_size, stride,
                                       padding, init={'W': weight, 'b': bias})

        output_torch = torch_conv(test_input)
        output_scidev = scidev_conv(test_input)

        assert output_torch.shape == output_scidev.shape
        assert torch.all(torch.abs(output_torch - output_scidev) < 1e-4)

    @pytest.mark.parametrize("output_channels, kernel_size, stride, padding, input_channels, batch_size",
                             [(3, 1, 1, 0, 3, 1),
                              (3, 3, 1, 0, 3, 1),
                              (5, 3, 1, 0, 3, 3),
                              (5, 3, 2, 0, 3, 3),
                              (5, 3, 2, 5, 3, 3),
                              (3, 3, 2, 5, 5, 3),
                              (3, 7, 1, 5, 5, 3)
                             ])
    def test_conv_backward(self, output_channels, kernel_size, stride, padding, input_channels, batch_size):
        test_input = torch.rand((batch_size, input_channels, 10, 10), dtype=torch.float32,
                              requires_grad=True)
        torch_conv = nn.Conv2d(input_channels, output_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)

        weight = torch.rand((output_channels, input_channels, kernel_size, kernel_size), dtype=torch.float32)
        bias = torch.rand(output_channels, dtype=torch.float32)
        torch_conv.weight.data = weight
        torch_conv.bias.data = bias

        scidev_conv = SDModules.Conv2D(input_channels, output_channels, kernel_size, stride,
                                       padding, init={'W': weight, 'b': bias})

        output_torch = torch_conv(test_input)
        output_scidev = scidev_conv(test_input)

        torch_conv.zero_grad()
        output_torch.backward(torch.ones_like(output_torch))
        scidev_conv.backward(torch.ones_like(output_scidev))

        grads = scidev_conv.get_grad_test()

        assert torch_conv.weight.grad.shape == grads['gradW'].shape
        assert torch.all(
            torch.abs(grads['gradW'] - torch_conv.weight.grad) < 1e-4)

        assert torch_conv.bias.grad.shape == grads['gradB'].shape
        assert torch.all(
            torch.abs(grads['gradB'] - torch_conv.bias.grad) < 1e-5)

        assert test_input.grad.shape == grads['gradOut'].shape
        assert torch.all(
            torch.abs(grads['gradOut'] - test_input.grad) < 1e-4)

class TestBatchNorm:
    @pytest.mark.parametrize("batch_size,input_channels,eps",
                             [(1, 1, 1e-5),
                              (1, 2, 1e-5),
                              (3, 1, 1e-5),
                              (3, 2, 1e-5),
                              (3, 5, 1e-5),
                              (3, 10, 1e-5),
                              (3, 10, 1e-3),
                              (5, 10, 1e-5),
                              ])
    def test_batchnorm_forward(self, batch_size, input_channels, eps):
        test_input = torch.rand((batch_size, input_channels, 10, 10), dtype=torch.float32)
        weight = torch.rand(input_channels, dtype=torch.float32)
        bias = torch.rand(input_channels, dtype=torch.float32)

        torch_batchnorm = nn.BatchNorm2d(input_channels, eps, momentum=0)
        torch_batchnorm.weight.data = weight
        torch_batchnorm.bias.data = bias
        scidev_batchnorm = SDModules.BatchNorm2D(input_channels, eps, init={'W': weight, 'b': bias})

        torch_output = torch_batchnorm(test_input)
        scidev_output = scidev_batchnorm(test_input)

        assert torch_output.shape == scidev_output.shape
        assert torch.all(torch.abs(torch_output - scidev_output) < 1e-4)

    @pytest.mark.parametrize("batch_size,input_channels,eps",
                             [(1, 1, 1e-5),
                              (1, 2, 1e-5),
                              (3, 1, 1e-5),
                              (3, 2, 1e-5),
                              (3, 5, 1e-5),
                              (3, 10, 1e-5),
                              (3, 10, 1e-3),
                              (5, 10, 1e-5),
                              ])
    def test_batchnorm_backward(self,  batch_size, input_channels, eps):
        test_input = torch.rand((batch_size, input_channels, 10, 10), dtype=torch.float32, requires_grad=True)

        weight = torch.rand(input_channels, dtype=torch.float32)
        bias = torch.rand(input_channels, dtype=torch.float32)
        torch_batchnorm = nn.BatchNorm2d(input_channels, eps, momentum=0)
        torch_batchnorm.weight.data = weight.T
        torch_batchnorm.bias.data = bias
        scidev_batchnorm = SDModules.BatchNorm2D(input_channels, eps, init={'W': weight, 'b': bias})

        torch_output = torch_batchnorm(test_input)
        scidev_output = scidev_batchnorm(test_input)

        torch_batchnorm.zero_grad()
        torch_output.backward(torch.ones_like(torch_output))
        scidev_batchnorm.backward(torch.ones_like(scidev_output))

        grads = scidev_batchnorm.get_grad_test()

        assert torch_batchnorm.weight.grad.shape == grads['gradW'].shape
        assert torch.all(
            torch.abs(grads['gradW'] - torch_batchnorm.weight.grad.data) < 1e-3)

        assert torch_batchnorm.bias.grad.shape == grads['gradB'].shape
        assert torch.all(
            torch.abs(grads['gradB'] - torch_batchnorm.bias.grad.data) < 1e-5)

        assert test_input.grad.shape == grads['gradOut'].shape
        assert torch.all(
            torch.abs(grads['gradOut'] - test_input.grad.data) < 1e-5)

class TestMaxPooling:
    @pytest.mark.parametrize("batch_size, input_channels, kernel_size, stride",
                             [(1, 1, 1, 1),
                              (1, 2, 2, 1),
                              (3, 1, 3, 2),
                              (3, 2, 3, 3),
                              (3, 5, 3, 1),
                              (3, 10, 5, 2),
                              (3, 10, 3, 2),
                              (5, 10, 7, 1),
                              ])
    def test_maxpool_forward(self, batch_size, input_channels, kernel_size, stride):
        input_size = (batch_size, input_channels, 10, 10)
        test_input = torch.rand(input_size, dtype=torch.float32)

        torch_maxpool = nn.MaxPool2d(stride=stride, kernel_size=(kernel_size, kernel_size))
        scidev_maxpool = SDModules.MaxPooling2D(input_size, stride=stride, kernel_size=kernel_size)

        torch_output = torch_maxpool(test_input)
        scidev_output = scidev_maxpool(test_input)

        assert torch_output.shape == scidev_output.shape
        assert torch.all(torch.abs(torch_output - scidev_output) < 1e-5)

    @pytest.mark.parametrize("batch_size, input_channels, kernel_size, stride",
                             [(1, 1, 1, 1),
                              (1, 2, 2, 1),
                              (3, 1, 3, 2),
                              (3, 2, 3, 3),
                              (3, 5, 3, 1),
                              (3, 10, 5, 2),
                              (3, 10, 3, 2),
                              (5, 10, 7, 1),
                              ])
    def test_maxpool_backward(self, batch_size, input_channels, kernel_size, stride):
        input_size = (batch_size, input_channels, 10, 10)
        test_input = torch.rand(input_size, dtype=torch.float32, requires_grad=True)

        torch_maxpool = nn.MaxPool2d(stride=stride, kernel_size=(kernel_size, kernel_size))
        scidev_maxpool = SDModules.MaxPooling2D(input_size, stride=stride, kernel_size=kernel_size)

        torch_output = torch_maxpool(test_input)
        scidev_output = scidev_maxpool(test_input)

        torch_maxpool.zero_grad()
        torch_output.backward(torch.ones_like(torch_output))
        scidev_maxpool.backward(test_input, torch.ones_like(scidev_output))

        grads = scidev_maxpool.get_grad_test()
        print(grads['gradOut'], test_input.grad.data)

        assert test_input.grad.shape == grads['gradOut'].shape
        assert torch.all( torch.abs(grads['gradOut'] - test_input.grad.data) < 1e-5)