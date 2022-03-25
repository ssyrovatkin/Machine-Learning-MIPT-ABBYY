import pytest
import torch
import torch.nn as nn

import SciDevNN.SDmodules as SDModules

class TestTanh:
    class TestForward:
        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_forward(self, input_shape):
            input_t = torch.randn(input_shape, dtype=torch.float32,
                                  requires_grad=False).detach()
            torch_tanh = nn.Tanh()
            scidev_tanh = SDModules.Tanh()

            output_torch = torch_tanh(input_t)
            output_scidev = scidev_tanh(input_t)

            assert output_torch.shape == output_scidev.shape
            assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

    class TestBackward:
        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_backward(self, input_shape):
            input_t = torch.randn(input_shape, dtype=torch.float32,
                                  requires_grad=True)
            torch_tanh = nn.Tanh()
            scidev_tanh = SDModules.Tanh()

            output_torch = torch_tanh(input_t)
            output_scidev = scidev_tanh(input_t)

            output_torch.backward(torch.ones_like(output_torch))
            scidev_tanh.backward(input_t, torch.ones_like(output_scidev))

            grad = scidev_tanh.get_grad_test()

            assert input_t.grad.shape == grad.shape
            assert torch.all(torch.abs(grad - input_t.grad) < 1e-5)