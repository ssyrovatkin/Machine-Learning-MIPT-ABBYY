import random
import os

import pytest
import torch
import torch.nn as nn

import SciDevNN.SDmodules as SDModules
import SciDevNN.SDoptim as SDOptim


def make_linears(in_dim, out_dim, bias):
    init = {'W': torch.randn((in_dim, out_dim), dtype=torch.float32)}
    if bias:
        init['b'] = torch.randn((out_dim,), dtype=torch.float32)

    torch_linear = nn.Linear(in_dim, out_dim, bias=bias)
    torch_linear.weight.data = init['W'].T

    if bias:
        torch_linear.bias.data = init['b']

    scidev_linear = SDModules.Linear(in_dim, out_dim, bias=bias, init=init)

    return torch_linear, scidev_linear


class TestLinear:
    @pytest.mark.parametrize("in_dim,out_dim,bias", [
        (1, 1, False),
        (1, 1, True),
        (3, 3, False),
        (3, 3, True),
        (4, 7, False),
        (4, 7, True),
        (7, 4, False),
        (7, 4, True),
    ])
    def test_linear_forward_simple(self, in_dim, out_dim, bias):
        torch_linear, scidev_linear = make_linears(in_dim, out_dim, bias)

        input_zeros = torch.zeros(in_dim, dtype=torch.float32)
        output_zeros_torch = torch_linear(input_zeros)
        output_zeros_scidev = scidev_linear(input_zeros)
        assert output_zeros_torch.shape == output_zeros_scidev.shape
        assert torch.all(
            torch.abs(output_zeros_torch - output_zeros_scidev) < 1e-5)

        input_ones = torch.ones(in_dim, dtype=torch.float32)
        output_ones_torch = torch_linear(input_ones)
        output_ones_scidev = scidev_linear(input_ones)
        assert torch.all(
            torch.abs(output_ones_torch - output_ones_scidev) < 1e-5)

    @pytest.mark.parametrize("input_shape,out_dim", [
        ((1, 4), 3),
        ((3, 4), 7),
        ((3, 7), 5),
        ((5, 8), 1),
    ])
    def test_linear_forward(self, input_shape, out_dim):
        torch_linear, scidev_linear = make_linears(input_shape[-1], out_dim,
                                                   True)
        input_t = torch.randn(input_shape, dtype=torch.float32)

        output_torch = torch_linear(input_t)
        output_scidev = scidev_linear(input_t)

        assert output_torch.shape == output_scidev.shape
        assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

    @pytest.mark.parametrize("input_shape,out_dim", [
        ((1, 4), 3),
        ((3, 4), 7),
        ((3, 7), 5),
        ((5, 8), 1),
    ])
    def test_linear_backward(self, input_shape, out_dim):
        torch_linear, scidev_linear = make_linears(input_shape[-1], out_dim,
                                                   True)
        input_t = torch.randn(input_shape, dtype=torch.float32,
                              requires_grad=True)

        output_torch = torch_linear(input_t)
        output_scidev = scidev_linear(input_t)

        torch_linear.zero_grad()
        output_torch.backward(torch.ones_like(output_torch))
        scidev_linear.backward(input_t, torch.ones_like(output_scidev))

        grads = scidev_linear.get_grad_test()

        assert input_t.grad.shape == grads['gradOut'].shape
        assert torch.all(
            torch.abs(grads['gradOut'] - input_t.grad) < 1e-5)

        assert torch_linear.weight.grad.shape == grads['gradW'].shape
        assert torch.all(
            torch.abs(grads['gradW'] - torch_linear.weight.grad) < 1e-5)

        assert torch_linear.bias.grad.shape == grads['gradB'].shape
        assert torch.all(
            torch.abs(grads['gradB'] - torch_linear.bias.grad) < 1e-5)


class TestActivations:
    class TestForward:
        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_softmax(self, input_shape):
            input_t = torch.randn(input_shape, dtype=torch.float32,
                                  requires_grad=False).detach()
            torch_softmax = nn.Softmax()
            scidev_softmax = SDModules.Softmax()

            output_torch = torch_softmax(input_t)
            output_scidev = scidev_softmax(input_t)

            assert output_torch.shape == output_scidev.shape
            assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_sigmoid(self, input_shape):
            input_t = torch.randn(input_shape, dtype=torch.float32,
                                  requires_grad=False).detach()
            torch_sigmoid = nn.Sigmoid()
            scidev_sigmoid = SDModules.Sigmoid()

            output_torch = torch_sigmoid(input_t)
            output_scidev = scidev_sigmoid(input_t)

            assert output_torch.shape == output_scidev.shape
            assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

    class TestBackward:
        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_softmax(self, input_shape):
            input_t = torch.randn(input_shape, dtype=torch.float32,
                                  requires_grad=True)
            torch_softmax = nn.Softmax()
            scidev_softmax = SDModules.Softmax()

            output_torch = torch_softmax(input_t)
            output_scidev = scidev_softmax(input_t)

            output_torch.backward(torch.ones_like(output_torch))
            scidev_softmax.backward(input_t, torch.ones_like(output_scidev))

            grad = scidev_softmax.get_grad_test()

            assert input_t.grad.shape == grad.shape
            assert torch.all(torch.abs(grad - input_t.grad) < 1e-5)

        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_sigmoid(self, input_shape):
            input_t = torch.randn(input_shape, dtype=torch.float32,
                                  requires_grad=True)
            torch_sigmoid = nn.Sigmoid()
            scidev_sigmoid = SDModules.Sigmoid()

            output_torch = torch_sigmoid(input_t)
            output_scidev = scidev_sigmoid(input_t)

            output_torch.backward(torch.ones_like(output_torch))
            scidev_sigmoid.backward(input_t, torch.ones_like(output_scidev))

            grad = scidev_sigmoid.get_grad_test()

            assert input_t.grad.shape == grad.shape
            assert torch.all(torch.abs(grad - input_t.grad) < 1e-5)


class TestLoss:
    class TestForward:
        @pytest.mark.parametrize("input_shape", [
            3,
            (1, 4),
            (7, 4, 9),
            (5, 8, 10, 3),
        ])
        def test_mse(self, input_shape):
            x = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=False).detach()
            y = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=False).detach()
            torch_mse = nn.MSELoss(reduction='mean')
            scidev_mse = SDModules.MSE()

            output_torch = torch_mse(x, y)
            output_scidev = scidev_mse(x, y)

            assert output_torch.shape == output_scidev.shape
            assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_crossentropy(self, input_shape):
            x = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=False).detach()
            y = torch.randint(low=0,
                              high=input_shape[-1],
                              size=(input_shape[0],),
                              requires_grad=False).detach()
            torch_ce = nn.CrossEntropyLoss(reduction='mean')
            scidev_ce = SDModules.CrossEntropy()

            output_torch = torch_ce(x, y)
            output_scidev = scidev_ce(x, y)

            assert output_torch.shape == output_scidev.shape
            assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

    class TestBackward:
        @pytest.mark.parametrize("input_shape", [
            (1, 2),
            (3, 4),
            (7, 4, 9),
            (5, 8, 10, 3),
        ])
        def test_mse(self, input_shape):
            x = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=True)

            y = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=False).detach()

            torch_mse = nn.MSELoss(reduction='mean')
            scidev_mse = SDModules.MSE()

            output_torch = torch_mse(x, y)
            _ = scidev_mse(x, y)

            output_torch.backward(torch.ones_like(output_torch))
            scidev_mse.backward(x, y)

            grad = scidev_mse.get_grad_test()

            assert x.grad.shape == grad.shape
            assert torch.all(torch.abs(grad - x.grad) < 1e-5)

        @pytest.mark.parametrize("input_shape", [
            (1, 4),
            (3, 4),
            (3, 7),
            (5, 8),
        ])
        def test_crossentropy(self, input_shape):
            x = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=True)
            y = torch.randint(low=0,
                              high=input_shape[-1],
                              size=(input_shape[0],),
                              requires_grad=False).detach()
            torch_ce = nn.CrossEntropyLoss(reduction='mean')
            scidev_ce = SDModules.CrossEntropy()

            output_torch = torch_ce(x, y)
            _ = scidev_ce(x, y)

            output_torch.backward(torch.ones_like(output_torch))
            scidev_ce.backward(x, y)

            grad = scidev_ce.get_grad_test()

            assert x.grad.shape == grad.shape
            assert torch.all(torch.abs(grad - x.grad) < 1e-5)


class TestGradientDescend:
    @pytest.mark.parametrize("input_shape,lr", [
        (3, 1),
        ((1, 4), 3),
        ((7, 4, 9), 1e-5),
        ((5, 8, 10, 3), 0.1),
    ])
    def test_sgd(self, input_shape, lr):
        test_tensor_torch = torch.randn(input_shape, dtype=torch.float32,
                            requires_grad=True)

        test_tensor_scidev = torch.clone(test_tensor_torch).detach()
        test_tensor_torch.grad = torch.ones_like(test_tensor_torch)

        torch_opt = torch.optim.SGD([test_tensor_torch], lr=lr)
        scidev_opt = SDOptim.GradientDescend(lr=lr)

        torch_opt.step()
        test_tensor_scidev = scidev_opt.step(test_tensor_scidev, torch.ones_like(test_tensor_scidev))

        assert test_tensor_scidev.shape == test_tensor_torch.shape
        assert torch.all(
            torch.abs(test_tensor_scidev - test_tensor_torch) < 1e-5)

    @pytest.mark.parametrize("input_shape,lr,betas", [
        (3, 1, (0.9, 0.999)),
        ((1, 4), 3, (0.99, 0.99)),
        ((7, 4, 9), 1e-5, (0.999, 0.9)),
        ((5, 8, 10, 3), 0.1, (0.1, 0.1)),
    ])
    def test_adam(self, input_shape, lr, betas):
        test_tensor_torch = torch.randn(input_shape, dtype=torch.float32,
                                        requires_grad=True)

        test_tensor_scidev = torch.clone(test_tensor_torch).detach()
        test_tensor_torch.grad = torch.ones_like(test_tensor_torch)
        torch_opt = torch.optim.Adam([test_tensor_torch], lr=lr, betas=betas)
        scidev_opt = SDOptim.Adam(lr=lr, betas=betas)
        for step in range(0, 5):
            torch_opt.step()
            test_tensor_scidev = scidev_opt.step(test_tensor_scidev,
                                                 torch.ones_like(
                                                     test_tensor_scidev))

            assert test_tensor_scidev.shape == test_tensor_torch.shape
            assert torch.all(
                torch.abs(test_tensor_scidev - test_tensor_torch) < 1e-5)
