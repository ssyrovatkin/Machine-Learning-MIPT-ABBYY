import pytest
import torch
import torch.nn as nn

import SciDevNN.SDmodules as SDModules

class TestTanh:
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

def make_RNNcell(input_size, hidden_size, bias):
    init = {'W_ih': torch.randn((hidden_size, input_size), dtype=torch.float32)}
    init ['W_hh'] = torch.randn((hidden_size, hidden_size), dtype=torch.float32)
    if bias:
        init['bi'] = torch.randn((hidden_size,), dtype=torch.float32)
        init['bh'] = torch.randn((hidden_size,), dtype=torch.float32)

    torch_rnn = nn.RNNCell(input_size, hidden_size, bias=bias)

    torch_rnn.weight_ih.data = init['W_ih']
    torch_rnn.weight_hh.data = init['W_hh']

    if bias:
        torch_rnn.bias_ih.data = init['bi']
        torch_rnn.bias_hh.data = init['bh']

    scidev_rnn = SDModules.RNNCell(input_size, hidden_size, bias=bias, init=init)

    return torch_rnn, scidev_rnn

class TestRNNCell:
    @pytest.mark.parametrize("batch, input_size,hidden_size, bias", [
        (1, 1, 2, False),
        (1, 1, 1, True),
        (3, 3, 3, False),
        (3, 3, 3, True),
        (4, 4, 7, False),
        (4, 4, 7, True),
        (5, 7, 4, False),
        (5, 7, 4, True),
    ])
    def test_forward(self, batch, input_size, hidden_size, bias):
        input_t = torch.randn((batch, input_size), dtype=torch.float32,
                              requires_grad=False).detach()
        input_h = torch.randn((batch, hidden_size), dtype=torch.float32,
                              requires_grad=False).detach()
        torch_rnn, scidev_rnn = make_RNNcell(input_size, hidden_size, bias)

        output_torch = torch_rnn(input_t, input_h)
        output_scidev = scidev_rnn(input_t, input_h)

        print(output_torch)
        print(output_scidev)

        assert output_torch.shape == output_scidev.shape
        assert torch.all(torch.abs(output_torch - output_scidev) < 1e-5)

    @pytest.mark.parametrize("batch, input_size,hidden_size, bias", [
        (1, 1, 2, False),
        (1, 1, 1, True),
        (3, 3, 3, False),
        (3, 3, 3, True),
        (4, 4, 7, False),
        (4, 4, 7, True),
        (5, 7, 4, False),
        (5, 7, 4, True),
    ])
    def test_backward(self, batch, input_size, hidden_size, bias):
        input_t = torch.randn((batch, input_size), dtype=torch.float32,
                              requires_grad=True)
        input_h = torch.randn((batch, hidden_size), dtype=torch.float32,
                              requires_grad=True)
        torch_rnn, scidev_rnn = make_RNNcell(input_size, hidden_size, bias)

        output_torch = torch_rnn(input_t, input_h)
        output_scidev = scidev_rnn(input_t, input_h)

        torch_rnn.zero_grad()
        output_torch.backward(torch.ones_like(output_torch))
        scidev_rnn.backward(input_t, torch.ones_like(output_scidev))

        grads = scidev_rnn.get_grad_test()

        assert input_t.grad.shape == grads['gradx'].shape
        assert torch.all(
            torch.abs(grads['gradx'] - input_t.grad) < 1e-5)

        assert input_h.grad.shape == grads['gradh'].shape
        assert torch.all(
            torch.abs(grads['gradh'] - input_h.grad) < 1e-5)

        assert torch_rnn.weight_hh.grad.shape == grads['gradWh'].shape
        assert torch.all(
            torch.abs(grads['gradWh'] - torch_rnn.weight_hh.grad) < 1e-5)

        assert torch_rnn.weight_ih.grad.shape == grads['gradWi'].shape
        assert torch.all(
            torch.abs(grads['gradWi'] - torch_rnn.weight_ih.grad) < 1e-5)

        if bias:
            assert torch_rnn.bias_hh.grad.shape == grads['gradbh'].shape
            assert torch.all(
                torch.abs(grads['gradbh'] - torch_rnn.bias_hh.grad) < 1e-5)

            assert torch_rnn.bias_ih.grad.shape == grads['gradbi'].shape
            assert torch.all(
                torch.abs(grads['gradbi'] - torch_rnn.bias_ih.grad) < 1e-5)

def make_LSTMcell(input_size, hidden_size, bias):
    init = {'W_ih': torch.randn((4*hidden_size, input_size), dtype=torch.float32)}
    init ['W_hh'] = torch.randn((4*hidden_size, hidden_size), dtype=torch.float32)
    if bias:
        init['bi'] = torch.randn((4*hidden_size,), dtype=torch.float32)
        init['bh'] = torch.randn((4*hidden_size,), dtype=torch.float32)

    torch_lstm = nn.LSTMCell(input_size, hidden_size, bias=bias)

    torch_lstm.weight_ih.data = init['W_ih']
    torch_lstm.weight_hh.data = init['W_hh']

    if bias:
        torch_lstm.bias_ih.data = init['bi']
        torch_lstm.bias_hh.data = init['bh']

    scidev_lstm = SDModules.LSTMCell(input_size, hidden_size, bias=bias, init=init)

    return torch_lstm, scidev_lstm

class TestLSTMCell:
    @pytest.mark.parametrize("batch, input_size,hidden_size, bias", [
        (1, 1, 2, False),
        (1, 1, 1, True),
        (3, 3, 3, False),
        (3, 3, 3, True),
        (4, 4, 7, False),
        (4, 4, 7, True),
        (5, 7, 4, False),
        (5, 7, 4, True),
    ])
    def test_forward(self, batch, input_size, hidden_size, bias):
        input_t = torch.randn((batch, input_size), dtype=torch.float32,
                              requires_grad=False).detach()
        input_h = torch.randn((batch, hidden_size), dtype=torch.float32,
                              requires_grad=False).detach()
        input_c = torch.randn((batch, hidden_size), dtype=torch.float32,
                              requires_grad=False).detach()
        torch_lstm, scidev_lstm = make_LSTMcell(input_size, hidden_size, bias)

        (output_torch_h, output_torch_c) = torch_lstm(input_t, (input_h, input_c))
        (output_scidev_h, output_scidev_c) = scidev_lstm(input_t, (input_h, input_c))

        assert output_torch_h.shape == output_scidev_h.shape
        assert torch.all(torch.abs(output_torch_h - output_scidev_h) < 1e-5)

        assert output_torch_c.shape == output_scidev_c.shape
        assert torch.all(torch.abs(output_torch_c - output_scidev_c) < 1e-5)

    @pytest.mark.parametrize("batch, input_size,hidden_size, bias", [
        (1, 1, 2, False),
        (1, 1, 1, True),
        (3, 3, 3, False),
        (3, 3, 3, True),
        (4, 4, 7, False),
        (4, 4, 7, True),
        (5, 7, 4, False),
        (5, 7, 4, True),
    ])
    def test_backward(self, batch, input_size, hidden_size, bias):
        input_t = torch.randn((batch, input_size), dtype=torch.float32,
                              requires_grad=True)
        input_h = torch.randn((batch, hidden_size), dtype=torch.float32,
                              requires_grad=True)
        input_c = torch.randn((batch, hidden_size), dtype=torch.float32,
                              requires_grad=True)
        torch_lstm, scidev_lstm = make_LSTMcell(input_size, hidden_size, bias)

        (output_torch_h, output_torch_c) = torch_lstm(input_t, (input_h, input_c))
        (output_scidev_h, output_scidev_c) = scidev_lstm(input_t, (input_h, input_c))

        torch_lstm.zero_grad()
        output_torch_c.backward(torch.ones_like(output_torch_c))
        scidev_lstm.backward(input_t, torch.ones_like(output_scidev_h))

        grads = scidev_lstm.get_grad_test()

        assert input_t.grad.shape == grads['gradx'].shape
        assert torch.all(
            torch.abs(grads['gradx'] - input_t.grad) < 1e-5)

        assert input_h.grad.shape == grads['gradh'].shape
        assert torch.all(
            torch.abs(grads['gradh'] - input_h.grad) < 1e-5)

        assert input_c.grad.shape == grads['gradc'].shape
        assert torch.all(
            torch.abs(grads['gradc'] - input_c.grad) < 1e-5)

        assert torch_lstm.weight_hh.grad.shape == grads['gradWh'].shape
        assert torch.all(
            torch.abs(grads['gradWh'] - torch_lstm.weight_hh.grad) < 1e-5)

        assert torch_lstm.weight_ih.grad.shape == grads['gradWi'].shape
        assert torch.all(
            torch.abs(grads['gradWi'] - torch_lstm.weight_ih.grad) < 1e-5)

        if bias:
            assert torch_lstm.bias_hh.grad.shape == grads['gradbh'].shape
            assert torch.all(
                torch.abs(grads['gradbh'] - torch_lstm.bias_hh.grad) < 1e-5)

            assert torch_lstm.bias_ih.grad.shape == grads['gradbi'].shape
            assert torch.all(
                torch.abs(grads['gradbi'] - torch_lstm.bias_ih.grad) < 1e-5)
