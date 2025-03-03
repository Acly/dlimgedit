import pytest
import torch

from . import workbench


def test_linear():
    x = torch.rand(2, 5)
    weight = torch.rand(3, 5)
    bias = torch.tensor([7, 21, -5]).float()
    result = torch.zeros(2, 3)

    workbench.invoke_test("linear", x, result, {}, weight=weight, bias=bias)

    expected = torch.nn.functional.linear(x, weight, bias)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("bias_mode", ["with_bias", "without_bias"])
def test_conv_2d(kernel_size: int, bias_mode: str):
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(2, 3, kernel_size, kernel_size)
    bias = None
    args = dict(weight=weight)
    if bias_mode == "with_bias":
        bias = torch.tensor([7, 21]).float()
        args["bias"] = bias
    expected = torch.nn.functional.conv2d(x, weight, bias=bias)

    result = torch.zeros_like(expected)
    workbench.invoke_test("conv_2d", x, result, args)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("bias_mode", ["with_bias", "without_bias"])
def test_conv_2d_channels(kernel_size: int, bias_mode: str):
    x = torch.rand(1, 3, 4, 5)
    kernel = torch.rand(2, 3, kernel_size, kernel_size)
    bias = None
    args = dict(weight=kernel)
    if bias_mode == "with_bias":
        bias = torch.tensor([7, 21]).float()
        args["bias"] = bias
    expected = torch.nn.functional.conv2d(x, kernel, bias=bias)

    x = x.permute(0, 2, 3, 1).contiguous()
    args["weight"] = kernel.permute(2, 3, 1, 0).contiguous()

    result = torch.zeros_like(expected).permute(0, 2, 3, 1).contiguous()
    workbench.invoke_test("conv_2d_channels", x, result, args)
    result = result.permute(0, 3, 1, 2).contiguous()

    workbench.print_results(result, expected)
    assert torch.allclose(result, expected)


def test_conv_2d_depth_wise():
    x1 = torch.tensor([[1, 2, 2, 1], [4, 4, 4, 4], [0, 2, 2, 4], [1, 1, 1, 1]]).float()
    k = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).float()

    x = torch.cat((x1, x1 * 2.0, x1 * 0.5), dim=1).reshape(1, 3, 4, 4)
    k = k.repeat(3, 1, 1, 1)
    result = torch.zeros(1, 3, 2, 2)

    workbench.invoke_test("conv_2d_depth_wise", x, result, dict(weight=k))

    expected = torch.nn.functional.conv2d(x, k, groups=3)
    assert torch.allclose(result, expected)


def test_batch_norm_2d():
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(3)
    bias = torch.rand(3)
    mean = torch.rand(3)
    var = torch.rand(3)
    expected = torch.nn.functional.batch_norm(x, mean, var, weight, bias, eps=1e-5)
    result = torch.zeros_like(expected)

    var = (var + 1e-5).sqrt()
    state = dict(weight=weight, bias=bias, running_mean=mean, running_var=var)
    workbench.invoke_test("batch_norm_2d", x, result, state)

    assert torch.allclose(result, expected)


def test_layer_norm():
    dim = 20
    x = torch.rand(4, 5, dim)
    weight = torch.rand(dim)
    bias = torch.rand(dim)
    result = torch.zeros(4, 5, dim)

    workbench.invoke_test("layer_norm", x, result, dict(weight=weight, bias=bias))

    expected = torch.nn.functional.layer_norm(x, [dim], weight, bias, eps=1e-5)
    assert torch.allclose(result, expected, atol=1e-6)
