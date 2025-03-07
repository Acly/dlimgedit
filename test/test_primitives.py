import pytest
import torch

from . import workbench
from .workbench import revert_channel_last, to_channel_last


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

    x = to_channel_last(x)
    args["weight"] = kernel.permute(2, 3, 1, 0)

    result = to_channel_last(torch.zeros_like(expected))
    workbench.invoke_test("conv_2d_channels", x, result, args)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("scenario", ["stride_1_pad_0", "stride_2_pad_1"])
@pytest.mark.parametrize("memory_layout", ["nchw", "nhwc"])
def test_conv_2d_depth_wise(scenario: str, memory_layout: str):
    stride, pad = {
        "stride_1_pad_0": (1, 0),
        "stride_2_pad_1": (2, 1),
    }[scenario]
    x1 = torch.tensor([[1, 2, 2, 1], [4, 4, 4, 4], [0, 2, 2, 4], [1, 1, 1, 1]]).float()
    k = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).float()

    x = torch.stack((x1, x1 * 2.0, x1 * 0.5)).reshape(1, 3, 4, 4)
    k = k.repeat(3, 1, 1, 1)
    expected = torch.nn.functional.conv2d(x, k, stride=stride, padding=pad, groups=3)

    result = torch.zeros_like(expected)
    if memory_layout == "nhwc":
        x = to_channel_last(x)
        k = k.permute(2, 3, 0, 1)
        result = to_channel_last(result)
    workbench.invoke_test(f"conv_2d_depthwise_{memory_layout}_{scenario}", x, result, dict(weight=k))
    if memory_layout == "nhwc":
        result = revert_channel_last(result)

    assert torch.allclose(result, expected)


def test_batch_norm_2d():
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(3)
    bias = torch.rand(3)
    mean = torch.rand(3)
    var = torch.rand(3)
    expected = torch.nn.functional.batch_norm(x, mean, var, weight, bias, eps=1e-5)

    x = to_channel_last(x)
    result = to_channel_last(torch.zeros_like(expected))

    var = (var + 1e-5).sqrt()
    state = dict(weight=weight, bias=bias, running_mean=mean, running_var=var)
    workbench.invoke_test("batch_norm_2d", x, result, state)
    result = revert_channel_last(result)

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
