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


conv2d_cases = {
    "1x1": dict(kernel_size=1, bias=False),
    "1x1_bias": dict(kernel_size=1, bias=True),
    "3x3": dict(kernel_size=3, bias=False),
    "3x3_bias": dict(kernel_size=3, bias=True),
    "stride2_pad1": dict(kernel_size=3, bias=False, stride=2, padding=1),
    "stride2_pad1_5x5": dict(kernel_size=5, bias=False, stride=2, padding=1),
}


@pytest.mark.parametrize("scenario", conv2d_cases.keys())
@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_conv_2d_channels(scenario: str, backend: str):
    s = conv2d_cases[scenario]
    kernel_size = s["kernel_size"]
    has_bias = s.get("bias", False)
    stride = s.get("stride", 1)
    padding = s.get("padding", 0)

    x = torch.arange(3 * 6 * 4).float().reshape(1, 6, 4, 3).permute(0, 3, 1, 2)
    kernel = (
        torch.arange(2 * 3 * kernel_size * kernel_size)
        .reshape(2, 3, kernel_size, kernel_size)
        .float()
    )  # .rand(2, 3, kernel_size, kernel_size)
    bias = None
    args = dict(weight=kernel)
    if has_bias:
        bias = torch.tensor([7, 21]).float()
        args["bias"] = bias
    expected = torch.nn.functional.conv2d(
        x, kernel, bias=bias, stride=stride, padding=padding
    )

    x = to_channel_last(x)
    args["weight"] = kernel.permute(0, 2, 3, 1)

    result = to_channel_last(torch.zeros_like(expected))
    workbench.invoke_test(f"conv_2d_channels_{scenario}", x, result, args, backend)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected, rtol=1e-5 if backend == "cpu" else 1e-3)


@pytest.mark.parametrize("scenario", ["stride_1_pad_0", "stride_2_pad_1"])
@pytest.mark.parametrize("memory_layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("batch", ["single", "batch"])
@pytest.mark.parametrize("backend", ["cpu", "vulkan"])
def test_depthwise_conv_2d(scenario: str, memory_layout: str, batch: str, backend: str):
    stride, pad = {"stride_1_pad_0": (1, 0), "stride_2_pad_1": (2, 1)}[scenario]
    x1 = torch.tensor([[1, 2, 2, 1], [4, 4, 4, 4], [0, 2, 2, 4], [1, 1, 1, 1]]).float()
    k = torch.tensor(
        [
            [[[1, 0, 1], [0, 1, 0], [1, 0, 1]]],
            [[[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
            [[[0, -1, 0], [-1, 1, -1], [-1, 0, -1]]],
        ]
    ).float()

    x = torch.stack((x1, x1 * 2.0, x1 * 0.5)).reshape(1, 3, 4, 4)
    if batch == "batch":
        x = torch.cat((x, x * -1), dim=0)

    expected = torch.nn.functional.conv2d(x, k, stride=stride, padding=pad, groups=3)

    result = torch.zeros_like(expected)
    if memory_layout == "nhwc":
        x = to_channel_last(x)
        k = k.permute(2, 3, 1, 0)
        result = to_channel_last(result)
    testcase = f"conv_2d_depthwise_{memory_layout}_{scenario}"
    workbench.invoke_test(testcase, x, result, dict(weight=k), backend)
    if memory_layout == "nhwc":
        result = revert_channel_last(result)

    assert torch.allclose(result, expected)


@pytest.mark.parametrize("scenario", ["3x3", "5x5", "stride2"])
def test_conv_transpose_2d(scenario:str):
    ksize, stride = {
        "3x3": (3, 1),
        "5x5": (5, 1),
        "stride2": (3, 2),
    }[scenario]
    x = torch.arange(11 * 4 * 5).reshape(1, 11, 4, 5).float()
    weight = torch.arange(11 * 2 * ksize * ksize).reshape(11, 2, ksize, ksize).float()
    bias = None
    expected = torch.nn.functional.conv_transpose2d(x, weight, bias, stride=stride)

    x = to_channel_last(x)  # -> [N, H, W, C_in]
    weight = weight.permute(1, 2, 3, 0).contiguous()  # -> [C_out, H, W, C_in]
    result = to_channel_last(torch.zeros_like(expected))

    workbench.invoke_test(
        f"conv_transpose_2d_{scenario}", x, result, dict(weight=weight), backend="vulkan"
    )
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)


def test_batch_norm_2d():
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(3)
    bias = torch.rand(3)
    mean = torch.rand(3)
    var = torch.arange(1, 4).float()
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
