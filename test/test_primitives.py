import pytest
import torch

from . import workbench


@pytest.mark.parametrize("bias_mode", ["with_bias", "without_bias"])
def test_conv_2d(bias_mode: str):
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(2, 3, 3, 3)
    bias = None
    inputs = [x, weight]
    if bias_mode == "with_bias":
        bias = torch.tensor([7, 21]).float()
        inputs.append(bias)
    result = torch.zeros(1, 2, 2, 3)

    workbench.invoke_test("conv_2d", inputs, result)

    expected = torch.nn.functional.conv2d(x, weight, bias=bias)
    assert torch.allclose(result, expected)


def test_conv_2d_depth_wise():
    x1 = torch.tensor([[1, 2, 2, 1], [4, 4, 4, 4], [0, 2, 2, 4], [1, 1, 1, 1]]).float()
    k = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]]).float()

    x = torch.cat((x1, x1 * 2.0, x1 * 0.5), dim=1).reshape(1, 3, 4, 4)
    k = k.repeat(3, 1, 1, 1)
    result = torch.zeros(1, 3, 2, 2)

    workbench.invoke_test("conv_2d_depth_wise", [x, k], result)

    expected = torch.nn.functional.conv2d(x, k, groups=3)
    assert torch.allclose(result, expected)


def test_batch_norm_2d():
    x = torch.rand(1, 3, 4, 5)
    weight = torch.rand(3)
    bias = torch.rand(3)
    mean = torch.rand(3)
    var = torch.rand(3)
    result = torch.zeros(1, 3, 4, 5)

    workbench.invoke_test("batch_norm_2d", [x, weight, bias, mean, var], result)

    expected = torch.nn.functional.batch_norm(x, mean, var, weight, bias, eps=0)
    assert torch.allclose(result, expected)
