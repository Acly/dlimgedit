import math
import torch
from torch import nn

from . import workbench
from .workbench import to_channel_last, revert_channel_last, convert_to_channel_last
from .workbench import input_tensor, generate_state

torch.set_printoptions(precision=3, sci_mode=False)

# Reference: https://github.com/chaiNNer-org/spandrel/blob/main/libs/spandrel/spandrel/architectures/ESRGAN/__arch/RRDB.py

def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size,
    stride=1,
    pad=1,
    dilation=1,
    groups=1,
    bias=True,
    act=True,
):
    m = [
        nn.Conv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
    ]
    if act:
        m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
    return nn.Sequential(*m)


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def upconv_block(
    in_nc: int, out_nc: int, upscale_factor=2, kernel_size=3, stride=1, mode="nearest"
):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride)
    return sequential(upsample, conv)


def test_upconv():
    block = upconv_block(3, 6, upscale_factor=2, kernel_size=3, stride=1)
    state = generate_state(block.state_dict())
    block.load_state_dict(state)
    block.eval()
    x = input_tensor(1, 3, 2, 2)
    expected = block(x)

    x = to_channel_last(x)
    state = convert_to_channel_last(state, "1.")
    result = to_channel_last(torch.zeros_like(expected))
    result = workbench.invoke_test("esrgan_upconv", x, result, state)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, kernel_size=3, gc=32, stride=1, bias=True, plus=False):
        super().__init__()

        self.conv1x1 = None  # conv1x1(nf, gc) if plus else None

        self.conv1 = conv_block(nf, gc, kernel_size, stride, bias=bias)
        self.conv2 = conv_block(nf + gc, gc, kernel_size, stride, bias=bias)
        self.conv3 = conv_block(nf + 2 * gc, gc, kernel_size, stride, bias=bias)
        self.conv4 = conv_block(nf + 3 * gc, gc, kernel_size, stride, bias=bias)
        self.conv5 = conv_block(nf + 4 * gc, nf, 3, stride, bias=bias, act=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


def test_residual_dense_block():
    block = ResidualDenseBlock_5C(nf=8, kernel_size=3, gc=4)
    state = generate_state(block.state_dict())
    block.load_state_dict(state)
    block.eval()
    x = 0.1 * (input_tensor(1, 8, 6, 6) - 0.5)
    expected = block(x)

    x = to_channel_last(x)
    state = convert_to_channel_last(state, "conv")
    result = to_channel_last(torch.zeros_like(expected))
    result = workbench.invoke_test("esrgan_residual_dense_block", x, result, state)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)


class RRDB(nn.Module):

    def __init__(self, nf, kernel_size=3, gc=32, stride=1, bias: bool = True):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias)
        self.RDB2 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias)
        self.RDB3 = ResidualDenseBlock_5C(nf, kernel_size, gc, stride, bias)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


def test_rrdb():
    block = RRDB(nf=8, kernel_size=3, gc=4)
    state = workbench.randomize(block.state_dict())
    for name, param in state.items():
        if "conv" in name and param.ndim == 4:
            state[name] = param - 0.5
    block.load_state_dict(state)
    block.eval()
    x = 0.1 * input_tensor(1, 8, 6, 6)
    expected = block(x)

    x = to_channel_last(x)
    state = convert_to_channel_last(state, "conv")
    result = to_channel_last(torch.zeros_like(expected))
    result = workbench.invoke_test("esrgan_rrdb", x, result, state)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected, atol=1e-5)


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super().__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = "Identity + \n|"
        modstr = self.sub.__repr__().replace("\n", "\n|")
        tmpstr = tmpstr + modstr
        return tmpstr


class RRDBNet(nn.Module):
    hyperparameters = {}

    def __init__(
        self,
        *,
        in_nc: int = 3,
        out_nc: int = 3,
        num_filters: int = 64,
        num_blocks: int = 23,
        scale: int = 4,
        shuffle_factor: int | None = None,
        upsampler: str = "upconv",
    ):
        super().__init__()

        self.shuffle_factor = shuffle_factor
        self.scale = scale

        upsample_block = {
            "upconv": upconv_block,
            # "pixel_shuffle": B.pixelshuffle_block,
        }.get(upsampler)
        if upsample_block is None:
            raise NotImplementedError(f"Upsample mode [{upsampler}] is not found")

        if scale == 3:
            upsample_blocks = upsample_block(
                in_nc=num_filters, out_nc=num_filters, upscale_factor=3
            )
        else:
            upsample_blocks = [
                upsample_block(in_nc=num_filters, out_nc=num_filters)
                for _ in range(int(math.log(scale, 2)))
            ]

        self.model = sequential(
            # fea conv
            conv_block(in_nc=in_nc, out_nc=num_filters, kernel_size=3, act=False),
            ShortcutBlock(
                sequential(
                    # rrdb blocks
                    *[
                        RRDB(nf=num_filters, kernel_size=3, gc=32, stride=1, bias=True)
                        for _ in range(num_blocks)
                    ],
                    # lr conv
                    conv_block(
                        in_nc=num_filters, out_nc=num_filters, kernel_size=3, act=False
                    ),
                )
            ),
            *upsample_blocks,
            # hr_conv0
            conv_block(in_nc=num_filters, out_nc=num_filters, kernel_size=3),
            # hr_conv1
            conv_block(in_nc=num_filters, out_nc=out_nc, kernel_size=3, act=False),
        )

    def forward(self, x):
        return self.model(x)


def test_rrdbnet():
    torch.manual_seed(42)

    model = RRDBNet(
        in_nc=3, out_nc=3, num_filters=8, num_blocks=2, scale=2, upsampler="upconv"
    )
    state = workbench.randomize(model.state_dict())
    for name, param in state.items():
        if param.ndim == 4 and name.endswith("weight"):
            state[name] = param * (1 / 100)
    model.load_state_dict(state)
    model.eval()
    x = input_tensor(1, 3, 6, 6)
    expected = model(x)

    x = to_channel_last(x)
    state = convert_to_channel_last(state, ".")
    result = to_channel_last(torch.zeros_like(expected))
    print(result.shape, expected.shape)
    result = workbench.invoke_test("esrgan_rrdbnet", x, result, state)
    result = revert_channel_last(result)

    workbench.print_results(result, expected)
    assert torch.allclose(result, expected, atol=1e-4)
