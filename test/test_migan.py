import torch
import torchvision
import numpy as np

from torch import Tensor, nn
from torch.nn import functional as F

from . import workbench
from .workbench import to_channel_last, revert_channel_last, convert_to_channel_last
from .workbench import input_tensor, generate_state

torch.set_printoptions(precision=3, sci_mode=False)


class lrelu_agc:
    def __init__(self, alpha=0.2, gain=1, clamp=None):
        self.alpha = alpha
        if gain == "sqrt_2":
            self.gain = np.sqrt(2)
        else:
            self.gain = gain
        self.clamp = clamp

    def __call__(self, x, gain=1):
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True)
        act_gain = self.gain * gain
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x


def test_lrelu_agc():
    x = input_tensor(1, 3, 8, 8) * 2.0 - 1.0
    lrelu = lrelu_agc(alpha=0.2, gain="sqrt_2", clamp=1.0)
    expected = lrelu(x.clone())

    result = torch.zeros_like(x)
    result = workbench.invoke_test("migan_lrelu_agc", x, result, {})

    assert torch.allclose(result, expected)


def setup_filter(
    f,
    device=torch.device("cpu"),
    normalize=True,
    flip_filter=False,
    gain=1,
    separable=None,
):
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


class Downsample2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            padding=1,
            bias=False,
            stride=2,
        )
        f = setup_filter([1, 3, 3, 1], gain=1)
        self.filter.weight = nn.Parameter(
            f.repeat([*self.filter.weight.shape[:2], 1, 1])
        )

    def forward(self, x):
        x = self.filter(x)
        return x


def test_downsample2d():
    x = input_tensor(1, 4, 8, 8)
    downsample = Downsample2d(in_channels=4)
    expected = downsample(x)
    state = downsample.state_dict()
    state = convert_to_channel_last(state, key="filter.")

    x = to_channel_last(x)
    result = to_channel_last(torch.zeros_like(expected))
    result = workbench.invoke_test("migan_downsample_2d", x, result, state)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)


class Upsample2d(nn.Module):
    def __init__(self, in_channels, resolution=None):
        super().__init__()
        self.nearest_up = nn.Upsample(scale_factor=2, mode="nearest")
        w = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        assert resolution is not None
        self.register_buffer(
            "filter_const", w.repeat(1, 1, resolution // 2, resolution // 2)
        )

        self.filter = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            groups=in_channels,
            bias=False,
        )

        f = setup_filter([1, 3, 3, 1], gain=4)
        self.filter.weight = nn.Parameter(
            f.repeat([*self.filter.weight.shape[:2], 1, 1])
        )

    def forward(self, x):
        x = self.nearest_up(x)
        x = x * self.filter_const
        x = F.pad(x, pad=(2, 1, 2, 1))
        x = self.filter(x)
        return x


def test_upsample2d():
    x = input_tensor(1, 5, 4, 4)
    upsample = Upsample2d(in_channels=5, resolution=8)
    expected = upsample(x)
    state = upsample.state_dict()
    state = convert_to_channel_last(state, key="filter.")

    x = to_channel_last(x)
    result = to_channel_last(torch.zeros_like(expected))
    result = workbench.invoke_test("migan_upsample_2d", x, result, state)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)


class SeparableConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bias=True,
        activation=None,
        resolution=None,
        use_noise=False,
        down=1,
        up=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
            groups=in_channels,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            groups=1,
        )

        self.downsample = None
        if down > 1:
            self.downsample = Downsample2d(in_channels)

        self.upsample = None
        if up > 1:
            self.upsample = Upsample2d(out_channels, resolution=resolution)

        self.use_noise = use_noise
        if use_noise:
            assert resolution is not None
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.downsample is not None:
            x = self.downsample(x)
        x = self.conv2(x)
        if self.upsample is not None:
            x = self.upsample(x)

        if self.use_noise:
            noise = self.noise_const * self.noise_strength
            x = x.add_(noise)
        if self.activation is not None:
            x = self.activation(x)
        return x


def test_separable_conv2d():
    separable_conv = SeparableConv2d(
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        bias=True,
        activation=lrelu_agc(alpha=0.2, gain="sqrt_2", clamp=256),
        resolution=8,
        use_noise=True,
    )
    x = input_tensor(1, 3, 8, 8)
    state = generate_state(separable_conv.state_dict())
    separable_conv.load_state_dict(state)
    expected = separable_conv(x)

    state = convert_to_channel_last(state, key="conv")    
    state["noise_strength"] = torch.tensor([0.5])
    x = to_channel_last(x)
    result = to_channel_last(torch.zeros_like(expected))
    result = workbench.invoke_test("migan_separable_conv_2d", x, result, state)
    result = revert_channel_last(result)

    assert torch.allclose(result, expected)
