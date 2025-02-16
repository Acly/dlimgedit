import torch

from . import workbench

torch.set_printoptions(precision=1, linewidth=160, edgeitems=8)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


def test_conv_2d_batch_norm():
    conv2dbn = Conv2d_BN(4, 6, ks=3, stride=2, pad=1)
    conv2dbn.eval()

    x = torch.rand(1, 4, 8, 8)
    state = {
        "c.weight": torch.rand_like(conv2dbn.c.weight),
        "bn.bias": torch.rand_like(conv2dbn.bn.bias),
        "bn.running_mean": torch.rand_like(conv2dbn.bn.running_mean),
        "bn.running_var": torch.rand_like(conv2dbn.bn.running_var),
        "bn.weight": torch.rand_like(conv2dbn.bn.weight),
    }

    conv2dbn.load_state_dict(state)
    expected = conv2dbn(x)

    state["bn.running_var"] = state["bn.running_var"].add_(1e-5).contiguous()
    result = torch.zeros_like(expected)
    workbench.invoke_test("conv_2d_batch_norm", x, result, state)

    assert torch.allclose(result, expected)


class MBConv(torch.nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation(approximate="tanh")

        self.conv2 = Conv2d_BN(
            self.hidden_chans,
            self.hidden_chans,
            ks=3,
            stride=1,
            pad=1,
            groups=self.hidden_chans,
        )
        self.act2 = activation(approximate="tanh")

        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation(approximate="tanh")

        self.drop_path = torch.nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


def test_mb_conv():
    mb_conv = MBConv(4, 4, 4.0, torch.nn.GELU, 0.2)
    mb_conv.eval()

    x = torch.rand(1, 4, 8, 8)
    state = mb_conv.state_dict()
    state = {
        k: torch.rand_like(v) for k, v in state.items() if v.dtype.is_floating_point
    }
    mb_conv.load_state_dict(state)
    expected = mb_conv(x)

    for k in state:
        if k.endswith("running_var"):
            state[k] = state[k].add_(1e-5).contiguous()
    result = torch.zeros_like(expected)
    workbench.invoke_test("mb_conv", x, result, state)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class PatchMerging(torch.nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if out_dim == 320 or out_dim == 448 or out_dim == 576:
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def test_patch_merging():
    patch_merging = PatchMerging((32, 32), 8, 16, torch.nn.GELU)
    patch_merging.eval()

    x = torch.rand(1, 8, 32, 32)
    state = patch_merging.state_dict()
    state = {
        k: torch.rand_like(v) for k, v in state.items() if v.dtype.is_floating_point
    }
    patch_merging.load_state_dict(state)
    expected = patch_merging(x)

    for k in state:
        if k.endswith("running_var"):
            state[k] = state[k].add_(1e-5).contiguous()
    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("patch_merging", x, result, state)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)
