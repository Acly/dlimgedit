import itertools
import torch

from . import workbench

torch.set_printoptions(precision=2, linewidth=200, edgeitems=8)


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


def add_variance_epsilon(state: dict[str, torch.Tensor], epsilon=1e-5):
    for k in state:
        if k.endswith("running_var"):
            state[k] = state[k].add_(1e-5).contiguous()


def test_conv_2d_batch_norm():
    conv2dbn = Conv2d_BN(4, 6, ks=3, stride=2, pad=1)
    state = workbench.randomize(conv2dbn.state_dict())
    conv2dbn.load_state_dict(state)
    conv2dbn.eval()

    x = torch.rand(1, 4, 8, 8)
    expected = conv2dbn(x)

    add_variance_epsilon(state)
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
    state = workbench.randomize(mb_conv.state_dict())
    mb_conv.load_state_dict(state)
    mb_conv.eval()

    x = torch.rand(1, 4, 8, 8)
    expected = mb_conv(x)

    add_variance_epsilon(state)
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
    state = workbench.randomize(patch_merging.state_dict())
    patch_merging.load_state_dict(state)
    patch_merging.eval()

    x = torch.rand(1, 8, 32, 32)
    expected = patch_merging(x)

    add_variance_epsilon(state)
    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("patch_merging", x, result, state)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class Mlp(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=torch.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = torch.nn.LayerNorm(in_features)
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def test_mlp():
    mlp = Mlp(4, 16)
    state = workbench.randomize(mlp.state_dict())
    mlp.load_state_dict(state)
    mlp.eval()

    x = torch.rand(1, 6, 4)
    expected = mlp(x)

    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("mlp", x, result, state)

    # precision: ggml_gelu uses fp16 look-up table & tanh approximation
    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
    ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, h)
        self.proj = torch.nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.register_buffer(
                "ab",
                self.attention_biases[:, self.attention_bias_idxs],
                persistent=False,
            )

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        qkv = qkv.view(B, N, self.num_heads, -1)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        attn = attn + (
            self.attention_biases[:, self.attention_bias_idxs]
            if self.training
            else self.ab
        )
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, self.dh)
        x = self.proj(x)
        return x


def test_attention():
    attention = Attention(4, 2, num_heads=2, attn_ratio=1, resolution=(3, 3))
    state = workbench.randomize(attention.state_dict())
    attention.load_state_dict(state)
    attention.eval()

    x = torch.rand(4, 9, 4)
    expected = attention(x)

    state["attention_biases_indexed"] = state["attention_biases"][
        :, attention.attention_bias_idxs
    ]
    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("attention", x, result, state)

    assert torch.allclose(result, expected, atol=0.001)


class TinyViTBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=torch.nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = torch.nn.Identity()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(
            dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_activation,
            drop=drop,
        )

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = (
                x.view(B, nH, self.window_size, nW, self.window_size, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size * self.window_size, C)
            )

            x = self.attn(x)

            # window reverse
            x = (
                x.view(B, nH, nW, self.window_size, self.window_size, C)
                .transpose(2, 3)
                .reshape(B, pH, pW, C)
            )

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x


def test_tiny_vit_block():
    tiny_vit_block = TinyViTBlock(4, (8, 8), num_heads=2, window_size=5)
    state = workbench.randomize(tiny_vit_block.state_dict())
    tiny_vit_block.load_state_dict(state)
    tiny_vit_block.eval()

    x = torch.rand(1, 64, 4)
    expected = tiny_vit_block(x)

    state["attn.attention_biases_indexed"] = state["attn.attention_biases"][
        :, tiny_vit_block.attn.attention_bias_idxs
    ]
    add_variance_epsilon(state)
    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("tiny_vit_block", x, result, state)

    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)
