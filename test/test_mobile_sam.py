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


class PatchEmbed(torch.nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: tuple[int, int] = (
            resolution
            if isinstance(resolution, tuple)
            else (
                resolution,
                resolution,
            )
        )
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = torch.nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


def test_patch_embed():
    patch_embed = PatchEmbed(3, 4, (8, 8), torch.nn.GELU)
    state = workbench.randomize(patch_embed.state_dict())
    patch_embed.load_state_dict(state)
    patch_embed.eval()

    x = torch.rand(1, 3, 8, 8)
    expected = patch_embed(x)

    add_variance_epsilon(state)
    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("patch_embed", x, result, state)

    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def test_layer_norm_2d():
    layer_norm = LayerNorm2d(4)
    state = workbench.randomize(layer_norm.state_dict())
    layer_norm.load_state_dict(state)
    layer_norm.eval()

    x = torch.rand(1, 4, 8, 8)
    expected = layer_norm(x)

    result = torch.zeros_like(expected).contiguous()
    workbench.invoke_test("layer_norm_2d", x, result, state)

    assert torch.allclose(result, expected, rtol=0.001, atol=0.02)


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


class ConvLayer(torch.nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = torch.nn.ModuleList(
            [
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    activation,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:

            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayer(torch.nn.Module):

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=torch.nn.GELU,
        out_dim=None,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = torch.nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    local_conv_size=local_conv_size,
                    activation=activation,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:

            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyViT(torch.nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = torch.nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            resolution=img_size,
            activation=activation,
        )

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0]
                    // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    patches_resolution[1]
                    // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                ),
                #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                #                     patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # Classifier head
        self.norm_head = torch.nn.LayerNorm(embed_dims[-1])
        self.head = (
            torch.nn.Linear(embed_dims[-1], num_classes)
            if num_classes > 0
            else torch.nn.Identity()
        )

        # init weights
        # self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            torch.nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        # print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, "lr_scale"), p.param_name

        self.apply(_check_lr_scale)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"attention_biases"}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B, _, C = x.size()
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.norm_head(x)
        # x = self.head(x)
        return x


def test_tiny_vit():
    tiny_vit = TinyViT(
        img_size=1024,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
    )
    state = workbench.randomize(tiny_vit.state_dict())
    tiny_vit.load_state_dict(state)
    tiny_vit.eval()

    x = torch.rand(1, 3, 1024, 1024)
    expected = tiny_vit(x)

    # !! running out of memory when using default resolution on workbench

    # add_variance_epsilon(state)
    # result = torch.zeros_like(expected).contiguous()
    # workbench.invoke_test("tiny_vit", x, result, state)

    # assert torch.allclose(result, expected, rtol=0.001, atol=0.02)
