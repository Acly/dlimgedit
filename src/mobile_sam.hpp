#include "ml.hpp"
#include "primitives.hpp"

#include <ggml.h>

namespace dlimg {

inline Tensor conv_2d_batch_norm(Model m, Tensor x, int stride = 1, int pad = 0, int dilation = 1,
                                 int groups = 1) {
    if (groups == 1) {
        x = conv_2d(m.group("c"), x, stride, pad, dilation);
    } else {
        x = conv_2d_depth_wise(m.group("c"), x, stride, pad, dilation);
    }
    x = batch_norm_2d(m.group("bn"), x);
    return x;
}

inline Tensor patch_embed(Model m, Tensor x) {
    x = conv_2d_batch_norm(m.group("seq.0"), x, 2, 1);
    x = ggml_gelu_inplace(m, x);
    x = conv_2d_batch_norm(m.group("seq.2"), x, 2, 1);
    return x;
}

inline Tensor mb_conv(Model m, Tensor x) {
    Tensor shortcut = x;

    x = conv_2d_batch_norm(m.group("conv1"), x);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m.group("conv2"), x, 1, 1, 1, /* groups */ x->ne[2]);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m.group("conv3"), x);
    x = ggml_add_inplace(m, x, shortcut);
    x = ggml_gelu_inplace(m, x);

    return x;
}

inline Tensor patch_merging(Model m, Tensor x, int input_resolution) {
    if (x->ne[2] == 1) {
        x = ggml_reshape_4d(m, x, x->ne[0], input_resolution, input_resolution, x->ne[3]);
        x = ggml_permute(m, x, 2, 0, 1, 3); // -> B C H W
    }
    x = conv_2d_batch_norm(m.group("conv1"), x);
    x = ggml_gelu_inplace(m, x);

    int out_c = m["conv2.c.weight"]->ne[0];
    int stride = (out_c == 320 || out_c == 448 || out_c == 576) ? 1 : 2;
    x = conv_2d_batch_norm(m.group("conv2"), x, stride, 1, 1, out_c);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m.group("conv3"), x);
    x = ggml_reshape_3d(m, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]); // flatten(2)
    x = ggml_transpose(m, x);
    return x;
}

inline Tensor mlp(Model m, Tensor x) {
    x = layer_norm(m.group("norm"), x);

    x = linear(m.group("fc1"), x);
    x = ggml_gelu_inplace(m, x);
    x = linear(m.group("fc2"), x);
    return x;
}

inline Tensor attention(Model m, Tensor x, int dim, int num_heads) {
    x = layer_norm(m.group("norm"), x);

    GGML_ASSERT(dim % num_heads == 0);
    int key_dim = dim / num_heads;
    int B = x->ne[3];
    int N = x->ne[2];

    Tensor qkv = linear(m.group("qkv"), x);
    qkv = ggml_reshape_4d(m, qkv, key_dim * 3, num_heads - 1, N, B);

    // ggml_tesnor* v = ...

    Tensor attn = nullptr;
    attn = ggml_soft_max_inplace(m, attn);
    // x = ggml_mul_mat(c, attn, v);
    x = ggml_transpose(m, x);
    x = ggml_reshape_3d(m, x, key_dim * num_heads, N, B);
    x = linear(m.group("proj"), x);
    return x;
}

} // namespace dlimg