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
    GGML_ASSERT(dim % num_heads == 0);
    int key_dim = dim / num_heads;
    int B = x->ne[2];
    int N = x->ne[1];

    x = layer_norm(m.group("norm"), x);

    Tensor qkv = linear(m.group("qkv"), x);
    qkv = ggml_reshape_4d(m, qkv, key_dim, 3, num_heads * N, B); // [B, N * num_heads, 3, key_dim]
    qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 3, 1, 2));        // [3, B, N * num_heads, key_dim]

    // split([key_dim, key_dim, key_dim], dim=3)
    size_t offset = qkv->nb[3];
    auto split = [=](Model m, Tensor tensor, size_t index) {
        tensor = ggml_view_3d(m, tensor, key_dim, num_heads * N, B, tensor->nb[1], tensor->nb[2],
                              index * offset);
        tensor = ggml_reshape_4d(m, tensor, key_dim, num_heads, N, B);
        return tensor;
    };

    Tensor q = split(m, qkv, 0);
    Tensor k = split(m, qkv, 1);
    Tensor v = split(m, qkv, 2);
    q = ggml_cont(m, ggml_permute(m, q, 0, 2, 1, 3));
    k = ggml_cont(m, ggml_permute(m, k, 0, 2, 1, 3));
    v = ggml_cont(m, ggml_permute(m, v, 1, 2, 0, 3)); // transpose for mul_mat later

    Tensor attn = ggml_mul_mat(m, k, q); // q @ k (k is transposed in mul_mat)
    attn = ggml_scale_inplace(m, attn, 1.0f / std::sqrtf(float(key_dim)));
    attn = ggml_add_inplace(m, attn, m["attention_biases_indexed"]);
    attn = ggml_soft_max(m, attn);

    x = ggml_mul_mat(m, v, attn);                     // attn @ v
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3)); // transpose(1, 2)
    x = ggml_reshape_3d(m, x, key_dim * num_heads, N, B);
    x = linear(m.group("proj"), x);
    return x;
}

inline Tensor tiny_vit_block(Model m, Tensor x, int input_resolution, int dim, int num_heads,
                             int window_size) {
    int H = input_resolution;
    int W = input_resolution;
    int B = x->ne[2];
    int L = x->ne[1];
    int C = x->ne[0];
    GGML_ASSERT(L == H * W);
    GGML_ASSERT(H != window_size && W != window_size);

    Tensor res_x = x;
    x = ggml_reshape_4d(m, x, C, W, H, B);

    // window partition
    x = ggml_win_part(m, x, window_size);
    x = ggml_reshape_3d(m, x, C, window_size * window_size, x->ne[3]);

    x = attention(m.group("attn"), x, dim, num_heads);

    // window reverse
    x = ggml_reshape_4d(m, x, C, window_size, window_size, x->ne[2]);
    x = ggml_win_unpart(m, x, W, H, window_size);

    x = ggml_reshape_3d(m, x, C, L, B);
    x = ggml_add_inplace(m, x, res_x);

    x = ggml_cont(m, ggml_transpose(m, x));
    x = ggml_reshape_4d(m, x, W, H, C, B);

    x = conv_2d_batch_norm(m.group("local_conv"), x, 1, 1, 1, /* groups */ dim);
    x = ggml_reshape_3d(m, x, L, C, B);
    x = ggml_cont(m, ggml_transpose(m, x));

    Tensor x_mlp = mlp(m.group("mlp"), x);
    x = ggml_add_inplace(m, x, x_mlp);
    return x;
}

} // namespace dlimg