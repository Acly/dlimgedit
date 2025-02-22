#include "image.hpp"
#include "ml.hpp"
#include "primitives.hpp"

#include <ggml.h>

namespace dlimg {

constexpr float pi = 3.14159265358979323846f;

inline Tensor conv_2d_batch_norm(Model m, Tensor x, int stride = 1, int pad = 0, int dilation = 1,
                                 int groups = 1) {
    if (groups == 1) {
        x = conv_2d(m["c"], x, stride, pad, dilation);
    } else {
        x = conv_2d_depth_wise(m["c"], x, stride, pad, dilation);
    }
    x = batch_norm_2d(m["bn"], x);
    return x;
}

inline Tensor patch_embed(Model m, Tensor x) {
    x = conv_2d_batch_norm(m["seq.0"], x, 2, 1);
    x = ggml_gelu_inplace(m, x);
    x = conv_2d_batch_norm(m["seq.2"], x, 2, 1);
    return x;
}

inline Tensor layer_norm_2d(Model m, Tensor x, float eps = 1e-6f) {
    Tensor weight = m.weights("weight");
    weight = ggml_reshape_3d(m, weight, 1, 1, weight->ne[0]);
    Tensor bias = m.weights("bias");
    bias = ggml_reshape_3d(m, bias, 1, 1, bias->ne[0]);

    x = ggml_cont(m, ggml_permute(m, x, 1, 2, 0, 3));
    x = ggml_norm(m, x, eps);
    x = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3));
    x = ggml_mul_inplace(m, x, weight);
    x = ggml_add_inplace(m, x, bias);
    return x;
}

inline Tensor mb_conv(Model m, Tensor x) {
    Tensor shortcut = x;

    x = conv_2d_batch_norm(m["conv1"], x);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m["conv2"], x, 1, 1, 1, /* groups */ x->ne[2]);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m["conv3"], x);
    x = ggml_add_inplace(m, x, shortcut);
    x = ggml_gelu_inplace(m, x);

    return x;
}

inline Tensor patch_merging(Model m, Tensor x, int input_resolution) {
    if (x->ne[2] == 1) {
        x = ggml_reshape_4d(m, x, x->ne[0], input_resolution, input_resolution, x->ne[3]);
        x = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3)); // -> B C H W
    }
    x = conv_2d_batch_norm(m["conv1"], x);
    x = ggml_gelu_inplace(m, x);

    int out_c = m.weights("conv2.c.weight")->ne[3];
    int stride = (out_c == 320 || out_c == 448 || out_c == 576) ? 1 : 2;
    x = conv_2d_batch_norm(m["conv2"], x, stride, 1, 1, out_c);
    x = ggml_gelu_inplace(m, x);

    x = conv_2d_batch_norm(m["conv3"], x);
    x = ggml_reshape_3d(m, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]); // flatten(2)
    x = ggml_transpose(m, x);
    return x;
}

inline Tensor mlp(Model m, Tensor x) {
    x = layer_norm(m["norm"], x);

    x = linear(m["fc1"], x);
    x = ggml_gelu_inplace(m, x);
    x = linear(m["fc2"], x);
    return x;
}

inline Tensor attention(Model m, Tensor x, int dim, int num_heads) {
    GGML_ASSERT(dim % num_heads == 0);
    int key_dim = dim / num_heads;
    int B = x->ne[2];
    int N = x->ne[1];

    x = layer_norm(m["norm"], x);

    Tensor qkv = linear(m["qkv"], x);
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
    attn = ggml_add_inplace(m, attn, m.weights("attention_biases_indexed"));
    attn = ggml_soft_max(m, attn);

    x = ggml_mul_mat(m, v, attn);                     // attn @ v
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3)); // transpose(1, 2)
    x = ggml_reshape_3d(m, x, key_dim * num_heads, N, B);
    x = linear(m["proj"], x);
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

    x = ggml_cont(m, x);
    Tensor res_x = x;
    x = ggml_reshape_4d(m, x, C, W, H, B);

    // window partition
    x = ggml_win_part(m, x, window_size);
    x = ggml_reshape_3d(m, x, C, window_size * window_size, x->ne[3]);

    x = attention(m["attn"], x, dim, num_heads);

    // window reverse
    x = ggml_reshape_4d(m, x, C, window_size, window_size, x->ne[2]);
    x = ggml_win_unpart(m, x, W, H, window_size);

    x = ggml_reshape_3d(m, x, C, L, B);
    x = ggml_add_inplace(m, x, res_x);

    x = ggml_cont(m, ggml_transpose(m, x));
    x = ggml_reshape_4d(m, x, W, H, C, B);

    x = conv_2d_batch_norm(m["local_conv"], x, 1, 1, 1, /* groups */ dim);
    x = ggml_reshape_3d(m, x, L, C, B);
    x = ggml_cont(m, ggml_transpose(m, x));

    Tensor x_mlp = mlp(m["mlp"], x);
    x = ggml_add_inplace(m, x, x_mlp);
    return x;
}

struct TinyViTParams {

    struct Layer {
        int resolution;
        int embed_dim;
        int depth;
        int num_heads;
        int window_size;
        bool downsample;
    };

    static constexpr int num_layers = 4;

    int img_size = 1024;
    // clang-format off
    std::array<Layer, num_layers> layers = {
        // reslution    dim     depth   attn heads  window size   downsample
        Layer{256,      64,     2,      2,          7,              true},
        Layer{128,      128,    2,      4,          7,              true},
        Layer{64,       160,    6,      5,          14,             true},
        Layer{64,       320,    2,      10,         7,              false}};
    // clang-format on
};

inline Tensor conv_layer(Model m, Tensor x, TinyViTParams::Layer p) {
    auto block = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        x = mb_conv(block[i], x);
    }
    x = patch_merging(m["downsample"], x, p.resolution);
    return x;
}

inline Tensor basic_layer(Model m, Tensor x, TinyViTParams::Layer const& p) {
    auto blocks = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        x = tiny_vit_block(blocks[i], x, p.resolution, p.embed_dim, p.num_heads, p.window_size);
    }
    if (p.downsample) {
        x = patch_merging(m["downsample"], x, p.resolution);
    }
    return x;
}

inline Tensor tiny_vit(Model m, Tensor x, TinyViTParams const& p) {
    x = patch_embed(m["patch_embed"], x);
    x = conv_layer(m["layers.0"], x, p.layers[0]);

    auto layers = m["layers"];
    for (int i = 1; i < p.num_layers; ++i) {
        x = basic_layer(layers[i], x, p.layers[i]);
    }

    int B = x->ne[2];
    int C = x->ne[0];
    x = ggml_reshape_4d(m, x, C, 64, 64, B);
    x = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3));

    // neck
    x = conv_2d(m["neck.0"], x);
    x = layer_norm_2d(m["neck.1"], x);
    x = conv_2d(m["neck.2"], x, 1, 1);
    x = layer_norm_2d(m["neck.3"], x);
    return x;
}

inline std::vector<float> preprocess_image(char const* filepath) {
    constexpr float mean[] = {123.675f, 116.28f, 103.53f};
    constexpr float std[] = {58.395f, 57.12f, 57.375f};

    auto image = Image::load(filepath);
    auto resized = resize(image, {1024, 1024});

    std::vector<float> result(3 * 1024 * 1024);
    int pixel_stride = count(resized.channels());
    int row_stride = resized.extent().width * pixel_stride;
    int result_stride = 1024 * 1024;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 1024; ++y) {
            for (int x = 0; x < 1024; ++x) {
                float value = float(resized.pixels()[y * row_stride + x * pixel_stride + c]);
                float normalized = (value - mean[c]) / std[c];
                result[c * result_stride + y * 1024 + x] = normalized;
            }
        }
    }
    return result;
}

//
// Prompt encoder
//

inline float transform_point_coord(int p, int image_size = 1024) {
    float center_normalized = (float(p) + 0.5f) / float(image_size);
    return 2.f * center_normalized - 1.f;
}

inline Tensor position_embedding_random(Model m, Tensor coords) {
    Tensor pe = m.weights("positional_encoding_gaussian_matrix");
    pe = ggml_cont(m, ggml_transpose(m, pe));
    coords = ggml_mul_mat(m, pe, coords);
    coords = ggml_scale_inplace(m, coords, 2.f * pi);
    Tensor coords_sin = ggml_sin(m, coords);
    Tensor coords_cos = ggml_cos(m, coords);
    return ggml_concat(m, coords_sin, coords_cos, 0);
}

inline Tensor embed_points(Model m, Tensor coords) {
    int64_t count = coords->ne[1] - 1; // last element is sentinel
    Tensor x = position_embedding_random(m["pe_layer"], coords);

    // Write "not_a_point_embed" value into the last coordinate
    Tensor label_end = ggml_view_2d(m, x, x->ne[0], 1, x->nb[1], /*offset*/ count * x->nb[1]);
    label_end = ggml_cpy(m, m.weights("not_a_point_embed.weight"), label_end);
    ggml_build_forward_expand(m.graph, label_end);

    // Add point_embeddings[1] weight to all foreground points (prior coordinates)
    Tensor label_one = ggml_view_2d(m, x, x->ne[0], count, x->nb[1], /* offset */ 0);
    label_one = ggml_add_inplace(m, label_one, m.weights("point_embeddings.1.weight"));
    ggml_build_forward_expand(m.graph, label_one);

    // NOTE: background points are not handled
    return x;
}

inline Tensor no_mask_embed(Model m, int embedding_size) {
    Tensor dense = m.weights("no_mask_embed.weight");
    dense = ggml_reshape_4d(m, dense, 1, 1, dense->ne[0], 1);
    dense = ggml_repeat(
        m, dense,
        ggml_new_tensor_4d(m, GGML_TYPE_F32, embedding_size, embedding_size, dense->ne[2], 1));
    return dense;
}

struct EncodePromptResult {
    Tensor sparse;
    Tensor dense;
};

inline EncodePromptResult encode_prompt(Model m, Tensor point_coords) {
    int image_size = 1024;
    int image_embedding_size = image_size / 16;

    EncodePromptResult result;
    result.sparse = embed_points(m, point_coords);
    result.dense = no_mask_embed(m, image_embedding_size);
    return result;
}

} // namespace dlimg