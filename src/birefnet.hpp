#pragma once

#include "image.hpp"
#include "ml.hpp"
#include "mobile_sam.hpp"

#include "assert.hpp"
#include <ggml.h>

namespace dlimg::birefnet {

using sam::conv_2d;
using sam::layer_norm;
using sam::linear;

inline Tensor mlp(Model m, Tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu_inplace(m, x);
    x = linear(m["fc2"], x);
    return m.named(x);
}

inline Tensor window_attention(Model m, Tensor x, int num_heads) {
    auto rel_pos_bias = m
        .with_prefix(TensorName("window_attention_{}", num_heads))
        .weights("relative_position_bias");
    auto [c, n, b, _] = nelements(x);

    Tensor qkv = linear(m["qkv"], x);
    qkv = ggml_reshape_4d(m, qkv, c / num_heads, num_heads, 3, n * b);
    qkv = ggml_cont(m, ggml_permute(m, qkv, 0, 1, 3, 2));

    size_t offset = qkv->nb[3];
    auto split = [=](Tensor tensor, size_t index, bool transpose = false) mutable {
        tensor = ggml_view_3d(m, tensor, c / num_heads, num_heads, n * b, tensor->nb[1],
                              tensor->nb[2], offset * index);
        tensor = ggml_reshape_4d(m, tensor, c / num_heads, num_heads, n, b);
        if (transpose) {
            tensor = ggml_cont(m, ggml_permute(m, tensor, 1, 2, 0, 3));
        } else {
            tensor = ggml_cont(m, ggml_permute(m, tensor, 0, 2, 1, 3));
        }
        return tensor;
    };
    Tensor q = split(qkv, 0);
    Tensor k = split(qkv, 1);
    Tensor v = split(qkv, 2, true);

    q = ggml_scale_inplace(m, q, 1.0f / std::sqrtf(float(c / num_heads)));

    Tensor attn = ggml_mul_mat(m, k, q);
    attn = ggml_add_inplace(m, attn, rel_pos_bias);
    attn = ggml_soft_max(m, attn);

    x = ggml_mul_mat(m, v, attn);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, n, b);
    
    x = linear(m["proj"], x);
    return m.named(x);    
}

inline Tensor patch_embed(Model m, Tensor x, int patch_size = 4) {
    ASSERT(x->ne[1] % patch_size == 0 && x->ne[2] % patch_size == 0);

    x = conv_2d(m["proj"], x, patch_size);
    auto [c, ww, wh, b] = nelements(x);
    x = ggml_reshape_3d(m, x, c, ww * wh, b);
    x = layer_norm(m["norm"], x);
    x = ggml_reshape_4d(m, x, c, ww, wh, b);
    return m.named(x);
}

struct SwinLayer {
    int depth;
    int num_heads;
    int num_features;
};

struct SwinLayerResult {
    Tensor x;
    Tensor x_out;
    int64_t ww;
    int64_t wh;
    int64_t w;
    int64_t h;
};

inline SwinLayerResult swin_layer(Model m, Tensor x, int64_t ww, int64_t wh, SwinLayer const& p) {
    return {};
}

struct SwinParams {
    static const int num_layers = 4;

    int embed_dim = 96;
    int window_size = 7;
    std::array<SwinLayer, num_layers> layers = {SwinLayer{2, 6, 96 * 1}, SwinLayer{2, 12, 96 * 2},
                                                SwinLayer{18, 24, 96 * 4},
                                                SwinLayer{2, 48, 96 * 8}};
};

using SwinResult = std::array<Tensor, SwinParams::num_layers>;

inline SwinResult swin_transformer(Model m, Tensor x, SwinParams const& p) {
    x = patch_embed(m["patch_embed"], x, 4);

    auto [c, ww, wh, b] = nelements(x);
    x = ggml_reshape_3d(
        m, x, c, ww * wh, b); // TODO: this just reverts the reshape at the end of patch_embed...

    SwinLayerResult r{x, nullptr, ww, wh, 0, 0};
    std::array<Tensor, SwinParams::num_layers> outs = {};

    for (int i = 0; i < SwinParams::num_layers; ++i) {
        auto layer = m["layers"][i];
        r = swin_layer(layer, r.x, r.ww, r.wh, p.layers[i]);

        Tensor out = layer_norm(layer["norm"], r.x_out);
        out = ggml_reshape_4d(m, out, p.embed_dim * (2 << i), r.w, r.h, b);
        outs[i] = out;
    }
    return outs;
}

} // namespace dlimg::birefnet