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

inline Tensor window_partition(Model m, Tensor x, int window) {
    auto [c, w, h, b] = nelements(x);
    ASSERT(w % window == 0 && h % window == 0 && "Expecting padded input");

    x = ggml_reshape_4d(m, x, c * window, w / window, window, (h / window) * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, window * window, (w / window) * (h / window) * b);
    return x;
}

inline Tensor window_reverse(Model m, Tensor x, int w, int h, int window) {
    int64_t c = x->ne[0];
    int64_t b = x->ne[2] / (w / window) / (h / window);
    ASSERT(x->ne[2] % (w / window) == 0 && "Expecting ne[2] to be multiple of window count");

    x = ggml_reshape_4d(m, x, c * window, window, w / window, (h / window) * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, c, w, h, b);
    return x;
}

inline Tensor window_attention(Model m, Tensor x, Tensor mask, int num_heads) {
    auto rel_pos_bias = m.with_prefix(TensorName("window_attention_{}", num_heads))
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
    if (mask) {
        int64_t nw = mask->ne[2];
        attn = ggml_reshape_4d(m, attn, n * n, num_heads, nw, b / nw);
        mask = ggml_reshape_4d(m, mask, n * n, 1, nw, 1);
        attn = ggml_add_inplace(m, attn, mask);
        attn = ggml_reshape_4d(m, attn, n, n, num_heads, b);
    }
    attn = ggml_soft_max(m, attn);

    x = ggml_mul_mat(m, v, attn);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, n, b);

    x = linear(m["proj"], x);
    return m.named(x);
}

struct SwinBlockParams {
    int num_heads = 6;
    int window_size = 7;
    int w = 0;
    int h = 0;
    int shift = 0;
};

inline Tensor swin_block(Model m, Tensor x, Tensor mask, SwinBlockParams const& p) {
    auto [c, n, b, _] = nelements(x);
    auto [num_heads, window, w, h, shift] = p;
    ASSERT(n == w * h && "Spatial dimensions do not match");

    Tensor shortcut = x;
    x = layer_norm(m["norm1"], x);
    x = ggml_reshape_4d(m, x, c, w, h, b);

    int pad_r = (window - w % window) % window;
    int pad_b = (window - h % window) % window;
    if (pad_r > 0 || pad_b > 0) {
        x = ggml_pad(m, x, 0, pad_r, pad_b, 0);
    }

    ASSERT(shift == 0 || mask != nullptr);
    if (shift > 0) {
        x = ggml_roll(m, x, 0, -shift, -shift, 0);
    }

    x = window_partition(m, x, window);
    x = window_attention(m["attn"], x, mask, num_heads);
    x = window_reverse(m, x, w, h, window);

    if (shift > 0) { // undo shift
        x = ggml_roll(m, x, 0, shift, shift, 0);
    }

    if (pad_r > 0 || pad_b > 0) { // undo padding
        size_t nb1 = x->nb[0] * c;
        x = ggml_view_4d(
            m, x, c, w, h, b, nb1, nb1 * (w + pad_r), nb1 * (w + pad_r) * (h + pad_b), 0);
        x = ggml_cont(m, x);
    }

    x = ggml_reshape_3d(m, x, c, n, b);
    x = ggml_add_inplace(m, x, shortcut);

    Tensor x_mlp = layer_norm(m["norm2"], x);
    x_mlp = mlp(m["mlp"], x_mlp);
    x = ggml_add_inplace(m, x, x_mlp);

    return m.named(x);
}

inline Tensor patch_merging(Model m, Tensor x, int w, int h) {
    auto [c, n, b, _] = nelements(x);
    ASSERT(n == w * h && "Spatial dimensions do not match");
    ASSERT(w % 2 == 0 && h % 2 == 0 && "Expecting even spatial dimensions");

    x = ggml_reshape_4d(m, x, c, w, h, b);
    Tensor x0 = ggml_view_4d(m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], 0);
    Tensor x1 = ggml_view_4d(m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], x->nb[2]);
    Tensor x2 = ggml_view_4d(m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], x->nb[1]);
    Tensor x3 = ggml_view_4d(m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], x->nb[1] + x->nb[2]);
    x = ggml_concat(m, x0, x1, 0);
    x = ggml_concat(m, x, x2, 0);
    x = ggml_concat(m, x, x3, 0);
    x = ggml_reshape_3d(m, x, c * 4, n / 4, b);

    x = layer_norm(m["norm"], x);
    x = linear(m["reduction"], x);
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

inline size_t attention_mask_size(int64_t w, int64_t h, int window_size) {
    int n = window_size;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    return sizeof(float) * n * n * n * n * nw_x * nw_y;
}

inline void compute_attention_mask(float * out, int64_t w, int64_t h, int window_size) {
    int n = window_size;
    int n2 = n * n;
    int n4 = n2 * n2;
    int shift = window_size / 2;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    int64_t w_pad = nw_x * n;
    int64_t h_pad = nw_y * n;

    // Tensor mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n2, n2, nw_x * nw_y);
    // float * out = (float *) mask->data;
    memset(out, 0, attention_mask_size(w, h, window_size));

    for (int iw_y = 0; iw_y < nw_y; ++iw_y) {
        for (int iw_x = 0; iw_x < nw_x; ++iw_x) {
            // Skip all windows that aren't at the right or bottom edges of the image
            if (iw_y < nw_y - 1 && iw_x < nw_x - 1) {
                continue;
            }
            int64_t base = iw_y * nw_x * n4 + iw_x * n4;

            for (int y0 = 0; y0 < n; ++y0) {
                for (int x0 = 0; x0 < n; ++x0) {
                    for (int y1 = 0; y1 < n; ++y1) {
                        for (int x1 = 0; x1 < n; ++x1) {
                            // Window-local coordinates to global image coordinates
                            int yy0 = iw_y * n + y0;
                            int xx0 = iw_x * n + x0;
                            int yy1 = iw_y * n + y1;
                            int xx1 = iw_x * n + x1;
                            // Check if two patches being matched belong to the same window
                            // that is: they are both in the shift zone, or both outside
                            bool match_y = (yy0 < h_pad - shift) == (yy1 < h_pad - shift);
                            bool match_x = (xx0 < w_pad - shift) == (xx1 < w_pad - shift);
                            // If not, set mask to -100 (added to attention before softmax)
                            if (!match_y || !match_x) {
                                int64_t idx = base + (y0 * n + x0) * n2 + (y1 * n + x1);
                                out[idx] = -100.f;
                            }                                
                        }
                    }
                }
            }
        }
    }
}

} // namespace dlimg::birefnet