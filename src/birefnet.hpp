#pragma once

#include "image.hpp"
#include "ml.hpp"
#include "mobile_sam.hpp"

#include "assert.hpp"
#include <ggml.h>

namespace dlimg::birefnet {

using sam::batch_norm_2d;
using sam::conv_2d;
using sam::layer_norm;
using sam::linear;

inline std::vector<float> preprocess_image(ImageView image, int image_size) {
    constexpr float mean[] = {123.675f, 116.28f, 103.53f}; // 0.485, 0.456, 0.406
    constexpr float std[] = {58.395f, 57.12f, 57.375f};    // 0.229, 0.224, 0.225

    std::optional<Image> resized;
    if (image.extent.width != image_size || image.extent.height != image_size) {
        resized = resize(image, Extent(image_size, image_size));
        image = ImageView(*resized);
    }

    auto input_pixel = PixelAccessor(image);
    std::vector<float> result(3 * image_size * image_size);
    for (int y = 0; y < image_size; ++y) {
        for (int x = 0; x < image_size; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = float(input_pixel.get(image.pixels, x, y, c));
                float normalized = (value - mean[c]) / std[c];
                result[y * image_size * 3 + x * 3 + c] = normalized;
            }
        }
    }
    return result;
}

inline Tensor mlp(ModelRef m, Tensor x) {
    x = linear(m["fc1"], x);
    x = ggml_gelu_inplace(m, x);
    x = linear(m["fc2"], x);
    return m.named(x);
}

inline void compute_relative_position_index(int32_t* dst, int window_size) {
    int n = window_size;
    int n2 = n * n;
    int n4 = n2 * n2;
    for (int i = 0; i < n4; ++i) {
        int x0 = i % n;
        int y0 = (i / n) % n;
        int x1 = (i / n2) % n;
        int y1 = (i / n2 / n) % n;
        dst[i] = (y1 - y0 + n - 1) * (2 * n - 1) + (x1 - x0 + n - 1);
    }
}

inline TensorAlloc<int32_t> create_relative_position_index(ggml_context* ctx, int window_size) {
    int n = window_size;
    auto result = TensorAlloc<int32_t>(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n * n * n * n));
    auto name = TensorName("window_attention_{}.rel_pos_index", n);
    compute_relative_position_index(result.data.get(), n);
    ggml_set_name(result.tensor, name.c_str());
    return result;
}

inline Tensor window_partition(ModelRef m, Tensor x, int window) {
    auto [c, w, h, b] = nelements(x);
    ASSERT(w % window == 0 && h % window == 0 && "Expecting padded input");

    x = ggml_reshape_4d(m, x, c * window, w / window, window, (h / window) * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(m, x, c, window * window, (w / window) * (h / window) * b);
    return x;
}

inline Tensor window_reverse(ModelRef m, Tensor x, int w, int h, int window) {
    int64_t c = x->ne[0];
    int64_t b = x->ne[2] / (w / window) / (h / window);
    ASSERT(x->ne[2] % (w / window) == 0 && "Expecting ne[2] to be multiple of window count");

    x = ggml_reshape_4d(m, x, c * window, window, w / window, (h / window) * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, c, w, h, b);
    return x;
}

inline Tensor window_attention(ModelRef m, Tensor x, Tensor mask, int num_heads, int window) {
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

    Tensor rel_pos_index =
        m.with_prefix(TensorName("window_attention_{}", window)).weights("rel_pos_index");
    Tensor rel_pos_table = m.weights("relative_position_bias_table");
    Tensor rel_pos_bias = ggml_get_rows(m, rel_pos_table, rel_pos_index);
    rel_pos_bias = ggml_reshape_4d(m, rel_pos_bias, num_heads, window * window, window * window, 1);
    rel_pos_bias = ggml_cont(m, ggml_permute(m, rel_pos_bias, 2, 0, 1, 3));
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
    int64_t w = 0;
    int64_t h = 0;
    int shift = 0;
};

inline Tensor swin_block(ModelRef m, Tensor x, Tensor mask, SwinBlockParams const& p) {
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
    x = window_attention(m["attn"], x, mask, num_heads, window);
    x = window_reverse(m, x, w + pad_r, h + pad_b, window);

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

inline Tensor patch_merging(ModelRef m, Tensor x, int w, int h) {
    auto [c, n, b, _] = nelements(x);
    ASSERT(n == w * h && "Spatial dimensions do not match");
    ASSERT(w % 2 == 0 && h % 2 == 0 && "Expecting even spatial dimensions");

    x = ggml_reshape_4d(m, x, c, w, h, b);
    Tensor x0 = ggml_view_4d(m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], 0);
    Tensor x1 = ggml_view_4d(
        m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], x->nb[2]);
    Tensor x2 = ggml_view_4d(
        m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], x->nb[1]);
    Tensor x3 = ggml_view_4d(
        m, x, c, w / 2, h / 2, b, x->nb[1] * 2, x->nb[2] * 2, x->nb[3], x->nb[1] + x->nb[2]);
    x = ggml_concat(m, x0, x1, 0);
    x = ggml_concat(m, x, x2, 0);
    x = ggml_concat(m, x, x3, 0);
    x = ggml_reshape_3d(m, x, c * 4, n / 4, b);

    x = layer_norm(m["norm"], x);
    x = linear(m["reduction"], x);
    return m.named(x);
}

inline void compute_attention_mask(float* out, int64_t w, int64_t h, int window_size) {
    int n = window_size;
    int n2 = n * n;
    int n4 = n2 * n2;
    int shift = window_size / 2;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    int64_t w_pad = nw_x * n;
    int64_t h_pad = nw_y * n;

    memset(out, 0, n4 * nw_x * nw_y * sizeof(float));

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

inline TensorAlloc<float> create_attention_mask(ggml_context* ctx, int64_t w, int64_t h,
                                                int window_size) {
    int n = window_size;
    int64_t nw_x = (w + n - 1) / n;
    int64_t nw_y = (h + n - 1) / n;
    auto result =
        TensorAlloc<float>(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n * n, n * n, nw_x * nw_y));
    auto name = TensorName("swin_layer_{}x{}.attn_mask", w, h);
    compute_attention_mask(result.data.get(), w, h, window_size);
    ggml_set_name(result.tensor, name.c_str());
    return result;
}

struct SwinLayer {
    int depth;
    int num_heads;
    int num_features;
    bool downsample;
};

struct SwinLayerResult {
    Tensor x_out;
    int64_t w_out;
    int64_t h_out;
    Tensor x_down;
    int64_t w_down;
    int64_t h_down;
};

inline SwinLayerResult swin_layer(ModelRef m, Tensor x, int64_t w, int64_t h, SwinLayer const& p,
                                  int window_size) {
    // Attention masks need to be precomputed
    Tensor attn_mask = m.with_prefix(TensorName("swin_layer_{}x{}", w, h)).find("attn_mask");

    ModelRef blocks = m["blocks"];
    for (int i = 0; i < p.depth; ++i) {
        SwinBlockParams b = {.num_heads = p.num_heads,
                             .window_size = window_size,
                             .w = w,
                             .h = h,
                             .shift = i % 2 == 0 ? 0 : window_size / 2};
        x = swin_block(blocks[i], x, attn_mask, b);
    }
    if (p.downsample) {
        Tensor x_down = patch_merging(m["downsample"], x, w, h);
        return {x, w, h, x_down, (w + 1) / 2, (h + 1) / 2};
    }
    return {x, w, h, x, w, h};
}

inline Tensor patch_embed(ModelRef m, Tensor x, int patch_size = 4) {
    ASSERT(x->ne[1] % patch_size == 0 && x->ne[2] % patch_size == 0);

    x = conv_2d(m["proj"], x, patch_size);
    auto [c, ww, wh, b] = nelements(x);
    x = ggml_reshape_3d(m, x, c, ww * wh, b);
    x = layer_norm(m["norm"], x);
    x = ggml_reshape_4d(m, x, c, ww, wh, b);
    return m.named(x);
}

struct SwinParams {
    static constexpr int num_layers = 4;

    int embed_dim;
    int window_size;
    std::array<SwinLayer, num_layers> layers;
};

using SwinResult = std::array<Tensor, SwinParams::num_layers>;

inline SwinResult swin_transformer(ModelRef m, Tensor x, SwinParams const& p) {
    x = patch_embed(m["patch_embed"], x, 4);

    auto [c, w, h, b] = nelements(x);
    x = ggml_reshape_3d(m, x, c, w * h, b);

    SwinLayerResult r{x, w, h, x, w, h};
    SwinResult outs = {};

    for (int i = 0; i < SwinParams::num_layers; ++i) {
        auto layer = m["layers"][i];
        r = swin_layer(layer, r.x_down, r.w_down, r.h_down, p.layers[i], p.window_size);

        TensorName norm_layer("norm{}", i);
        Tensor out = layer_norm(m[norm_layer.c_str()], r.x_out);
        out = ggml_reshape_4d(m, out, p.layers[i].num_features, r.w_out, r.h_out, b);
        outs[i] = out;
    }
    return outs;
}

// clang-format off

constexpr SwinParams swin_t_params = {
    .embed_dim = 96,
    .window_size = 7,
    .layers = {
        //     depth  n_heads   n_features   downsample
        SwinLayer{2,    3,        96 * 1,     true},
        SwinLayer{2,    6,        96 * 2,     true},
        SwinLayer{6,    12,       96 * 4,     true},
        SwinLayer{2,    24,       96 * 8,     false}}};

constexpr SwinParams swin_l_params = {
    .embed_dim = 192,
    .window_size = 12,
    .layers = {
        //     depth  n_heads   n_features   downsample
        SwinLayer{2,    6,        192 * 1,     true},
        SwinLayer{2,    12,       192 * 2,     true},
        SwinLayer{18,   24,       192 * 4,     true},
        SwinLayer{2,    48,       192 * 8,     false}}};

// clang-format on

inline Tensor upscale_to_whcn(ModelRef m, Tensor x, Tensor target) {
    return ggml_upscale_ext(m, x, target->ne[0], target->ne[1], x->ne[2], x->ne[3],
                            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_ALIGN_CORNERS);
}

inline Tensor upscale_to(ModelRef m, Tensor x, Tensor target) {
    x = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3)); // cwhn -> whcn
    x = ggml_upscale_ext(m, x, target->ne[1], target->ne[2], x->ne[2], x->ne[3],
                            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_ALIGN_CORNERS);
    x = ggml_cont(m, ggml_permute(m, x, 1, 2, 0, 3)); // whcn -> cwhn
    return x;
}

inline Tensor downscale_by_whcn(ModelRef m, Tensor x, int f) {
    return ggml_upscale_ext(m, x, x->ne[0] / f, x->ne[1] / f, x->ne[2], x->ne[3],
                            GGML_SCALE_MODE_BILINEAR | GGML_SCALE_ALIGN_CORNERS);
}

inline Tensor downscale_by(ModelRef m, Tensor x, int f) {
    x = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3)); // cwhn -> whcn
    x = downscale_by_whcn(m, x, f);
    x = ggml_cont(m, ggml_permute(m, x, 1, 2, 0, 3)); // whcn -> cwhn
    return x;
}

inline SwinResult encode_concat(ModelRef m, SwinResult& xs, SwinResult& xs_low) {
    // TODO: implement cwhn upscale/interpolate which allows downscale & align_corners=True
    // cwhn -> whcn
    for (int i = 0; i < 4; ++i) {
        xs[i] = ggml_cont(m, ggml_permute(m, xs[i], 2, 0, 1, 3));
        xs_low[i] = ggml_cont(m, ggml_permute(m, xs_low[i], 2, 0, 1, 3));
    }

    xs[0] = ggml_concat(m, xs[0], upscale_to_whcn(m, xs_low[0], xs[0]), 2);
    xs[1] = ggml_concat(m, xs[1], upscale_to_whcn(m, xs_low[1], xs[1]), 2);
    xs[2] = ggml_concat(m, xs[2], upscale_to_whcn(m, xs_low[2], xs[2]), 2);
    xs[3] = ggml_concat(m, xs[3], upscale_to_whcn(m, xs_low[3], xs[3]), 2);

    Tensor x3 = downscale_by_whcn(m, xs[0], 8);
    x3 = ggml_concat(m, x3, downscale_by_whcn(m, xs[1], 4), 2);
    x3 = ggml_concat(m, x3, downscale_by_whcn(m, xs[2], 2), 2);
    xs[3] = ggml_concat(m, x3, xs[3], 2);

    // whcn -> cwhn
    for (int i = 0; i < 4; ++i) {
        xs[i] = ggml_cont(m, ggml_permute(m, xs[i], 1, 2, 0, 3));
    }
    return xs;
}

inline SwinResult encode(ModelRef m, Tensor x, SwinParams const& p) {
    auto [c, w, h, b] = nelements(x);

    auto xs = swin_transformer(m["bb"], x, p);
    auto x_low = downscale_by(m, x, 2);
    auto xs_low = swin_transformer(m["bb"], x_low, p);
    encode_concat(m, xs, xs_low);
    return xs;
}

//
// Decoder
//

inline Tensor conv_2d_deform(ModelRef m, Tensor x, Tensor weight, Tensor offset, Tensor mask,
                             int stride, int pad) {
    x = ggml_permute(m, x, 2, 0, 1, 3);           // cwhn -> whcn
    weight = ggml_permute(m, weight, 2, 0, 1, 3); // cwho -> whco
    offset = ggml_permute(m, offset, 2, 0, 1, 3); // cwhn -> whcn
    if (mask) {
        mask = ggml_permute(m, mask, 2, 0, 1, 3); // cwhn -> whcn
    }
    x = ggml_conv_2d_deform(m, weight, x, offset, mask, stride, stride, pad, pad);
    x = ggml_permute(m, x, 1, 2, 0, 3); // whcn -> cwhn
    return x;
}

inline Tensor deformable_conv_2d(ModelRef m, Tensor x, int stride = 1, int pad = 0) {
    Tensor offset = conv_2d(m["offset"], x, stride, pad);
    Tensor modulator = conv_2d(m["modulator"], x, stride, pad);
    modulator = ggml_sigmoid_inplace(m, modulator);
    modulator = ggml_scale_inplace(m, modulator, 2.0f);

    x = conv_2d_deform(m, x, m.weights("conv.weight"), offset, modulator, stride, pad);
    return m.named(x);
}

inline Tensor mean_2d(ModelRef m, Tensor x) {
    auto [c, w, h, n] = nelements(x);
    x = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3)); // cwhn -> whcn
    x = ggml_mean(m, x);
    x = ggml_reshape_3d(m, x, h, c, n);
    x = ggml_mean(m, x);
    x = ggml_reshape_4d(m, x, c, 1, 1, n);
    return x;
}

inline Tensor global_avg_pool(ModelRef m, Tensor x) {
    x = mean_2d(m[0], x);
    x = conv_2d(m[1], x);
    x = batch_norm_2d(m[2], x);
    x = ggml_relu_inplace(m, x);
    return m.named(x);
}

inline Tensor aspp_module_deformable(ModelRef m, Tensor x, int padding = 0) {
    x = deformable_conv_2d(m["conv"], x, 1, padding);
    x = batch_norm_2d(m["bn"], x);
    x = ggml_relu_inplace(m, x);
    return m.named(x);
}

inline Tensor aspp_deformable(ModelRef m, Tensor x) {
    const int kernel_sizes[] = {1, 3, 7};

    Tensor x1 = aspp_module_deformable(m["aspp1"], x);
    ModelRef aspp_deforms = m["aspp_deforms"];
    Tensor x_deforms[3];
    for (int i = 0; i < 3; ++i) {
        int padding = kernel_sizes[i] / 2;
        x_deforms[i] = aspp_module_deformable(aspp_deforms[i], x, padding);
    }
    Tensor x5 = global_avg_pool(m["global_avg_pool"], x);
    x5 = ggml_reshape_4d(m, x5, 1, 1, x5->ne[0], x5->ne[3]);
    x5 = ggml_upscale_ext(m, x5, x1->ne[1], x1->ne[2], x5->ne[2], x5->ne[3],
                          GGML_SCALE_MODE_BILINEAR | GGML_SCALE_ALIGN_CORNERS);
    x5 = ggml_cont(m, ggml_permute(m, x5, 1, 2, 0, 3)); // whcn -> cwhn
    x = ggml_concat(m, x1, x_deforms[0], 0);
    x = ggml_concat(m, x, x_deforms[1], 0);
    x = ggml_concat(m, x, x_deforms[2], 0);
    x = ggml_concat(m, x, x5, 0);

    x = conv_2d(m["conv1"], x);
    x = batch_norm_2d(m["bn1"], x);
    x = ggml_relu_inplace(m, x);
    return m.named(x);
}

inline Tensor basic_decoder_block(ModelRef m, Tensor x) {
    x = conv_2d(m["conv_in"], x, 1, 1);
    x = batch_norm_2d(m["bn_in"], x);
    x = ggml_relu_inplace(m, x);
    x = aspp_deformable(m["dec_att"], x);
    x = conv_2d(m["conv_out"], x, 1, 1);
    x = batch_norm_2d(m["bn_out"], x);
    return m.named(x);
}

inline Tensor simple_conv(ModelRef m, Tensor x) {
    x = conv_2d(m["conv1"], x, 1, 1);
    x = conv_2d(m["conv_out"], x, 1, 1);
    return m.named(x);
}

inline Tensor image_to_patches(ModelRef m, Tensor x, int out_w, int out_h) {
    auto [w, h, c, b] = nelements(x);
    ASSERT(w % out_w == 0 && h % out_h == 0 && "Grid must divide image size");
    int grid_w = w / out_w;
    int grid_h = h / out_h;
    x = ggml_reshape_4d(m, x, out_w, grid_w, out_h, grid_h * c * b);
    x = ggml_cont(m, ggml_permute(m, x, 0, 2, 1, 3));
    x = ggml_reshape_4d(m, x, out_w, out_h, grid_w * grid_h * c, b);
    return x;
}

inline Tensor gdt_conv(ModelRef m, Tensor x) {
    x = conv_2d(m[0], x, 1, 1);
    x = batch_norm_2d(m[1], x);
    x = ggml_relu_inplace(m, x);
    return x;
}

inline Tensor decode(ModelRef m, Tensor x, SwinResult const& features) {
    Tensor x1 = features[0];
    Tensor x2 = features[1];
    Tensor x3 = features[2];
    Tensor x4 = features[3];
    Tensor x_whcn = ggml_cont(m, ggml_permute(m, x, 2, 0, 1, 3)); // cwhn -> whcn

    {
        Tensor patches = image_to_patches(m, x_whcn, x4->ne[1], x4->ne[2]);
        patches = ggml_cont(m, ggml_permute(m, patches, 1, 2, 0, 3)); // whcn -> cwhn
        patches = simple_conv(m["ipt_blk5"], patches);
        x4 = ggml_concat(m, x4, patches, 0);
    }
    Tensor p4 = basic_decoder_block(m["block4"], x4);
    Tensor p4_gdt = gdt_conv(m["gdt_convs_4"], p4);
    Tensor gdt_attn_4 = conv_2d(m["gdt_convs_attn_4.0"], p4_gdt);
    gdt_attn_4 = ggml_sigmoid(m, gdt_attn_4);
    p4 = ggml_mul(m, p4, gdt_attn_4);

    x3 = conv_2d(m["lateral_block4.conv"], x3);
    Tensor _p4 = upscale_to(m, p4, x3);
    Tensor _p3 = ggml_add_inplace(m, _p4, x3);

    {
        Tensor patches = image_to_patches(m, x_whcn, _p3->ne[1], _p3->ne[2]);
        patches = ggml_cont(m, ggml_permute(m, patches, 1, 2, 0, 3)); // whcn -> cwhn
        patches = simple_conv(m["ipt_blk4"], patches);
        _p3 = ggml_concat(m, _p3, patches, 0);
    }
    Tensor p3 = basic_decoder_block(m["block3"], _p3);
    Tensor p3_gdt = gdt_conv(m["gdt_convs_3"], p3);
    Tensor gdt_attn_3 = conv_2d(m["gdt_convs_attn_3.0"], p3_gdt);
    gdt_attn_3 = ggml_sigmoid(m, gdt_attn_3);
    p3 = ggml_mul(m, p3, gdt_attn_3);

    _p3 = upscale_to(m, p3, x2);
    x2 = conv_2d(m["lateral_block3.conv"], x2);
    Tensor _p2 = ggml_add_inplace(m, _p3, x2);

    {
        Tensor patches = image_to_patches(m, x_whcn, _p2->ne[1], _p2->ne[2]);
        patches = ggml_cont(m, ggml_permute(m, patches, 1, 2, 0, 3)); // whcn -> cwhn
        patches = simple_conv(m["ipt_blk3"], patches);
        _p2 = ggml_concat(m, _p2, patches, 0);
    }
    Tensor p2 = basic_decoder_block(m["block2"], _p2);
    Tensor p2_gdt = gdt_conv(m["gdt_convs_2"], p2);
    Tensor gdt_attn2 = conv_2d(m["gdt_convs_attn_2.0"], p2_gdt);
    gdt_attn2 = ggml_sigmoid(m, gdt_attn2);
    p2 = ggml_mul(m, p2, gdt_attn2);

    _p2 = upscale_to(m, p2, x1);
    x1 = conv_2d(m["lateral_block2.conv"], x1);
    Tensor _p1 = ggml_add_inplace(m, _p2, x1);

    {
        Tensor patches = image_to_patches(m, x_whcn, _p1->ne[1], _p1->ne[2]);
        patches = ggml_cont(m, ggml_permute(m, patches, 1, 2, 0, 3)); // whcn -> cwhn
        patches = simple_conv(m["ipt_blk2"], patches);
        _p1 = ggml_concat(m, _p1, patches, 0);
    }
    _p1 = basic_decoder_block(m["block1"], _p1);
    _p1 = upscale_to(m, _p1, x);
    // ... weird image_to_patches stuff even though _p1 is same size as x
    Tensor p1_ipt = simple_conv(m["ipt_blk1"], x);
    _p1 = ggml_concat(m, _p1, p1_ipt, 0);

    Tensor p1_out = conv_2d(m["conv_out1.0"], _p1);
    return m.named(p1_out);
}

//
// 
//

inline Tensor run(ModelRef m, Tensor image, SwinParams const& encoder_params) {
    // Encoder
    SwinResult features = encode(m, image, encoder_params);
    // Squeeze block
    features[3] = basic_decoder_block(m["squeeze_module.0"], features[3]);
    // Decoder
    Tensor scaled_preds = decode(m["decoder"], image, features);

    return scaled_preds;
}

} // namespace dlimg::birefnet