#pragma once

#include "image.hpp"
#include "ml.hpp"

namespace dlimg::birefnet {
struct SwinParams;

std::vector<float> preprocess_image(ImageView image, int image_size);

Tensor run(ModelRef m, Tensor image, SwinParams const& encoder_params);

// SWIN Transformer

struct SwinBlockParams {
    int num_heads = 6;
    int window_size = 7;
    int64_t w = 0;
    int64_t h = 0;
    int shift = 0;
};

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

struct SwinParams {
    static constexpr int num_layers = 4;

    int embed_dim;
    int window_size;
    std::array<SwinLayer, num_layers> layers;

    static SwinParams detect(ModelRef m);
};

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

using SwinResult = std::array<Tensor, SwinParams::num_layers>;

Tensor mlp(ModelRef m, Tensor x);
void compute_relative_position_index(int32_t* dst, int window_size);
TensorAlloc<int32_t> create_relative_position_index(ggml_context* ctx, int window_size);
Tensor window_partition(ModelRef m, Tensor x, int window);
Tensor window_reverse(ModelRef m, Tensor x, int w, int h, int window);
Tensor window_attention(ModelRef m, Tensor x, Tensor mask, int num_heads, int window);
Tensor swin_block(ModelRef m, Tensor x, Tensor mask, SwinBlockParams const& p);
Tensor patch_merging(ModelRef m, Tensor x, int w, int h);
void compute_attention_mask(float* out, int64_t w, int64_t h, int window_size);
TensorAlloc<float> create_attention_mask(ggml_context* ctx, int64_t w, int64_t h, int window_size);
SwinLayerResult swin_layer(ModelRef m, Tensor x, int64_t w, int64_t h, SwinLayer const& p,
                           int window_size);
Tensor patch_embed(ModelRef m, Tensor x, int patch_size = 4);
SwinResult swin_transformer(ModelRef m, Tensor x, SwinParams const& p);

// Encoder

SwinResult encode_concat(ModelRef m, SwinResult& xs, SwinResult& xs_low);
SwinResult encode(ModelRef m, Tensor x, SwinParams const& p);

// Decoder

Tensor conv_2d_deform(ModelRef m, Tensor x, Tensor weight, Tensor offset, Tensor mask, int stride,
                      int pad);
Tensor deformable_conv_2d(ModelRef m, Tensor x, int stride = 1, int pad = 0);
Tensor mean_2d(ModelRef m, Tensor x);
Tensor global_avg_pool(ModelRef m, Tensor x);
Tensor aspp_module_deformable(ModelRef m, Tensor x, int padding = 0);
Tensor aspp_deformable(ModelRef m, Tensor x);
Tensor basic_decoder_block(ModelRef m, Tensor x);
Tensor simple_conv(ModelRef m, Tensor x);
Tensor image_to_patches(ModelRef m, Tensor x, int out_w, int out_h);
Tensor gdt_conv(ModelRef m, Tensor x);
Tensor decode(ModelRef m, Tensor x, SwinResult const& features);

} // namespace dlimg::birefnet