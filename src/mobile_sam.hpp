#pragma once

#include "image.hpp"
#include "ml.hpp"

#include <array>
#include <tuple>
#include <vector>

namespace dlimg::sam {

constexpr int image_size = 1024;
constexpr int mask_size = image_size / 4;

// Common

Tensor linear(ModelRef m, Tensor x);
Tensor conv_2d(ModelRef m, Tensor x, int stride = 1, int pad = 0);
Tensor depthwise_conv_2d(ModelRef m, Tensor x, int stride = 1, int pad = 0);
Tensor layer_norm(ModelRef m, Tensor x, float eps = 1e-5f);
Tensor batch_norm_2d(ModelRef m, Tensor x);

// Image encoder

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

    int img_size = image_size;
    // clang-format off
    std::array<Layer, num_layers> layers = {
        // resolution   dim     depth   attn heads  window size   downsample
        Layer{256,      64,     2,      2,          7,              true},
        Layer{128,      128,    2,      4,          7,              true},
        Layer{64,       160,    6,      5,          14,             true},
        Layer{64,       320,    2,      10,         7,              false}};
    // clang-format on
};

Tensor tiny_vit(ModelRef m, Tensor x, TinyViTParams const& p);

float resize_longest_side(Extent extent, int target_longest_side);
std::vector<float> preprocess_image(ImageView image);
Tensor conv_2d_batch_norm(ModelRef m, Tensor x, int stride = 1, int pad = 0, int groups = 1);
Tensor patch_embed(ModelRef m, Tensor x);
Tensor layer_norm_2d(ModelRef m, Tensor x, float eps = 1e-6f);
Tensor layer_norm_2d_channels(ModelRef m, Tensor x, float eps = 1e-6f);
Tensor mb_conv(ModelRef m, Tensor x);
Tensor patch_merging(ModelRef m, Tensor x, int input_resolution);
Tensor mlp(ModelRef m, Tensor x);
Tensor attention_rel_bias(ModelRef m, Tensor x, int dim, int num_heads);
Tensor window_partition(ModelRef m, Tensor x, int window);
Tensor window_reverse(ModelRef m, Tensor x, int w, int h, int window);
Tensor tiny_vit_block(ModelRef m, Tensor x, int input_resolution, int dim, int num_heads,
                      int window_size);
Tensor conv_layer(ModelRef m, Tensor x, TinyViTParams::Layer p);
Tensor basic_layer(ModelRef m, Tensor x, TinyViTParams::Layer const& p);

// Prompt encoder

std::array<float, 4> preprocess_prompt(Point point, Extent input_image_extent);
std::array<float, 4> preprocess_prompt(Region region, Extent input_image_extent);

Tensor embed_points(ModelRef m, Tensor coords);
Tensor embed_box(ModelRef m, Tensor coords);
Tensor no_mask_embed(ModelRef m, int embedding_size = 64);

float transform_coord(int p, float scale, int image_size = 1024);
Tensor position_embedding_random(ModelRef m, Tensor coords);

// Mask decoder

struct MaskPrediction {
    Tensor masks;
    Tensor iou;
};

MaskPrediction predict_masks(ModelRef m, Tensor image_embeddings, Tensor sparse_prompt,
                             Tensor dense_prompt);

Tensor mlp_block(ModelRef m, Tensor x);
Tensor separate_attention_heads(ModelRef m, Tensor x, int num_heads);
Tensor attention(ModelRef m, Tensor q, Tensor k, Tensor v, int num_heads);
std::tuple<Tensor, Tensor> two_way_attention_block(ModelRef m, Tensor queries, Tensor keys,
                                                   Tensor query_pe, Tensor key_pe, int num_heads,
                                                   bool skip_first_layer_pe = false);
std::tuple<Tensor, Tensor> two_way_transformer(ModelRef m, Tensor image_embedding, Tensor image_pe,
                                               Tensor point_embedding, int depth, int num_heads);
Tensor conv_transpose_2d(ModelRef m, Tensor x, int stride);
Tensor upscale_outputs(ModelRef m, Tensor x);
Tensor hypernetwork_mlp(ModelRef m, Tensor x, int num_layers);
Image postprocess_mask(std::span<float const> mask_data, Extent target_extent);

} // namespace dlimg::sam