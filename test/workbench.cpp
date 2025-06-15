#include "birefnet.hpp"
#include "migan.hpp"
#include "mobile_sam.hpp"

#include <fmt/format.h>
#include <ggml-blas.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <ggml.h>

#include <cassert>
#include <numeric>
#include <string_view>
#include <vector>

#ifdef _MSC_VER
#    define API __declspec(dllexport)
#else
#    define API
#endif

namespace dlimg {

Tensor conv_2d_nchw(ModelRef m, Tensor x, int stride = 1, int pad = 0) {
    x = ggml_conv_2d(m, m.weights("weight"), x, stride, stride, pad, pad, 1, 1);
    if (auto bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

Tensor depthwise_conv_2d_nchw(ModelRef m, Tensor x, int stride = 1, int pad = 0) {
    return ggml_conv_2d_dw_direct(m, m.weights("weight"), x, stride, stride, pad, pad, 1, 1);
}

struct RawTensor {
    char const* name;
    float* data;
    int32_t type_;
    int32_t ne[4];

    ggml_type type() const { return ggml_type(type_); }
    size_t size() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    size_t size_bytes() const { return size() * ggml_type_size(type()); }
};

struct Workbench {

    Workbench(int input_count, RawTensor* inputs_raw, RawTensor const& output_raw,
              GGMLBackend backend) {

        auto context_params = ggml_init_params{};
        context_params.mem_size = ggml_tensor_overhead() * (input_count + 1) +
                                  ggml_graph_overhead() + 2048 * ggml_tensor_overhead();
        context_params.no_alloc = true;
        m.weights_context = m.graph_context = ggml_init(context_params);
        m.graph = ggml_new_graph(m);
        m.backend = backend;
        backends[1] = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backends[1], 1);

        if (backend == GGMLBackend::vulkan) {
            backends[0] = ggml_backend_vk_init(0);
        } else {
            backends[0] = backends[1];
        }

        for (int i = 0; i < input_count; ++i) {
            auto& raw = inputs_raw[i];
            auto tensor = ggml_new_tensor_4d(
                m, GGML_TYPE_F32, raw.ne[0], raw.ne[1], raw.ne[2], raw.ne[3]);
            ggml_set_name(tensor, raw.name);
        }
        auto output = ggml_new_tensor_4d(m, GGML_TYPE_F32, output_raw.ne[0], output_raw.ne[1],
                                         output_raw.ne[2], output_raw.ne[3]);
        ggml_set_name(output, output_raw.name);

        ggml_backend_alloc_ctx_tensors(m, backends[0]);
        for (auto&& raw : std::span(inputs_raw, input_count)) {
            auto tensor = ggml_get_tensor(m, raw.name);
            ggml_backend_tensor_set(tensor, raw.data, 0, raw.size_bytes());
        }
    }

    void output(Tensor tensor, RawTensor dest) {
        GGML_ASSERT(dest.size_bytes() == ggml_nbytes(tensor));

        Tensor out = ggml_cont(m, tensor);
        ggml_build_forward_expand(m.graph, out);
        outputs.emplace_back(out, dest);
    }

    void run() {
        ggml_backend_alloc_ctx_tensors(m, backends[0]);
        ggml_backend_buffer_type_t buffer_types[] = {
            ggml_backend_get_default_buffer_type(backends[0]),
            ggml_backend_get_default_buffer_type(backends[1]),
        };
        int backend_count = m.backend == GGMLBackend::cpu ? 1 : 2;
        auto sched = ggml_backend_sched_new(
            backends.data(), buffer_types, backend_count, ggml_graph_size(m.graph), false, false);

        ggml_backend_sched_graph_compute(sched, m.graph);

        for (auto& [output, output_raw] : outputs) {
            ggml_backend_tensor_get(output, output_raw.data, 0, ggml_nbytes(output));
        }
    }

    ModelRef m;
    std::vector<std::tuple<Tensor, RawTensor>> outputs;
    std::array<ggml_backend_t, 2> backends;
};

} // namespace dlimg

#ifdef __cplusplus
extern "C" {
#endif

API int32_t dlimg_workbench(char const* testcase, int input_count, dlimg::RawTensor* inputs,
                            dlimg::RawTensor const& output, int32_t backend) {
    using namespace dlimg;
    using namespace dlimg::sam;

    try {
        auto name = std::string_view(testcase);
        auto w = dlimg::Workbench(input_count, inputs, output, GGMLBackend(backend));
        ModelRef m = w.m;
        Tensor input = m.weights("input");

        if (name == "conv_2d_depthwise_nchw_stride_1_pad_0") {
            w.output(depthwise_conv_2d_nchw(m, input), output);
        } else if (name == "conv_2d_depthwise_nchw_stride_2_pad_1") {
            w.output(depthwise_conv_2d_nchw(m, input, 2, 1), output);
        } else if (name == "conv_2d_depthwise_nhwc_stride_1_pad_0") {
            w.output(depthwise_conv_2d(m, input), output);
        } else if (name == "conv_2d_depthwise_nhwc_stride_2_pad_1") {
            w.output(depthwise_conv_2d(m, input, 2, 1), output);
        } else if (name == "conv_2d_depthwise_nchw_dilation_2_pad_2") {
            w.output(
                ggml_conv_2d_dw_direct(m, m.weights("weight"), input, 1, 1, 2, 2, 2, 2), output);
        } else if (name == "conv_2d_depthwise_nhwc_dilation_2_pad_2") {
            Tensor weight = ggml_permute(m, m.weights("weight"), 3, 2, 0, 1);
            input = ggml_permute(m, input, 2, 0, 1, 3);
            Tensor result = ggml_conv_2d_dw_direct(m, weight, input, 1, 1, 2, 2, 2, 2);
            result = ggml_permute(m, result, 1, 2, 0, 3);
            w.output(result, output);
        } else if (name == "conv_2d") {
            w.output(conv_2d_nchw(m, input), output);
        } else if (name.starts_with("conv_2d_channels_stride2_pad1")) {
            w.output(conv_2d(m, input, 2, 1), output);
        } else if (name.starts_with("conv_2d_channels")) {
            w.output(conv_2d(m, input), output);
        } else if (name.starts_with("conv_transpose_2d_stride2")) {
            w.output(conv_transpose_2d(m, input, 2), output);
        } else if (name.starts_with("conv_transpose_2d_nchw")) {
            w.output(ggml_conv_transpose_2d_p0(m, m.weights("weight"), input, 1), output);
        } else if (name.starts_with("conv_transpose_2d")) {
            w.output(conv_transpose_2d(m, input, 1), output);
        } else if (name == "conv_2d_deform") {
            Tensor weight = m.weights("weight");
            Tensor offset = m.weights("offset");
            Tensor mask = m.find("mask");
            w.output(birefnet::conv_2d_deform(m, input, weight, offset, mask, 1, 1), output);
        } else if (name == "batch_norm_2d") {
            w.output(batch_norm_2d(m, input), output);
        } else if (name == "roll_(0, 2, -1, 0)") {
            w.output(ggml_roll(m, input, 0, -1, 2, 0), output);
        } else if (name == "roll_(0, -2, 0, 3)") {
            w.output(ggml_roll(m, input, 3, 0, -2, 0), output);
        } else if (name == "layer_norm") {
            w.output(layer_norm(m, input), output);
        } else if (name == "upscale_align_corners") {
            int mode = GGML_SCALE_MODE_BILINEAR | GGML_SCALE_ALIGN_CORNERS;
            w.output(ggml_upscale_ext(m, input, output.ne[0], output.ne[1], output.ne[2], 1, mode),
                     output);
        } else if (name == "downscale_align_corners") {
            int mode = GGML_SCALE_MODE_BILINEAR | GGML_SCALE_ALIGN_CORNERS;
            w.output(ggml_upscale_ext(m, input, output.ne[0], output.ne[1], output.ne[2], 1, mode),
                     output);
        } else if (name == "upscale_bilinear") {
            int mode = GGML_SCALE_MODE_BILINEAR;
            w.output(ggml_upscale_ext(m, input, output.ne[0], output.ne[1], output.ne[2], 1, mode),
                     output);
        } else if (name == "linear") {
            w.output(linear(m, input), output);
        } else if (name == "conv_2d_batch_norm") {
            w.output(conv_2d_batch_norm(m, input, 2, 1), output);
        } else if (name == "layer_norm_2d") {
            w.output(layer_norm_2d(m, input), output);
        } else if (name == "patch_embed") {
            w.output(patch_embed(m, input), output);
        } else if (name == "mb_conv") {
            w.output(mb_conv(m, input), output);
        } else if (name == "patch_merging") {
            w.output(patch_merging(m, input, 32), output);
        } else if (name == "mlp") {
            w.output(mlp(m, input), output);
        } else if (name == "attention_rel_bias") {
            w.output(attention_rel_bias(m, input, 4, 2), output);
        } else if (name == "window_partition") {
            w.output(window_partition(m, input, 3), output);
        } else if (name == "tiny_vit_block") {
            w.output(tiny_vit_block(m, input, 8, /*dim*/ 4, /*num_heads*/ 2,
                                    /*window_size*/ 5),
                     output);
        } else if (name == "tiny_vit") {
            TinyViTParams p;
            w.output(tiny_vit(m, input, p), output);
        } else if (name == "position_embedding_random") {
            float* input_data = reinterpret_cast<float*>(input->data);
            for (int i = 0; i < ggml_nelements(input); ++i) {
                input_data[i] = (input_data[i] / 64.f) * 2.f - 1.f;
            }
            w.output(position_embedding_random(m, input), output);
        } else if (name == "embed_points") {
            float* input_data = reinterpret_cast<float*>(input->data);
            for (int i = 0; i < ggml_nelements(input) - 2; ++i) {
                input_data[i] = transform_coord(input_data[i], 1.0f, 64);
            }
            w.output(embed_points(m, input), output);
        } else if (name == "embed_box") {
            float* input_data = reinterpret_cast<float*>(input->data);
            for (int i = 0; i < ggml_nelements(input); ++i) {
                input_data[i] = transform_coord(input_data[i], 1.0f, 64);
            }
            w.output(embed_box(m, input), output);
        } else if (name == "attention") {
            Tensor q = input;
            Tensor k = m.weights("input_k");
            Tensor v = m.weights("input_v");
            w.output(attention(m, q, k, v, 2), output);
        } else if (name.starts_with("two_way_attention_block")) {
            Tensor queries = input;
            Tensor keys = m.weights("input_keys");
            Tensor query_pe = m.weights("input_query_pe");
            Tensor key_pe = m.weights("input_key_pe");
            bool skip_first_layer_pe = name.ends_with("skip_first_layer_pe");
            auto [result_queries, result_keys] = two_way_attention_block(
                m, queries, keys, query_pe, key_pe, 2, skip_first_layer_pe);
            w.output(result_queries, output);
            w.output(result_keys, inputs[input_count - 1]);
        } else if (name == "two_way_transformer") {
            Tensor image_embedding = input;
            Tensor image_pe = m.weights("input_image_pe");
            Tensor point_embedding = m.weights("input_point_embedding");
            auto [result_queries, result_keys] = two_way_transformer(
                m, image_embedding, image_pe, point_embedding, 2, 2);
            w.output(result_queries, output);
            w.output(result_keys, inputs[input_count - 1]);
        } else if (name == "hypernetwork_mlp") {
            w.output(hypernetwork_mlp(m, input, 2), output);
        } else if (name == "output_upscaling") {
            w.output(upscale_outputs(m, input), output);
        } else if (name == "predict_masks") {
            Tensor image_embeddings = input;
            Tensor sparse_prompt = m.weights("input_sparse_prompt");
            Tensor dense_prompt = m.weights("input_dense_prompt");
            auto [masks, iou] = predict_masks(m, image_embeddings, sparse_prompt, dense_prompt);
            w.output(masks, output);
            w.output(iou, inputs[input_count - 1]);
        } else if (name == "biref_patch_embed") {
            w.output(birefnet::patch_embed(m, input), output);
        } else if (name == "biref_relative_position_index") {
            birefnet::compute_relative_position_index(reinterpret_cast<int32_t*>(output.data), 3);
            return 0;
        } else if (name == "biref_window_attention") {
            int window_size = 3;
            Tensor mask = m.find("mask");
            auto rel_pos_index = birefnet::create_relative_position_index(m, window_size);
            ggml_backend_alloc_ctx_tensors(m, w.backends[0]);
            rel_pos_index.copy_to_backend_buffer();
            w.output(birefnet::window_attention(m, input, mask, 2, window_size), output);
        } else if (name == "biref_swin_block") {
            birefnet::SwinBlockParams p;
            p.num_heads = 2;
            p.window_size = 3;
            p.w = 6;
            p.h = 6;
            p.shift = 0;
            Tensor mask = m.find("mask");
            auto rel_pos_index = birefnet::create_relative_position_index(m, 3);
            ggml_backend_alloc_ctx_tensors(m, w.backends[0]);
            rel_pos_index.copy_to_backend_buffer();
            w.output(birefnet::swin_block(m, input, mask, p), output);
        } else if (name == "biref_patch_merging") {
            w.output(birefnet::patch_merging(m, input, 6, 4), output);
        } else if (name == "biref_attention_mask") {
            birefnet::compute_attention_mask(output.data, 18, 18, 6);
            return 0;
        } else if (name == "biref_swin_layer") {
            birefnet::SwinLayer p;
            p.depth = 2;
            p.num_heads = 2;
            p.num_features = 8;
            p.downsample = true;
            auto rel_pos_index = birefnet::create_relative_position_index(m, 3);
            auto result = birefnet::swin_layer(m, input, 6, 6, p, 3);
            ASSERT(result.w_down == 3 && result.h_down == 3);
            w.output(result.x_down, output);
        } else if (name == "biref_swin_transformer") {
            birefnet::SwinParams p = {.embed_dim = 8,
                                      .window_size = 3,
                                      .layers = {
                                          birefnet::SwinLayer{2, 2, 8 * 1, true},
                                          birefnet::SwinLayer{2, 2, 8 * 2, true},
                                          birefnet::SwinLayer{2, 4, 8 * 4, true},
                                          birefnet::SwinLayer{2, 2, 8 * 8, false},
                                      }};
            auto rel_pos_index = birefnet::create_relative_position_index(m, 3);
            auto attn_masks = std::array{birefnet::create_attention_mask(m, 8, 8, 3),
                                         birefnet::create_attention_mask(m, 4, 4, 3),
                                         birefnet::create_attention_mask(m, 2, 2, 3),
                                         birefnet::create_attention_mask(m, 1, 1, 3)};
            ggml_backend_alloc_ctx_tensors(m, w.backends[0]);
            rel_pos_index.copy_to_backend_buffer();
            for (auto&& attn_mask : attn_masks) {
                attn_mask.copy_to_backend_buffer();
            }
            auto result = birefnet::swin_transformer(m, input, p);
            w.output(result[0], output);
            w.output(result[1], inputs[input_count - 3]);
            w.output(result[2], inputs[input_count - 2]);
            w.output(result[3], inputs[input_count - 1]);
        } else if (name == "biref_encode") {
            birefnet::SwinResult xs, xs_low;
            for (int i = 0; i < 4; ++i) {
                xs[i] = m.find(TensorName("input{}", i).c_str());
                xs_low[i] = m.find(TensorName("input_low{}", i).c_str());
            }
            birefnet::encode_concat(m, xs, xs_low);
            for (int i = 0; i < 4; ++i) {
                w.output(xs[i], inputs[input_count - 4 + i]);
            }
        } else if (name == "biref_deformable_conv_2d") {
            w.output(birefnet::deformable_conv_2d(m, input, 1, 1), output);
        } else if (name == "biref_global_avg_pool") {
            w.output(birefnet::global_avg_pool(m, input), output);
        } else if (name == "biref_aspp_deformable") {
            w.output(birefnet::aspp_deformable(m, input), output);
        } else if (name == "biref_basic_dec_blk") {
            w.output(birefnet::basic_decoder_block(m, input), output);
        } else if (name == "biref_image_to_patches_2") {
            w.output(birefnet::image_to_patches(m, input, 4, 4), output);
        } else if (name == "biref_decode") {
            birefnet::SwinResult features;
            for (int i = 0; i < 4; ++i) {
                features[i] = m.find(TensorName("x{}", i + 1).c_str());
            }
            w.output(birefnet::decode(m, input, features), output);
        } else if (name == "blur") {
            auto img = std::span(reinterpret_cast<float4*>(input->data), ggml_nelements(input) / 4);
            auto out = std::span(reinterpret_cast<float4*>(output.data), output.size() / 4);
            blur(img, out, Extent{1024, 1024}, 30);
            return 0;
        } else if (name == "estimate_foreground") {
            auto mask = m.find("mask");
            auto img = std::span(reinterpret_cast<float4*>(input->data), ggml_nelements(input) / 4);
            auto alpha = std::span(reinterpret_cast<float*>(mask->data), ggml_nelements(mask));
            auto result = estimate_foreground(img, alpha, Extent{256, 256}, 30);
            ASSERT(result.size() == output.size() / 4);
            memcpy(output.data, result.data(), output.size_bytes());
            return 0;
        } else if (name == "migan_lrelu_agc") {
            w.output(migan::lrelu_agc(m, input, 0.2f, std::sqrtf(2), 1.0f), output);
        } else if (name == "migan_downsample_2d") {
            w.output(migan::downsample_2d(m, input), output);
        } else if (name == "migan_upsample_2d") {
            w.output(ggml_cont(m, migan::upsample_2d(m, input)), output);
        } else if (name == "migan_separable_conv_2d") {
            auto flags = migan::conv::noise | migan::conv::activation;
            w.output(migan::separable_conv_2d(m, input, flags), output);
        } else {
            throw std::runtime_error("Unknown testcase: " + std::string(testcase));
        }

        w.run();

    } catch (std::exception const& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif