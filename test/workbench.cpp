#include "mobile_sam.hpp"
#include "primitives.hpp"

#include <fmt/format.h>
#include <ggml-blas.h>
#include <ggml-cpu.h>
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

struct RawTensor {
    char const* name;
    float* data;
    int32_t w;
    int32_t h;
    int32_t c;
    int32_t n;

    size_t size() const { return n * c * w * h; }
    size_t size_bytes() const { return size() * sizeof(float); }
};

struct Workbench {

    Workbench(int input_count, RawTensor* inputs_raw, RawTensor const& output_raw) {
        auto context_params = ggml_init_params{};
        context_params.mem_size = 48 * output_raw.size_bytes() +
                                  ggml_tensor_overhead() * (input_count + 1) +
                                  ggml_graph_overhead() + 2048 * ggml_tensor_overhead();
        context_params.no_alloc = true;
        model.model_context = model.graph_context = ggml_init(context_params);
        model.graph = ggml_new_graph(model);
        backends[0] = ggml_backend_blas_init();
        backends[1] = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backends[1], 1);


        for (int i = 0; i < input_count; ++i) {
            auto& raw = inputs_raw[i];
            auto tensor = ggml_new_tensor_4d(model,GGML_TYPE_F32 ,raw.w,raw.h,raw.c,raw.n);
            ggml_set_name(tensor, raw.name);
        }
        ggml_backend_alloc_ctx_tensors(model, backends[0]);
        for (auto&& raw : std::span(inputs_raw, input_count)) {
            auto tensor = ggml_get_tensor(model, raw.name);
            ggml_backend_tensor_set(tensor, raw.data, 0, raw.size_bytes());
        }
    }

    void output(Tensor tensor, RawTensor dest) {
        GGML_ASSERT(dest.size_bytes() == ggml_nbytes(tensor));

        Tensor out = ggml_cont(model, tensor);
        ggml_build_forward_expand(model.graph, out);
        outputs.emplace_back(out, dest);
    }

    void run() {
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backends[0]));
        ggml_backend_buffer_type_t buffer_types[] = {
            ggml_backend_get_default_buffer_type(backends[0]),
            ggml_backend_get_default_buffer_type(backends[1]),
        };
        auto sched = ggml_backend_sched_new(
            backends.data(), buffer_types, 2, ggml_graph_size(model.graph), false);

        ggml_backend_sched_graph_compute(sched, model.graph);

        for (auto& [output, output_raw] : outputs) {
            memcpy(output_raw.data, ggml_get_data_f32(output), ggml_nbytes(output));
        }
    }

    Model model;
    std::vector<std::tuple<Tensor, RawTensor>> outputs;
    std::array<ggml_backend_t, 2> backends;
};

} // namespace dlimg

#ifdef __cplusplus
extern "C" {
#endif

API int32_t dlimg_workbench(char const* testcase, int input_count, dlimg::RawTensor* inputs,
                            dlimg::RawTensor const& output) {
    using namespace dlimg;
    using namespace dlimg::sam;

    try {
        auto name = std::string_view(testcase);
        auto w = dlimg::Workbench(input_count, inputs, output);
        Tensor input = w.model.weights("input");

        if (name == "conv_2d_depthwise_nchw_stride_1_pad_0") {
            w.output(conv_2d_depth_wise(w.model, input), output);
        } else if (name == "conv_2d_depthwise_nchw_stride_2_pad_1") {
            w.output(conv_2d_depth_wise(w.model, input, 2, 1), output);
        } else if (name == "conv_2d_depthwise_nhwc_stride_1_pad_0") {
            w.output(conv_2d_depth_wise_channels(w.model, input), output);
        } else if (name == "conv_2d_depthwise_nhwc_stride_2_pad_1") {
            w.output(conv_2d_depth_wise_channels(w.model, input, 2, 1), output);
        } else if (name == "conv_2d") {
            w.output(conv_2d(w.model, input), output);
        } else if (name == "conv_2d_channels") {            
            w.output(conv_2d_channels(w.model, input), output);
        } else if (name == "batch_norm_2d") {
            w.output(batch_norm_2d(w.model, input), output);
        } else if (name == "layer_norm") {
            w.output(layer_norm(w.model, input), output);
        } else if (name == "linear") {
            w.output(linear(w.model, input), output);
        } else if (name == "conv_2d_batch_norm") {
            w.output(conv_2d_batch_norm(w.model, input, 2, 1), output);
        } else if (name == "layer_norm_2d") {
            w.output(layer_norm_2d(w.model, input), output);
        } else if (name == "patch_embed") {
            w.output(patch_embed(w.model, input), output);
        } else if (name == "mb_conv") {
            w.output(mb_conv(w.model, input), output);
        } else if (name == "patch_merging") {
            w.output(patch_merging(w.model, input, 64), output);
        } else if (name == "mlp") {
            w.output(mlp(w.model, input), output);
        } else if (name == "attention_rel_bias") {
            w.output(attention_rel_bias(w.model, input, 4, 2), output);
        } else if (name == "tiny_vit_block") {
            w.output(tiny_vit_block(w.model, input, 8, /*dim*/ 4, /*num_heads*/ 2,
                                    /*window_size*/ 5),
                     output);
        } else if (name == "tiny_vit") {
            TinyViTParams p;
            w.output(tiny_vit(w.model, input, p), output);
        } else if (name == "position_embedding_random") {
            float* input_data = reinterpret_cast<float*>(input->data);
            for (int i = 0; i < ggml_nelements(input); ++i) {
                input_data[i] = (input_data[i] / 64.f) * 2.f - 1.f;
            }
            w.output(position_embedding_random(w.model, input), output);
        } else if (name == "embed_points") {
            float* input_data = reinterpret_cast<float*>(input->data);
            for (int i = 0; i < ggml_nelements(input) - 2; ++i) {
                input_data[i] = transform_coord(input_data[i], 1.0f, 64);
            }
            w.output(embed_points(w.model, input), output);
        } else if (name == "no_mask_embed") {
            w.output(no_mask_embed(w.model, 8), output);
        } else if (name == "attention") {
            Tensor q = input;
            Tensor k = w.model.weights("input_k");
            Tensor v = w.model.weights("input_v");
            w.output(attention(w.model, q, k, v, 2), output);
        } else if (name.starts_with("two_way_attention_block")) {
            Tensor queries = input;
            Tensor keys = w.model.weights("input_keys");
            Tensor query_pe = w.model.weights("input_query_pe");
            Tensor key_pe = w.model.weights("input_key_pe");
            bool skip_first_layer_pe = name.ends_with("skip_first_layer_pe");
            auto [result_queries, result_keys] = two_way_attention_block(
                w.model, queries, keys, query_pe, key_pe, 2, skip_first_layer_pe);
            w.output(result_queries, output);
            w.output(result_keys, inputs[input_count - 1]);
        } else if (name == "two_way_transformer") {
            Tensor image_embedding = input;
            Tensor image_pe = w.model.weights("input_image_pe");
            Tensor point_embedding = w.model.weights("input_point_embedding");
            auto [result_queries, result_keys] = two_way_transformer(
                w.model, image_embedding, image_pe, point_embedding, 2, 2);
            w.output(result_queries, output);
            w.output(result_keys, inputs[input_count - 1]);
        } else if (name == "hypernetwork_mlp") {
            w.output(hypernetwork_mlp(w.model, input, 2), output);
        } else if (name == "output_upscaling") {
            w.output(upscale_outputs(w.model, input), output);
        } else if (name == "predict_masks") {
            Tensor image_embeddings = input;
            Tensor sparse_prompt = w.model.weights("input_sparse_prompt");
            Tensor dense_prompt = w.model.weights("input_dense_prompt");
            auto [masks, iou] = predict_masks(
                w.model, image_embeddings, sparse_prompt, dense_prompt, 2);
            w.output(masks, output);
            w.output(iou, inputs[input_count - 1]);
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