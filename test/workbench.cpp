#include "mobile_sam.hpp"
#include "primitives.hpp"

#include <fmt/format.h>
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

    Workbench(int input_count, RawTensor* inputs_raw, RawTensor* output_raw) {
        auto context_params = ggml_init_params{};
        context_params.mem_size = output_raw->size_bytes() +
                                  ggml_tensor_overhead() * (input_count + 1) +
                                  ggml_graph_overhead() + 2048 * ggml_tensor_overhead();
        context_params.no_alloc = false;
        context = ggml_init(context_params);
        model = Model{context};

        for (int i = 0; i < input_count; ++i) {
            auto& raw = inputs_raw[i];
            model.create_tensor(raw.name, {raw.n, raw.c, raw.h, raw.w},
                                std::span(raw.data, raw.size()));
        }
        graph = ggml_new_graph(context);
    }

    void run(RawTensor* output_raw) {
        GGML_ASSERT(output_raw->size_bytes() == ggml_nbytes(output));

        ggml_build_forward_expand(graph, output);
        ggml_graph_compute_with_ctx(context, graph, 1);
        memcpy(output_raw->data, output->data, ggml_nbytes(output));
    }

    ggml_context* context;
    ggml_tensor* output;
    ggml_cgraph* graph;
    Model model;
};

} // namespace dlimg

#ifdef __cplusplus
extern "C" {
#endif

API int32_t dlimg_workbench(char const* testcase, int input_count, dlimg::RawTensor* inputs,
                            dlimg::RawTensor* output) {
    using namespace dlimg;

    try {
        auto name = std::string_view(testcase);
        auto w = dlimg::Workbench(input_count, inputs, output);

        if (name == "conv_2d_depth_wise") {
            w.output = conv_2d_depth_wise(w.model, w.model["input"]);
        } else if (name == "conv_2d") {
            w.output = conv_2d(w.model, w.model["input"]);
        } else if (name == "batch_norm_2d") {
            w.output = batch_norm_2d(w.model, w.model["input"]);
        } else if (name == "layer_norm") {
            w.output = layer_norm(w.model, w.model["input"]);
        } else if (name == "linear") {
            w.output = linear(w.model, w.model["input"]);
        } else if (name == "patch_embed") {
            w.output = patch_embed(w.model, w.model["input"]);
        } else if (name == "mb_conv") {
            w.output = mb_conv(w.model, w.model["input"]);
        } else if (name == "patch_merging") {
            w.output = patch_merging(w.model, w.model["input"], 64);
        } else if (name == "mlp") {
            w.output = mlp(w.model, w.model["input"]);
        } else {
            throw std::runtime_error("Unknown testcase: " + std::string(testcase));
        }

        w.run(output);

    } catch (std::exception const& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return -1;
    }
    return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif