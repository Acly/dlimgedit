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
    float* data;
    int32_t w;
    int32_t h;
    int32_t c;
    int32_t n;

    size_t size_bytes() const { return n * c * w * h * sizeof(float); }
};

struct Workbench {

    Workbench(int input_count, RawTensor* inputs_raw, RawTensor* output_raw) {
        auto inputs_size =
            std::accumulate(inputs_raw, inputs_raw + input_count, size_t{0},
                            [](size_t acc, const RawTensor& t) { return acc + t.size_bytes(); });

        auto context_params = ggml_init_params{};
        context_params.mem_size = inputs_size + output_raw->size_bytes() +
                                  ggml_tensor_overhead() * (input_count + 1) +
                                  ggml_graph_overhead() + 2048 * ggml_tensor_overhead();
        context_params.no_alloc = false;
        context = ggml_init(context_params);

        for (int i = 0; i < input_count; ++i) {
            auto tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32, inputs_raw[i].w,
                                             inputs_raw[i].h, inputs_raw[i].c, inputs_raw[i].n);
            assert(output_raw[i].size_bytes() == ggml_nbytes(tensor));
            auto input_name = fmt::format("input_{}", i);
            ggml_set_name(tensor, input_name.c_str());
            memcpy(tensor->data, inputs_raw[i].data, inputs_raw[i].size_bytes());
            this->inputs.push_back(tensor);
        }
        graph = ggml_new_graph(context);
    }

    void run(RawTensor* output_raw) {
        assert(output_raw->size_bytes() == ggml_nbytes(output));

        ggml_build_forward_expand(graph, output);
        ggml_graph_compute_with_ctx(context, graph, 1);
        memcpy(output_raw->data, output->data, ggml_nbytes(output));
    }

    ggml_context* context;
    std::vector<ggml_tensor*> inputs;
    ggml_tensor* output;
    ggml_cgraph* graph;
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
            w.output = conv_2d_depth_wise(w.context, w.inputs[0], w.inputs[1]);
        } else if (name == "conv_2d") {
            auto bias = w.inputs.size() > 2 ? w.inputs[2] : nullptr;
            w.output = conv_2d(w.context, w.inputs[0], w.inputs[1], 1, 0, 1, bias);
        } else if (name == "batch_norm_2d") {
            w.output = batch_norm_2d(w.context, w.inputs[0], w.inputs[1], w.inputs[2], w.inputs[3],
                                     w.inputs[4]);
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