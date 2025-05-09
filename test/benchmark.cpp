#include "image.hpp"
#include "mobile_sam.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <fmt/format.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-blas.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <ggml.h>
#include <gguf.h>

#include <chrono>
#include <filesystem>
#include <vector>

namespace dlimg {

using Path = std::filesystem::path;

void print_tensor(Tensor t) {
    char const* name = ggml_get_name(t);
    fmt::print("{}: [{}, {}, {}, {}]\n", name, t->ne[3], t->ne[2], t->ne[1], t->ne[0]);

    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));

    for (int c = 0; c < std::min(4i64, t->ne[2]); ++c) {
        for (int y = 0; y < std::min(6i64, t->ne[1]); ++y) {
            for (int x = 0; x < std::min(6i64, t->ne[0]); ++x) {
                fmt::print("{:6.3f} ", data[c * t->nb[2] / 4 + y * t->nb[1] / 4 + x]);
            }
            fmt::print("\n");
        }
        fmt::print("\n\n");
    }
}

struct Timer {

    struct Event {
        Tensor node;
        int64_t elapsed;
    };

    std::vector<Event> events;

    explicit Timer(ggml_cgraph* graph, ggml_backend_sched_t sched) {
        events.reserve(ggml_graph_n_nodes(graph));
        ggml_backend_sched_set_eval_callback(sched, &Timer::callback, this);
    }

    static bool callback(Tensor t, bool ask, void* user_data) {
        if (ask) {
            Timer& timer = *reinterpret_cast<Timer*>(user_data);
            timer.events.push_back({t, ggml_time_us()});
            return true;
        }
        return true;
    }

    void print() {
        for (int i = 0; i < events.size(); ++i) {
            if (i < events.size() - 1) {
                int64_t elapsed = events[i + 1].elapsed - events[i].elapsed;
                fmt::print("{:<16} {:8d} us | {}\n", ggml_op_name(events[i].node->op), elapsed,
                           ggml_get_name(events[i].node));
            }
        }
    }
};

bool debug_printer(Tensor t, bool ask, void* user_data) {
    auto name = std::string_view(ggml_get_name(t));
    if (name.starts_with("PRINT!")) {
        if (ask) {
            return true;
        }
        print_tensor(t);
        return true;
    }
    return !ask;
}

void run_sam_ggml2(Path const& model_path, Path const& input_path, dlimg::Region const& region,
                   Path const& output_path, GGMLBackend backend_kind) {

    auto time = std::chrono::steady_clock::now();

    auto input_path_str = input_path.string();
    auto input_image = Image::load(input_path_str.c_str());
    auto image_data = sam::preprocess_image(input_image);

    Backend_ backend = Backend_::init(backend_kind);
    Model model = Model::load_gguf(model_path, backend);

    auto time_load = std::chrono::steady_clock::now();

    std::vector<float> image_embeddings_data;

    auto time_preprocess = std::chrono::steady_clock::now();
    auto time_first_run = std::chrono::steady_clock::time_point{};
    auto time_second_run = std::chrono::steady_clock::time_point{};

    {
        Graph embed_graph = Graph::create(backend);
        ModelRef m = ModelRef(model, embed_graph);

        Tensor x = ggml_new_tensor_4d(m, GGML_TYPE_F32, 3, 1024, 1024, 1);
        ggml_set_name(x, "input");

        Tensor image_embeddings = sam::tiny_vit(m["enc"], x, sam::TinyViTParams{});
        ggml_set_name(image_embeddings, "image_embeddings");
        ggml_build_forward_expand(embed_graph.graph, image_embeddings);

        embed_graph.allocate();
        embed_graph.set_input(x, std::span(image_data));

        embed_graph.compute(backend);

        time_first_run = std::chrono::steady_clock::now();

        time_second_run = std::chrono::steady_clock::now();

        image_embeddings_data.resize(ggml_nelements(image_embeddings));
        ggml_backend_tensor_get(
            image_embeddings, image_embeddings_data.data(), 0, ggml_nbytes(image_embeddings));
        // print_tensor(image_embeddings_out);
    }

    auto time_image_embed = std::chrono::steady_clock::now();

    int n = 256;
    std::vector<float> mask_data(4 * n * n);

    {
        Graph decode_graph = Graph::create(backend);
        ModelRef m = ModelRef(model, decode_graph);

        Tensor image_embeddings = ggml_new_tensor_4d(m, GGML_TYPE_F32, 256, 64, 64, 1);
        ggml_set_name(image_embeddings, "image_embeddings");
        ggml_set_input(image_embeddings);

        Tensor point_coords = ggml_new_tensor_2d(m, GGML_TYPE_F32, 2, 2);
        ggml_set_name(point_coords, "point_coords");
        ggml_set_input(point_coords);

        // auto prompt_embeddings = sam::embed_points(model["prompt_encoder"], point_coords);
        auto prompt_embeddings = sam::embed_box(m["prompt_encoder"], point_coords);
        ggml_set_name(prompt_embeddings, "sparse_prompt");
        auto dense_prompt = sam::no_mask_embed(m["prompt_encoder"]);

        auto [masks, iou] = sam::predict_masks(
            m["dec"], image_embeddings, prompt_embeddings, dense_prompt);
        ggml_set_name(masks, "masks");
        ggml_set_name(iou, "iou");
        ggml_set_output(masks);
        ggml_set_output(iou);
        ggml_build_forward_expand(m.graph, masks);
        ggml_build_forward_expand(m.graph, iou);

        decode_graph.allocate();
        decode_graph.set_input(image_embeddings, std::span(image_embeddings_data));

        auto points = sam::preprocess_prompt(region, input_image.extent());
        decode_graph.set_input(point_coords, std::span(points));

        decode_graph.compute(backend);

        print_tensor(iou);
        ggml_backend_tensor_get(masks, mask_data.data(), 0, ggml_nbytes(masks));
    }

    auto time_mask_decode = std::chrono::steady_clock::now();

    int chosen_mask = 2;
    auto filepath = std::format("{}_mask.png", output_path.string());
    auto data = std::span(mask_data).subspan(chosen_mask * n * n, n * n);
    auto output_mask_img = sam::postprocess_mask(data, input_image.extent());
    Image::save(output_mask_img, filepath.c_str());

    auto time_save = std::chrono::steady_clock::now();

    fmt::print("Load model: {}ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(time_load - time).count());
    fmt::print("Image embeddings: {}ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(time_image_embed - time_load)
                   .count());
    fmt::print("Mask decode: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(
                                          time_mask_decode - time_image_embed)
                                          .count());
    fmt::print("Save: {}ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(time_save - time_mask_decode)
                   .count());
}

struct ggml_tensor* ggml_conv_2d_dw_f32(struct ggml_context* ctx, struct ggml_tensor* a,
                                        struct ggml_tensor* b, int s0, int s1, int p0, int p1,
                                        int d0, int d1) {
    struct ggml_tensor* new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    struct ggml_tensor* im2col = ggml_im2col(
        ctx, new_a, ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]), s0, s1, p0,
        p1, d0, d1, true, GGML_TYPE_F32); // [N * IC, OH, OW, KH * KW]
    struct ggml_tensor* new_b = ggml_reshape_4d(
        ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2],
        b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3],
                            1); // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    struct ggml_tensor* result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(
        ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    return result;
}

void test_depthwise_conv_2d(std::string_view method) {
    ggml_time_init();
    const int iter = 20;

    int c = 256;
    int w = 512;
    int h = 512;
    int n = 1;
    int stride = 1;
    int pad = 1;
    int kw = 3;
    int kh = 3;

    std::vector<float> input_data(c * w * h * n);
    std::vector<float> weight_data(c * kw * kh);
    std::vector<float> output_data(c * w * h * n);

    for (int i = 0; i < input_data.size(); ++i) {
        input_data[i] = 2.f * float(i) / float(input_data.size()) - 1.f;
    }
    for (int i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 2.f * float(i) / float(weight_data.size()) - 1.f;
    }

    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, n);
    ggml_tensor* weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, 1, c);
    ggml_set_input(input);
    ggml_set_input(weight);

    ggml_backend_t backend = ggml_backend_vk_init(0);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    ggml_init_params graph_params{};
    graph_params.mem_size = GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                            ggml_graph_overhead();
    graph_params.no_alloc = true;
    ggml_context* graph_ctx = ggml_init(graph_params);
    ggml_cgraph* graph = ggml_new_graph(graph_ctx);

    ggml_tensor* output = nullptr;
    if (method == "nchw") {
        output = ggml_conv_2d_dw_direct(graph_ctx, weight, input, stride, stride, pad, pad, 1, 1);
    } else if (method == "nhwc") {
        weight = ggml_reshape_4d(graph_ctx, weight, c, 1, kw, kh);
        weight = ggml_permute(graph_ctx, weight, 3, 2, 0, 1);
        input = ggml_reshape_4d(graph_ctx, input, c, w, h, n);
        input = ggml_permute(graph_ctx, input, 2, 0, 1, 3);
        output = ggml_conv_2d_dw_direct(graph_ctx, weight, input, stride, stride, pad, pad, 1, 1);
        output = ggml_permute(graph_ctx, output, 1, 2, 0, 3);
    } else if (method == "old") {
        output = ggml_conv_2d_dw_f32(graph_ctx, weight, input, stride, stride, pad, pad, 1, 1);
    } else {
        fmt::print("Unknown method: {}\n", method);
        return;
    }
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    // warm-up
    ggml_backend_graph_compute(backend, graph);

    auto timings = std::vector<int64_t>(iter);
    for (int i = 0; i < iter; ++i) {
        auto time = ggml_time_us();
        ggml_backend_graph_compute(backend, graph);
        timings[i] = ggml_time_us() - time;
    }
    // compute time mean and std
    double mean = 0;
    for (int i = 0; i < iter; ++i) {
        mean += timings[i];
    }
    mean /= iter;
    double std = 0;
    for (int i = 0; i < iter; ++i) {
        std += (timings[i] - mean) * (timings[i] - mean);
    }
    std = std::sqrt(std / iter);

    fmt::print("Depthwise conv 2d: {} +/- {} ms\n", mean / 1000.0, std / 1000.0);

    ggml_backend_tensor_get(output, output_data.data(), 0, ggml_nbytes(output));
}

void test_conv_2d(std::string_view method) {
    ggml_time_init();
    const int iter = 20;

    int ci = 128;
    int co = 160;
    int w = 256;
    int h = 256;
    int n = 1;
    int stride = 1;
    int pad = 1;
    int kw = 3;
    int kh = 3;

    std::vector<float> input_data(ci * w * h * n);
    std::vector<float> weight_data(ci * co * kw * kh);
    std::vector<float> output_data(co * w * h * n);

    for (int i = 0; i < input_data.size(); ++i) {
        input_data[i] = 2.f * float(i) / float(input_data.size()) - 1.f;
    }
    for (int i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 2.f * float(i) / float(weight_data.size()) - 1.f;
    }

    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, ci, n);
    ggml_tensor* weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, ci, co);
    ggml_set_input(input);
    ggml_set_input(weight);

    ggml_backend_t backends[] = {ggml_backend_vk_init(0), ggml_backend_cpu_init()};
    ggml_backend_cpu_set_n_threads(backends[1], 6);
    ggml_backend_buffer_t buffer1 = ggml_backend_alloc_ctx_tensors(ctx, backends[1]);
    ggml_backend_buffer_t buffer0 = ggml_backend_alloc_ctx_tensors(ctx, backends[0]);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    ggml_init_params graph_params{};
    graph_params.mem_size = GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                            ggml_graph_overhead();
    graph_params.no_alloc = true;
    ggml_context* graph_ctx = ggml_init(graph_params);
    ggml_cgraph* graph = ggml_new_graph(graph_ctx);

    ggml_tensor* output = nullptr;
    if (method == "nchw") {
        output = ggml_conv_2d(graph_ctx, weight, input, stride, stride, pad, pad, 1, 1);
    } else if (method == "nhwc") {
        weight = ggml_reshape_4d(graph_ctx, weight, ci, kw, kh, co);
        Tensor permuted_weight = ggml_permute(graph_ctx, weight, 2, 0, 1, 3);
        input = ggml_reshape_4d(graph_ctx, input, ci, w, h, n);
        input = ggml_permute(graph_ctx, input, 2, 0, 1, 3);
        {
            Tensor cols = ggml_im2col(graph_ctx, permuted_weight, input, stride, stride, pad, pad,
                                      1, 1, true, GGML_TYPE_F32);
            Tensor a = ggml_reshape_2d(
                graph_ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2] * cols->ne[3]);
            Tensor b = ggml_reshape_2d(
                graph_ctx, weight, weight->ne[0] * weight->ne[1] * weight->ne[2], weight->ne[3]);
            input = ggml_mul_mat(graph_ctx, b, a);
            input = ggml_reshape_4d(
                graph_ctx, input, weight->ne[3], cols->ne[1], cols->ne[2], cols->ne[3]);
        }
        output = ggml_permute(graph_ctx, input, 1, 2, 0, 3);
    } else {
        fmt::print("Unknown method: {}\n", method);
        return;
    }
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backends[0]));
    ggml_gallocr_alloc_graph(allocr, graph);

    ggml_backend_buffer_type_t buffer_types[] = {ggml_backend_get_default_buffer_type(backends[0]),
                                                 ggml_backend_get_default_buffer_type(backends[1])};
    auto sched = ggml_backend_sched_new(backends, buffer_types, 2, ggml_graph_size(graph), false);

    // warm-up
    ggml_backend_sched_graph_compute(sched, graph);

    auto timings = std::vector<int64_t>(iter);
    for (int i = 0; i < iter; ++i) {
        auto time = ggml_time_us();
        ggml_backend_sched_graph_compute(sched, graph);
        timings[i] = ggml_time_us() - time;
    }
    // compute time mean and std
    double mean = 0;
    for (int i = 0; i < iter; ++i) {
        mean += timings[i];
    }
    mean /= iter;
    double std = 0;
    for (int i = 0; i < iter; ++i) {
        std += (timings[i] - mean) * (timings[i] - mean);
    }
    std = std::sqrt(std / iter);

    fmt::print("Conv 2d: {} +/- {} ms\n", mean / 1000.0, std / 1000.0);

    ggml_backend_tensor_get(output, output_data.data(), 0, ggml_nbytes(output));
}

void test_conv_transpose_2d(std::string_view method) {
    ggml_time_init();
    const int iter = 10;

    int ci = 128;
    int co = 160;
    int w = 256;
    int h = 256;
    int n = 1;
    int stride = 2;
    int kw = 3;
    int kh = 3;
    int wo = (w - 1) * stride + kw;
    int ho = (h - 1) * stride + kh;

    std::vector<float> input_data(ci * w * h * n);
    std::vector<float> weight_data(ci * co * kw * kh);
    std::vector<float> output_data(co * wo * ho * n);

    for (int i = 0; i < input_data.size(); ++i) {
        input_data[i] = 2.f * float(i) / float(input_data.size()) - 1.f;
    }
    for (int i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = 2.f * float(i) / float(weight_data.size()) - 1.f;
    }

    ggml_init_params params{};
    params.mem_size = 2 * ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, ci, n);
    ggml_tensor* weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, co, ci);
    ggml_set_input(input);
    ggml_set_input(weight);

    ggml_backend_t backend = ggml_backend_vk_init(0);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    ggml_init_params graph_params{};
    graph_params.mem_size = GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                            ggml_graph_overhead();
    graph_params.no_alloc = true;
    ggml_context* graph_ctx = ggml_init(graph_params);
    ggml_cgraph* graph = ggml_new_graph(graph_ctx);

    ggml_tensor* output = nullptr;
    if (method == "nchw") {
        output = ggml_conv_transpose_2d_p0(graph_ctx, weight, input, stride);
    } else if (method == "nhwc") {
        weight = ggml_reshape_4d(graph_ctx, weight, ci, kw, kh, co);
        weight = ggml_permute(graph_ctx, weight, 3, 0, 1, 2);
        input = ggml_reshape_4d(graph_ctx, input, ci, w, h, n);
        input = ggml_permute(graph_ctx, input, 2, 0, 1, 3);
        output = ggml_conv_transpose_2d_p0(graph_ctx, weight, input, stride);
        output = ggml_permute(graph_ctx, output, 1, 2, 0, 3);
    } else {
        fmt::print("Unknown method: {}\n", method);
        return;
    }
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    // warm-up
    ggml_backend_graph_compute(backend, graph);

    auto timings = std::vector<int64_t>(iter);
    for (int i = 0; i < iter; ++i) {
        auto time = ggml_time_us();
        ggml_backend_graph_compute(backend, graph);
        timings[i] = ggml_time_us() - time;
    }
    // compute time mean and std
    double mean = 0;
    for (int i = 0; i < iter; ++i) {
        mean += timings[i];
    }
    mean /= iter;
    double std = 0;
    for (int i = 0; i < iter; ++i) {
        std += (timings[i] - mean) * (timings[i] - mean);
    }
    std = std::sqrt(std / iter);

    fmt::print("Conv transpose 2d: {} +/- {} ms\n", mean / 1000.0, std / 1000.0);

    ggml_backend_tensor_get(output, output_data.data(), 0, ggml_nbytes(output));
    // print_tensor(output);
}

} // namespace dlimg

int main(int argc, char** argv) {
    auto arg1 = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
    if (arg1 == "depthwise_conv_2d") {
        dlimg::test_depthwise_conv_2d(argv[2]);
    } else if (arg1 == "conv_2d") {
        dlimg::test_conv_2d(argv[2]);
    } else if (arg1 == "conv_transpose_2d") {
        dlimg::test_conv_transpose_2d(argv[2]);
    } else if (arg1 == "vulkan") {
        dlimg::run_sam_ggml2("script/.ggml/mobile_sam.gguf", "test/input/cat_and_hat.png",
                             dlimg::Region{dlimg::Point{180, 110}, dlimg::Extent{325, 220}},
                             "test/result/sam_ggml", dlimg::GGMLBackend::vulkan);
    } else {
        dlimg::run_sam_ggml2("script/.ggml/mobile_sam.gguf", "test/input/cat_and_hat.png",
                             dlimg::Region{dlimg::Point{180, 110}, dlimg::Extent{325, 220}},
                             "test/result/sam_ggml", dlimg::GGMLBackend::cpu);
    }
    return 0;
}