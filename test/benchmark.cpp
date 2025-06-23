#include "birefnet.hpp"
#include "esrgan.hpp"
#include "image.hpp"
#include "migan.hpp"
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
#include <numeric>
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
                int xi = x < 3 ? x : t->ne[0] - 6 + x;
                if (x == 3) {
                    fmt::print(" ... ");
                }
                fmt::print("{:6.3f} ", data[c * t->nb[2] / 4 + y * t->nb[1] / 4 + xi]);
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
    Model model = Model::load(model_path, backend);

    auto time_load = std::chrono::steady_clock::now();

    std::vector<float> image_embeddings_data;

    auto time_preprocess = std::chrono::steady_clock::now();
    auto time_first_run = std::chrono::steady_clock::time_point{};
    auto time_second_run = std::chrono::steady_clock::time_point{};

    {
        Graph embed_graph = Graph::create(backend);
        ModelRef m = ModelRef(model, embed_graph);

        Tensor x = create_input(m, "input", GGML_TYPE_F32, {3, 1024, 1024, 1});

        Tensor image_embeddings = sam::tiny_vit(m["enc"], x, sam::TinyViTParams{});
        mark_output(m, image_embeddings, "image_embeddings");

        embed_graph.allocate();
        set_tensor_data(x, image_data);

        compute(backend, embed_graph);

        time_first_run = std::chrono::steady_clock::now();

        time_second_run = std::chrono::steady_clock::now();

        image_embeddings_data = get_tensor_data<float>(image_embeddings);
    }

    auto time_image_embed = std::chrono::steady_clock::now();

    int n = 256;
    std::vector<float> mask_data(4 * n * n);

    {
        Graph decode_graph = Graph::create(backend);
        ModelRef m = ModelRef(model, decode_graph);

        Tensor img_embeddings = create_input(
            m, "image_embeddings", GGML_TYPE_F32, {256, 64, 64, 1});
        Tensor point_coords = create_input(m, "point_coords", GGML_TYPE_F32, {2, 2, 1, 1});

        // auto prompt_embeddings = sam::embed_points(m["prompt_encoder"], point_coords);
        auto prompt_embeddings = sam::embed_box(m["prompt_encoder"], point_coords);
        auto dense_prompt = sam::no_mask_embed(m["prompt_encoder"]);

        auto [masks, iou] = sam::predict_masks(
            m["dec"], img_embeddings, prompt_embeddings, dense_prompt);

        decode_graph.allocate();
        set_tensor_data(img_embeddings, image_embeddings_data);

        auto points = sam::preprocess_prompt(region, input_image.extent());
        set_tensor_data(point_coords, points);

        compute(backend, decode_graph);

        print_tensor(iou);
        mask_data = get_tensor_data<float>(masks);
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

void run_birefnet(Path const& model_path, Path const& input_path, Path const& output_path,
                  GGMLBackend backend_kind) {
    auto time = std::chrono::steady_clock::now();
    auto input_path_str = input_path.string();
    auto input_image = Image::load(input_path_str.c_str());
    auto image_data = birefnet::preprocess_image(input_image, 1024);

    Backend_ backend = Backend_::init(backend_kind);

    ModelLoadParams mparams;
    mparams.float_type = backend.preferred_float_type();
    mparams.n_extra_tensors = 6; // for relative position index and attention masks
    Model model = Model::load(model_path, backend, mparams);

    auto params = birefnet::SwinParams::detect(model);
    {
        ModelRef m = ModelRef(model);
        auto rel_pos_index = birefnet::create_relative_position_index(m, params.window_size);
        auto attn_masks = std::array{
            birefnet::create_attention_mask(m, 256, 256, params.window_size),
            birefnet::create_attention_mask(m, 128, 128, params.window_size),
            birefnet::create_attention_mask(m, 64, 64, params.window_size),
            birefnet::create_attention_mask(m, 32, 32, params.window_size),
            birefnet::create_attention_mask(m, 16, 16, params.window_size)};

        model.allocate();
        rel_pos_index.copy_to_backend_buffer();
        for (auto&& attn_mask : attn_masks) {
            attn_mask.copy_to_backend_buffer();
        }
    }
    {
        Graph graph = Graph::create(backend, 6 * 1024);
        ModelRef m = ModelRef(model, graph);

        Tensor x = create_input(m, "input", GGML_TYPE_F32, {3, 1024, 1024, 1});

        auto result = birefnet::run(m, x, params);

        graph.allocate();
        set_tensor_data(x, image_data);

        auto time = std::chrono::steady_clock::now();
        compute(backend, graph);
        fmt::print("Compute time: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::steady_clock::now() - time)
                                               .count());

        std::vector<float> output_data = get_tensor_data<float>(result);
        Image output_image = Image({1024, 1024}, Channels::mask);
        for (int i = 0; i < output_image.size(); ++i) {
            output_image.pixels()[i] = uint8_t(std::clamp(output_data[i], 0.f, 1.f) * 255);
        }

        Image output_mask = Image(input_image.extent(), Channels::mask);
        resize_mask(output_image, input_image.extent(), output_mask.pixels());
        std::string output_path_mask = fmt::format("{}_mask.png", output_path.string());
        Image::save(output_mask, output_path_mask.c_str());
        fmt::print("Saved to {}\n", output_path_mask);

        auto input_float =
            std::vector<rgba32_t>(input_image.extent().width * input_image.extent().height);
        image_to_float(input_image, image_span<rgba32_t>(input_image.extent(), input_float.data()));
        auto resized_mask =
            std::vector<float>(input_image.extent().width * input_image.extent().height);
        image_to_float(output_mask, image_span<float>(input_image.extent(), resized_mask));
        auto foreground = estimate_foreground(input_float, resized_mask, input_image.extent());

        Image output_image_rgba = Image(input_image.extent(), Channels::rgba);
        uint8_t* rgba = output_image_rgba.pixels();
        for (int i = 0; i < input_image.extent().width * input_image.extent().height; ++i) {
            float4 fg = foreground[i];
            rgba[i * 4 + 0] = uint8_t(255.f * fg[0]);
            rgba[i * 4 + 1] = uint8_t(255.f * fg[1]);
            rgba[i * 4 + 2] = uint8_t(255.f * fg[2]);
            rgba[i * 4 + 3] = output_mask.pixels()[i];
        }
        std::string output_path_rgba = fmt::format("{}_rgba.png", output_path.string());
        Image::save(output_image_rgba, output_path_rgba.c_str());
        fmt::print("Saved to {}\n", output_path_rgba);
    }
}

void run_migan(Path const& model_path, Path const& image_path, Path const& mask_path,
               Path const& output_path, GGMLBackend backend) {

    Backend_ backend_ = Backend_::init(backend);
    ModelLoadParams mparams;
    mparams.float_type = backend_.preferred_float_type();
    Model model = Model::load(model_path, backend_, mparams);
    auto params = migan::MIGANParams::detect(model);
    params.invert_mask = true;
    model.allocate();

    auto input_image = Image::load(image_path.string().c_str());
    auto mask_image = Image::load(mask_path.string().c_str());
    auto image_data = migan::preprocess(input_image, mask_image, params);

    Graph graph = Graph::create(backend_, GGML_DEFAULT_GRAPH_SIZE);
    ModelRef m = ModelRef(model, graph);

    Tensor x = create_input(
        m, "input", GGML_TYPE_F32, {4, params.resolution, params.resolution, 1});
    Tensor result = migan::generate(m, x, params);

    graph.allocate();
    set_tensor_data(x, image_data);

    auto time = std::chrono::steady_clock::now();

    compute(backend_, graph);

    auto time_end = std::chrono::steady_clock::now();
    fmt::print("MIGAN processing time: {}ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time).count());

    std::vector<float> output_data = get_tensor_data<float>(result);
    Image output_image = migan::postprocess(output_data, input_image.extent(), params);
    Image composited(input_image.extent(), Channels::rgb);
    alpha_composite(output_image, input_image, mask_image, composited.pixels());

    auto output_path_str = fmt::format("{}_output.png", output_path.string());
    Image::save(composited, output_path_str.c_str());
    fmt::print("Saved to {}\n", output_path_str);
}

void run_esrgan(Path const& model_path, Path const& input_path, Path const& output_path,
                GGMLBackend backend) {
    Backend_ backend_ = Backend_::init(backend);
    ModelLoadParams mparams;
    mparams.float_type = backend_.preferred_float_type();
    Model model = Model::load(model_path, backend_, mparams);
    auto params = esrgan::ESRGANParams{};
    model.allocate();

    auto input_image = Image::load(input_path.string().c_str());
    auto tiles = tile_layout(input_image.extent(), 224, 16);
    auto out_tiles = tile_layout::scale(tiles, params.scale);
    auto image_data = std::vector<float>(3 * tiles.tile_size[0] * tiles.tile_size[1]);
    auto tile = image_span<rgb32_t>(tiles.tile_extent(), image_data);
    auto out_data = std::vector<float>(
        3 * input_image.extent().width * params.scale * input_image.extent().height * params.scale,
        0.0f);
    auto out = image_span<rgb32_t>(
        {input_image.extent().width * params.scale, input_image.extent().height * params.scale},
        out_data);
    auto out_tile_data = std::vector<float>(out_tiles.tile_size[0] * out_tiles.tile_size[1] * 3);
    auto out_tile = image_span<rgb32_t>(out_tiles.tile_extent(), out_tile_data);

    Graph graph = Graph::create(backend_, 4 * 1024);
    ModelRef m = ModelRef(model, graph);

    Tensor x = create_input(
        m, "input", GGML_TYPE_F32, {3, tiles.tile_size[0], tiles.tile_size[1], 1});
    Tensor result = esrgan::upscale(m, x, params);

    graph.allocate();

    auto time = std::chrono::steady_clock::now();
    for (int t = 0; t < tiles.total(); ++t) {
        i32x2 tile_coord = tiles.coord(t);
        image_to_float(input_image, tile, float4(0), float4(1), tiles.start(tile_coord));
        set_tensor_data(x, tile.as_float());
        compute(backend_, graph);
        ggml_backend_tensor_get(result, out_tile_data.data(), 0, ggml_nbytes(result));
        merge_tile(out_tile, out, tile_coord, out_tiles);
    }
    auto time_end = std::chrono::steady_clock::now();
    fmt::print("ESRGAN processing time: {}ms\n",
               std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time).count());

    Image output_image = Image(
        {input_image.extent().width * 4, input_image.extent().height * 4}, Channels::rgb);
    image_from_float(out_data, std::span(output_image.pixels(), output_image.size()));

    auto output_path_str = fmt::format("{}_output.png", output_path.string());
    Image::save(output_image, output_path_str.c_str());
    fmt::print("Saved to {}\n", output_path_str);
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
    auto sched = ggml_backend_sched_new(
        backends, buffer_types, 2, ggml_graph_size(graph), false, false);

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

    int ci = 160;
    int co = 128;
    int w = 256;
    int h = 128;
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

const Shape4 no_permutation = Shape4{0, 1, 2, 3};

struct concat_src {
    Shape4 ne;
    Shape4 p = no_permutation;
    ggml_tensor* tensor = nullptr;
    ggml_tensor* permuted = nullptr;
    std::vector<float> data;
};

void permute(void* data, ggml_tensor* src, Shape4 p) {
    Backend_ backend = Backend_::init(GGMLBackend::cpu);
    Model model = Model::init(backend, 4);
    Graph graph = Graph::create(backend);
    ModelRef m = ModelRef(model, graph);
    Tensor input = ggml_dup(m, src);
    Tensor output = ggml_cont(m, ggml_permute(m, input, p[0], p[1], p[2], p[3]));
    mark_output(m, output, "result");
    graph.allocate();
    ggml_backend_tensor_set(input, data, 0, ggml_nbytes(input));
    compute(backend, graph);
    ggml_backend_tensor_get(output, data, 0, ggml_nbytes(output));
}

std::vector<float> concat_reference(std::vector<concat_src> const& src, int dim) {
    Shape4 oe = nelements(src[0].permuted);
    for (int t = 1; t < src.size(); ++t) {
        oe[dim] += src[t].ne[dim];
    }
    std::vector<float> result(oe[0] * oe[1] * oe[2] * oe[3]);
    int64_t o[4] = {0, 0, 0, 0};

    for (int t = 0; t < src.size(); ++t) {
        Shape4 ne = nelements(src[t].permuted);

        for (int i3 = 0; i3 < ne[3]; ++i3) {
            for (int i2 = 0; i2 < ne[2]; ++i2) {
                for (int i1 = 0; i1 < ne[1]; ++i1) {
                    for (int i0 = 0; i0 < ne[0]; ++i0) {

                        size_t src_idx = i3 * ne[2] * ne[1] * ne[0] +
                                         i2 * ne[1] * ne[0] +
                                         i1 * ne[0] + i0;
                        size_t dst_idx = (i3 + o[3]) * oe[2] * oe[1] * oe[0] +
                                         (i2 + o[2]) * oe[1] * oe[0] +
                                         (i1 + o[1]) * oe[0] + (i0 + o[0]);

                        result[dst_idx] = src[t].data[src_idx];
                    }
                }
            }
        }
        o[dim] += ne[dim];
    }
    return result;
}

void test_concat(std::vector<concat_src> input, int dim) {
    Backend_ backend = Backend_::init(GGMLBackend::cpu);
    ggml_backend_cpu_set_n_threads(backend, 3);
    Model model = Model::init(backend, 4);
    Graph graph = Graph::create(backend);
    ModelRef m = ModelRef(model, graph);

    std::vector<ggml_tensor*> to_concat;
    for (int i = 0; i < input.size(); ++i) {
        concat_src& src = input[i];
        src.tensor = create_input(m, "t", GGML_TYPE_F32, src.ne);
        if (src.p != no_permutation) {
            src.permuted = ggml_permute(m, src.tensor, src.p[0], src.p[1], src.p[2], src.p[3]);
        } else {
            src.permuted = src.tensor;
        }
        to_concat.push_back(src.permuted);
        src.data = std::vector<float>(ggml_nelements(src.tensor));
        std::iota(src.data.begin(), src.data.end(), i * 1000.f);
    }
    Tensor result = ggml_concat_n(m, to_concat.data(), to_concat.size(), dim);
    mark_output(m, result, "result");

    graph.allocate();

    for (concat_src& src : input) {
        ggml_backend_tensor_set(src.tensor, src.data.data(), 0, ggml_nbytes(src.tensor));
    }
    compute(backend, graph);

    std::vector<float> result_data(ggml_nelements(result));
    ggml_backend_tensor_get(result, result_data.data(), 0, ggml_nbytes(result));

    // Compute expected result with simple operations
    for (concat_src& src : input) {
        if (src.p != no_permutation) {
            permute(src.data.data(), src.tensor, src.p);
        }
    }
    std::vector<float> expected = concat_reference(input, dim);

    // Compare results
    if (result_data.size() != expected.size()) {
        fmt::print("Result size mismatch: {} vs {}\n", result_data.size(), expected.size());
        return;
    }
    for (size_t i = 0; i < result_data.size(); ++i) {
        if (result_data[i] != expected[i]) {
            fmt::print("Mismatch at index {}: {} vs {}\n", i, result_data[i], expected[i]);
            return;
        }
    }
    fmt::print("Concat test passed for dim {} with {} sources\n", dim, input.size());
}

void bench_concat(std::string_view arg) {
    ggml_time_init();
    const int iter = 100;
    const auto ne0 = std::array<int64_t, 4>{256, 192, 64, 2};
    const auto ne1 = std::array<int64_t, 4>{256, 192, 32, 2};
    const auto ne2 = std::array<int64_t, 4>{256, 192, 16, 2};
    const auto ne3 = std::array<int64_t, 4>{256, 192, 8, 2};

    Backend_ backend = Backend_::init(GGMLBackend::cpu);
    ggml_backend_cpu_set_n_threads(backend, 3);
    Model model = Model::init(backend, 4);
    Graph graph = Graph::create(backend);
    ModelRef m = ModelRef(model, graph);

    Tensor t0 = create_input(m, "t0", GGML_TYPE_F32, ne0);
    Tensor t1 = create_input(m, "t1", GGML_TYPE_F32, ne1);
    Tensor t2 = create_input(m, "t2", GGML_TYPE_F32, ne2);
    Tensor t3 = create_input(m, "t3", GGML_TYPE_F32, ne3);
    Tensor x = nullptr;
    if (arg == "old") {
        x = ggml_concat(m, t0, t1, 2);
        x->op_params[1] = 42;
        x = ggml_concat(m, x, t2, 2);
        x->op_params[1] = 42;
        x = ggml_concat(m, x, t3, 2);
        x->op_params[1] = 42;
    } else {
        auto ts = std::array{t0, t1, t2, t3};
        x = ggml_concat_n(m, ts.data(), ts.size(), 2);
    }
    mark_output(m, x, "result");

    std::vector<float> data0(ggml_nelements(t0), 1.0f);
    std::vector<float> data1(ggml_nelements(t1), 2.0f);
    std::vector<float> data2(ggml_nelements(t2), 3.0f);
    std::vector<float> data3(ggml_nelements(t3), 4.0f);

    graph.allocate();
    set_tensor_data(t0, data0);
    set_tensor_data(t1, data1);
    set_tensor_data(t2, data2);
    set_tensor_data(t3, data3);

    auto t = ggml_time_ms();
    for (int i = 0; i < iter; ++i) {
        compute(backend, graph);
    }
    t = ggml_time_ms() - t;
    double mean = double(t) / double(iter);
    fmt::print("Concat: {:.2f} ms\n", mean);
}

} // namespace dlimg

int main(int argc, char** argv) {
    try {
        auto arg1 = argc > 1 ? std::string_view(argv[1]) : std::string_view{};
        if (arg1 == "depthwise_conv_2d") {
            dlimg::test_depthwise_conv_2d(argv[2]);
        } else if (arg1 == "conv_2d") {
            dlimg::test_conv_2d(argv[2]);
        } else if (arg1 == "conv_transpose_2d") {
            dlimg::test_conv_transpose_2d(argv[2]);
        } else if (arg1 == "concat") {
            dlimg::bench_concat(argv[2]);
        } else if (arg1 == "test_concat") {
            // clang-format off
            // dim 0
            dlimg::test_concat({
                {{6, 5, 3, 2}, dlimg::no_permutation},
                {{9, 5, 3, 2}, dlimg::no_permutation}}, 0);
            // dim 1
            dlimg::test_concat({
                {{16, 4, 3, 2}, dlimg::no_permutation},
                {{16, 5, 3, 2}, dlimg::no_permutation}}, 1);
            // dim 2
            dlimg::test_concat({
                {{16, 5, 3, 2}, dlimg::no_permutation},
                {{16, 5, 7, 2}, dlimg::no_permutation}}, 2);
            // dim 3
            dlimg::test_concat({
                {{16, 5, 3, 1}, dlimg::no_permutation},
                {{16, 5, 3, 2}, dlimg::no_permutation}}, 3);
            // 4 sources
            dlimg::test_concat({
                {{7, 5, 4, 3}, dlimg::no_permutation},
                {{2, 5, 4, 3}, dlimg::no_permutation},
                {{2, 5, 4, 3}, dlimg::no_permutation},
                {{1, 5, 4, 3}, dlimg::no_permutation}}, 0);
            // 3D tensor
            dlimg::test_concat({
                {{7, 3, 2, 1}, dlimg::no_permutation},
                {{7, 5, 2, 1}, dlimg::no_permutation}}, 1);
            // 2D tensor
            dlimg::test_concat({
                {{7, 3, 1, 1}, dlimg::no_permutation},
                {{7, 5, 1, 1}, dlimg::no_permutation}}, 1);
            // 1D tensor
            dlimg::test_concat({
                {{12, 1, 1, 1}, dlimg::no_permutation},
                {{10, 1, 1, 1}, dlimg::no_permutation}}, 0);
            // permutation (result ne is {5, 20, 4, 3})
            dlimg::test_concat({
                {{5, 4, 8, 3}, {0, 2, 1, 3}},
                {{5, 6, 3, 4}, {0, 1, 3, 2}},
                {{5, 3, 4, 3}, {0, 3, 2, 1}},
                {{5, 3, 3, 4}, {0, 3, 1, 2}}}, 1);
            // clang-format on
        } else if (arg1 == "birefnet") {
            auto backend = argc > 2 && std::string_view(argv[2]) == "vulkan"
                               ? dlimg::GGMLBackend::vulkan
                               : dlimg::GGMLBackend::cpu;
            dlimg::run_birefnet("script/.ggml/birefnet-fp16.gguf", "test/input/cat_and_hat.png",
                                "test/result/birefnet_ggml", backend);
        } else if (arg1 == "migan") {
            auto backend = argc > 2 && std::string_view(argv[2]) == "vulkan"
                               ? dlimg::GGMLBackend::vulkan
                               : dlimg::GGMLBackend::cpu;
            dlimg::run_migan("script/.ggml/migan_512_places2-fp16.gguf",
                             "test/input/inpaint_image.png", "test/input/inpaint_mask.png",
                             "test/result/migan_ggml", backend);
        } else if (arg1 == "esrgan") {
            auto backend = argc > 2 && std::string_view(argv[2]) == "vulkan"
                               ? dlimg::GGMLBackend::vulkan
                               : dlimg::GGMLBackend::cpu;
            dlimg::run_esrgan("script/.ggml/4x_NMKD-Superscale-SP_178000_Gh.gguf",
                              "test/input/wardrobe.png", "test/result/esrgan_ggml",
                              backend);
        } else if (arg1 == "vulkan") {
            dlimg::run_sam_ggml2("script/.ggml/mobile_sam.gguf", "test/input/cat_and_hat.png",
                                 dlimg::Region{dlimg::Point{180, 110}, dlimg::Extent{325, 220}},
                                 "test/result/sam_ggml", dlimg::GGMLBackend::vulkan);
        } else {
            dlimg::run_sam_ggml2("script/.ggml/mobile_sam.gguf", "test/input/cat_and_hat.png",
                                 dlimg::Region{dlimg::Point{180, 110}, dlimg::Extent{325, 220}},
                                 "test/result/sam_ggml", dlimg::GGMLBackend::cpu);
        }
    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    } catch (...) {
        fmt::print(stderr, "Unknown error\n");
        return 1;
    }
    return 0;
}