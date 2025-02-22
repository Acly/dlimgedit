#define _USE_MATH_DEFINES // for M_PI
#include <catch2/catch_test_macros.hpp>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <ggml-cpu.h>
#include <ggml.h>
#include <gguf.h>

#pragma optimize("", off)

#include "image.hpp"
#include "mobile_sam.hpp"
#include "primitives.hpp"
#include "test_utils.hpp"

#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/format.h>

using Path = std::filesystem::path;

namespace dlimg {

void run_sam_ggml2(Path const& model_path, Path const& input_path, dlimg::Point const& point,
                   Path const& output_path) {

    ggml_context* data_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &data_ctx;

    auto model_str = model_path.string();
    gguf_context* gguf_ctx = gguf_init_from_file(model_str.c_str(), params);
    if (!gguf_ctx) {
        throw std::runtime_error("Failed to load GGUF model");
    }

    ggml_init_params model_ctx_params{};
    model_ctx_params.mem_size = gguf_get_n_tensors(gguf_ctx) * ggml_tensor_overhead();
    model_ctx_params.no_alloc = true;
    ggml_context* model_ctx = ggml_init(model_ctx_params);

    for (size_t i = 0; i < gguf_get_n_tensors(gguf_ctx); ++i) {
        auto name = gguf_get_tensor_name(gguf_ctx, i);
        auto orig = ggml_get_tensor(data_ctx, name);
        auto dup = ggml_dup_tensor(model_ctx, orig);
        ggml_set_name(dup, name);
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend);
    for (ggml_tensor* t = ggml_get_first_tensor(model_ctx); t != nullptr;
         t = ggml_get_next_tensor(model_ctx, t)) {
        auto data_tensor = ggml_get_tensor(data_ctx, ggml_get_name(t));
        ggml_backend_tensor_set(t, ggml_get_data(data_tensor), 0, ggml_nbytes(data_tensor));
    }

    ggml_free(data_ctx);
    gguf_free(gguf_ctx);

    auto input_path_str = input_path.string();
    auto image_data = preprocess_image(input_path_str.c_str());

    ggml_init_params graph_ctx_params{};
    graph_ctx_params.mem_size =
        GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead();
    graph_ctx_params.no_alloc = true;
    ggml_context* graph_ctx = ggml_init(graph_ctx_params);

    ggml_cgraph* graph = ggml_new_graph(graph_ctx);

    Model model{model_ctx, graph_ctx, graph};

    {
        ggml_tensor* x = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, 1024, 1024, 3, 1);
        ggml_set_name(x, "input");

        ggml_tensor* point_coords = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 1, 2);
        ggml_set_name(point_coords, "point_coords");

        Tensor image_embeddings = tiny_vit(model["enc"], x, TinyViTParams{});
        ggml_set_name(image_embeddings, "image_embeddings");

        auto prompt_embeddings = encode_prompt(model["prompt_encoder"], point_coords);

        ggml_build_forward_expand(graph, image_embeddings);

        ggml_graph_print(graph);
    }
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, graph);

    ggml_tensor* input_image_tensor = ggml_graph_get_tensor(graph, "input");
    ggml_backend_tensor_set(input_image_tensor, image_data.data(), 0,
                            ggml_nbytes(input_image_tensor));

    auto points = std::array{transform_point_coord(float(point.x)),
                             transform_point_coord(float(point.y)), 0.f, 0.f};

    ggml_tensor* point_coords = ggml_graph_get_tensor(graph, "point_coords");
    ggml_backend_tensor_set(point_coords, points.data(), 0, ggml_nbytes(point_coords));

    ggml_backend_graph_compute(backend, graph);

    auto output_tensor = ggml_graph_get_tensor(graph, "image_embeddings");

    fmt::print("output tensor: {} {} {} {}\n", output_tensor->ne[3], output_tensor->ne[2],
               output_tensor->ne[1], output_tensor->ne[0]);

    std::vector<float> output_data(256 * 64 * 64);
    ggml_backend_tensor_get(output_tensor, output_data.data(), 0, ggml_nbytes(output_tensor));

    for (int c = 0; c < 4; ++c) {
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 8; ++x) {
                fmt::print("{:6.3f} ", output_data[c * 64 * 64 + y * 64 + x]);
            }
            fmt::print("\n");
        }
        fmt::print("\n\n");
    }

    // std::vector<float> output_data(3 * 1024 * 1024);
    // auto output_image = dlimg::Image(dlimg::Extent(1024, 1024), dlimg::Channels::rgb);
    // ggml_backend_tensor_get(output_tensor, output_data.data(), 0, ggml_nbytes(output_tensor));

    // for (int y = 0; y < 1024; ++y) {
    //     for (int x = 0; x < 1024; ++x) {
    //         for (int c = 0; c < 3; ++c) {
    //             output_image.pixels()[3 * ((y * 1024) + x) + c] =
    //                 uint8_t(output_data[c * 1024 * 1024 + y * 1024 + x] * 255.0f);
    //         }
    //     }
    // }
    // dlimg::Image::save(output_image, output_path);
}

TEST_CASE("GGML_MOBILE", "[ggml]") {
    Path model_path = dlimg::model_dir() / ".." / "script" / ".ggml" / "mobile_sam.gguf";
    // {
    //     Path input_path = dlimg::test_dir() / "input" / "cat_and_hat.png";
    //     Path output_path = dlimg::test_dir() / "result" / "sam_ggml_mask.png";
    //     run_sam_ggml(model_path, input_path, dlimg::Point{320, 210}, output_path);
    // }
    {
        Path input_path = dlimg::test_dir() / "input" / "cat_and_hat.png";
        {
            Path output_path = dlimg::test_dir() / "result" / "sam_ggml_wardrobe_1.png";
            run_sam_ggml2(model_path, input_path, dlimg::Point{136, 211}, output_path);
        }
        // {
        //     Path output_path = dlimg::test_dir() / "result" / "sam_ggml_wardrobe_2.png";
        //     run_sam_ggml2(model_path, input_path, dlimg::Point{45, 450}, output_path);
        // }
    }
}

} // namespace dlimg