#include "image.hpp"
#include "mobile_sam.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <fmt/format.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-blas.h>
#include <ggml-cpu.h>
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

void run_sam_ggml2(Path const& model_path, Path const& input_path, dlimg::Point const& point,
                   Path const& output_path) {

    ggml_context* data_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx = &data_ctx;

    auto time = std::chrono::steady_clock::now();

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

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_t backend_blas = ggml_backend_blas_init();
    // ggml_backend_t backend = ggml_backend_vk_init(0);

    // 
    //
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // TODO: fix race in depthwise conv
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //
    ggml_backend_cpu_set_n_threads(backend_cpu, 1);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(model_ctx, backend_cpu);
    ggml_backend_buffer_t buffer_blas = ggml_backend_alloc_ctx_tensors(model_ctx, backend_blas);
    for (ggml_tensor* t = ggml_get_first_tensor(model_ctx); t != nullptr;
         t = ggml_get_next_tensor(model_ctx, t)) {
        auto data_tensor = ggml_get_tensor(data_ctx, ggml_get_name(t));
        ggml_backend_tensor_set(t, ggml_get_data(data_tensor), 0, ggml_nbytes(data_tensor));
    }

    ggml_free(data_ctx);
    gguf_free(gguf_ctx);

    auto time_load = std::chrono::steady_clock::now();

    auto input_path_str = input_path.string();
    auto input_image = Image::load(input_path_str.c_str());
    auto image_data = sam::preprocess_image(input_image);

    ggml_init_params graph_ctx_params{};
    graph_ctx_params.mem_size = GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() +
                                ggml_graph_overhead();
    graph_ctx_params.no_alloc = true;
    ggml_context* graph_ctx = ggml_init(graph_ctx_params);

    ggml_cgraph* graph = ggml_new_graph(graph_ctx);

    Model model{model_ctx, graph_ctx, graph};

    std::vector<float> image_embeddings_data;

    {
        ggml_tensor* x = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, 3, 1024, 1024, 1);
        ggml_set_name(x, "input");

        Tensor image_embeddings = ggml_cont(
            model, sam::tiny_vit(model["enc"], x, sam::TinyViTParams{}));
        ggml_set_name(image_embeddings, "image_embeddings");
        ggml_build_forward_expand(graph, image_embeddings);
    }
    {
        ggml_gallocr_t allocr =
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_blas));
        if (!ggml_gallocr_alloc_graph(allocr, graph)) {
            throw std::runtime_error("Failed to allocate graph");
        }

        ggml_tensor* input_image_tensor = ggml_graph_get_tensor(graph, "input");
        ggml_backend_tensor_set(
            input_image_tensor, image_data.data(), 0, ggml_nbytes(input_image_tensor));

        ggml_backend_t backends[] = {backend_blas, backend_cpu};
        ggml_backend_buffer_type_t buffer_types[] = {
            ggml_backend_get_default_buffer_type(backend_blas),
            ggml_backend_get_default_buffer_type(backend_cpu),
        };
        auto sched = ggml_backend_sched_new(
            backends, buffer_types, 2, ggml_graph_size(graph), false);

        // Timer timer(graph, sched);
        ggml_backend_sched_set_eval_callback(sched, debug_printer, nullptr);
        ggml_backend_sched_graph_compute(sched, graph);
        // timer.print();

        auto image_embeddings_out = ggml_graph_get_tensor(graph, "image_embeddings");
        image_embeddings_data.resize(ggml_nelements(image_embeddings_out));
        ggml_backend_tensor_get(image_embeddings_out, image_embeddings_data.data(), 0,
                                ggml_nbytes(image_embeddings_out));
        // print_tensor(image_embeddings_out);

        ggml_gallocr_free(allocr);
        ggml_free(graph_ctx);
    }

    auto time_image_embed = std::chrono::steady_clock::now();

    graph_ctx = ggml_init(graph_ctx_params);
    graph = ggml_new_graph(graph_ctx);
    model.graph_context = graph_ctx;
    model.graph = graph;

    // { // load image embeddings from file
    //     auto fd = fopen("_image_embeddings.raw", "rb");
    //     if (!fd) {
    //         throw std::runtime_error("Failed to open _image_embeddings.raw");
    //     }
    //     image_embeddings_data.resize(64 * 64 * 256);
    //     fread(image_embeddings_data.data(), sizeof(float), image_embeddings_data.size(), fd);
    //     fclose(fd);
    // }

    {
        Tensor image_embeddings = ggml_new_tensor_4d(graph_ctx, GGML_TYPE_F32, 64, 64, 256, 1);
        ggml_set_name(image_embeddings, "image_embeddings");
        ggml_set_input(image_embeddings);

        ggml_tensor* point_coords = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, 2, 2);
        ggml_set_name(point_coords, "point_coords");
        ggml_set_input(point_coords);

        auto prompt_embeddings = sam::encode_prompt(model["prompt_encoder"], point_coords);
        ggml_set_name(prompt_embeddings.sparse, "sparse_prompt");
        ggml_set_name(prompt_embeddings.dense, "dense_prompt");

        auto [masks, iou] = sam::predict_masks(
            model["dec"], image_embeddings, prompt_embeddings.sparse, prompt_embeddings.dense);
        ggml_set_name(masks, "masks");
        ggml_set_name(iou, "iou");
        ggml_set_output(masks);
        ggml_set_output(iou);
        ggml_build_forward_expand(graph, masks);
        ggml_build_forward_expand(graph, iou);
    }
    {
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_cpu));
        if (!ggml_gallocr_alloc_graph(allocr, graph)) {
            throw std::runtime_error("Failed to allocate graph");
        }

        Tensor image_embeddings = ggml_graph_get_tensor(graph, "image_embeddings");
        ggml_backend_tensor_set(
            image_embeddings, image_embeddings_data.data(), 0, ggml_nbytes(image_embeddings));

        auto points = sam::preprocess_prompt(point, input_image.extent());
        ggml_tensor* point_coords = ggml_graph_get_tensor(graph, "point_coords");
        ggml_backend_tensor_set(point_coords, points.data(), 0, ggml_nbytes(point_coords));

        ggml_backend_graph_compute(backend_cpu, graph);
    }

    auto output_iou = ggml_graph_get_tensor(graph, "iou");
    print_tensor(output_iou);

    int n = 256;
    auto output_mask = ggml_graph_get_tensor(graph, "masks");
    std::vector<float> mask_data(4 * n * n);
    ggml_backend_tensor_get(output_mask, mask_data.data(), 0, ggml_nbytes(output_mask));

    // { // load low res masks from file
    //     auto fd = fopen("_low_res_masks.raw", "rb");
    //     if (!fd) {
    //         throw std::runtime_error("Failed to open _low_res_masks.raw");
    //     }
    //     fread(mask_data.data(), sizeof(float), mask_data.size(), fd);
    //     fclose(fd);
    // }

    auto time_mask_decode = std::chrono::steady_clock::now();

    int chosen_mask = 3;
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

} // namespace dlimg

int main(int argc, char** argv) {
    dlimg::run_sam_ggml2("script/.ggml/mobile_sam.gguf", "test/input/cat_and_hat.png",
                         dlimg::Point{320, 210}, "test/result/sam_ggml");
    return 0;
}