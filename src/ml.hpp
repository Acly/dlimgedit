#pragma once

#include "assert.hpp"
#include "fixed_string.hpp"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml-vulkan.h>
#include <ggml.h>

#include <filesystem>
#include <span>

namespace dlimg {

using Path = std::filesystem::path;
using TensorName = FixedString<GGML_MAX_NAME>;
using Tensor = ggml_tensor*;
using Shape4 = std::array<int64_t, 4>;

Shape4 nelements(Tensor t) { return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}; }

enum class GGMLBackend { cpu = 1, vulkan = 2 };

bool is_gpu(GGMLBackend backend) { return backend == GGMLBackend::vulkan; }

struct Backend_ {
    GGMLBackend kind = GGMLBackend::cpu;
    ggml_backend_ptr handle;

    Backend_() = default;

    static Backend_ init(GGMLBackend kind) {
        Backend_ backend;
        backend.kind = kind;
        if (kind == GGMLBackend::vulkan) {
            backend.handle.reset(ggml_backend_vk_init(0));
        } else {
            backend.handle.reset(ggml_backend_cpu_init());
        }
        return backend;
    }

    operator ggml_backend_t() const { return handle.get(); }
};

struct gguf_context_deleter {
    void operator()(gguf_context* ctx) { gguf_free(ctx); }
};
using gguf_context_ptr = std::unique_ptr<gguf_context, gguf_context_deleter>;

struct Model {
    ggml_context_ptr context;
    ggml_backend_buffer_ptr buffer;
    GGMLBackend backend = GGMLBackend::cpu;

    static Model load_gguf(Path const& filepath, Backend_ const& backend) {
        ggml_context* data_ctx;
        gguf_init_params params;
        params.no_alloc = false;
        params.ctx = &data_ctx;

        auto filepath_str = filepath.string();
        gguf_context_ptr gguf_ctx(gguf_init_from_file(filepath_str.c_str(), params));
        if (!gguf_ctx) {
            throw std::runtime_error("Failed to load GGUF model");
        }
        ggml_context_ptr data_ctx_ptr(data_ctx);

        ggml_init_params model_ctx_params{};
        model_ctx_params.mem_size = gguf_get_n_tensors(gguf_ctx.get()) * ggml_tensor_overhead();
        model_ctx_params.no_alloc = true;
        ggml_context_ptr model_ctx(ggml_init(model_ctx_params));

        for (int64_t i = 0; i < gguf_get_n_tensors(gguf_ctx.get()); ++i) {
            auto name = gguf_get_tensor_name(gguf_ctx.get(), i);
            auto orig = ggml_get_tensor(data_ctx, name);
            auto dup = ggml_dup_tensor(model_ctx.get(), orig);
            ggml_set_name(dup, name);
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(model_ctx.get(), backend));

        for (ggml_tensor* t = ggml_get_first_tensor(model_ctx.get()); t != nullptr;
             t = ggml_get_next_tensor(model_ctx.get(), t)) {
            auto data_tensor = ggml_get_tensor(data_ctx, ggml_get_name(t));
            ggml_backend_tensor_set(t, ggml_get_data(data_tensor), 0, ggml_nbytes(data_tensor));
        }
        return Model{std::move(model_ctx), std::move(buffer), backend.kind};
    }
};

struct Graph {
    ggml_context_ptr context;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_ptr allocr;

    static Graph create(Backend_ const& backend, size_t size = GGML_DEFAULT_GRAPH_SIZE) {
        ggml_init_params graph_ctx_params{};
        graph_ctx_params.mem_size = size * ggml_tensor_overhead() + ggml_graph_overhead();
        graph_ctx_params.no_alloc = true;
        ggml_context* ctx = ggml_init(graph_ctx_params);
        ggml_context_ptr ctx_ptr(ctx);
        ggml_cgraph* graph = ggml_new_graph(ctx);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return Graph{std::move(ctx_ptr), graph, ggml_gallocr_ptr(allocr)};
    }

    void allocate() {
        bool result = ggml_gallocr_alloc_graph(allocr.get(), graph);
        if (!result) {
            throw std::runtime_error("Failed to allocate buffer for graph");
        }
    }

    template <typename T, size_t N = std::dynamic_extent>
    void set_input(Tensor tensor, std::span<T, N> data) {
        ASSERT(ggml_nbytes(tensor) == data.size_bytes());
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }

    void compute(Backend_ const& backend) const { ggml_backend_graph_compute(backend, graph); }
};

struct ModelRef {
    ggml_context* weights_context = nullptr;
    ggml_context* graph_context = nullptr;
    ggml_cgraph* graph = nullptr;
    TensorName prefix;
    GGMLBackend backend = GGMLBackend::cpu;

    ModelRef() = default;

    ModelRef(Model& m, Graph& g)
        : weights_context(m.context.get()),
          graph_context(g.context.get()),
          graph(g.graph),
          backend(m.backend) {}

    explicit ModelRef(ggml_context* weights_context, ggml_context* graph_context = nullptr,
                      ggml_cgraph* graph = nullptr, TensorName prefix = {},
                      GGMLBackend backend = GGMLBackend::cpu)
        : weights_context(weights_context),
          graph_context(graph_context ? graph_context : weights_context),
          graph(graph),
          prefix(prefix),
          backend(backend) {}

    Tensor find(char const* name) const {
        auto full_name = TensorName();
        if (prefix) {
            name = full_name.format("{}.{}", prefix.c_str(), name);
        }
        return ggml_get_tensor(weights_context, name);
    }

    Tensor weights(char const* name) const {
        if (Tensor result = find(name)) {
            return result;
        }
        throw std::runtime_error(fmt::format("Tensor not found: {}.{}", prefix.view(), name));
    }

    ModelRef with_prefix(TensorName prefix) const {
        return ModelRef{weights_context, graph_context, graph, prefix, backend};
    }

    ModelRef operator[](char const* sub_module) const {
        if (prefix) {
            return with_prefix(TensorName("{}.{}", prefix.c_str(), sub_module));
        } else {
            return with_prefix(TensorName(sub_module));
        }
    }

    ModelRef operator[](int index) const {
        if (prefix) {
            return with_prefix(TensorName("{}.{}", prefix.view(), index));
        } else {
            return with_prefix(TensorName("{}", index));
        }
    }

    void add_tensor(char const* name, Tensor tensor) const {
        auto full_name = TensorName();
        if (prefix) {
            name = full_name.format("{}.{}", prefix.c_str(), name);
        }
        ggml_set_name(tensor, name);
    }

    void create_tensor(char const* name, Shape4 shape, std::span<float> data) {
        auto tensor = ggml_new_tensor_4d(
            weights_context, GGML_TYPE_F32, shape[3], shape[2], shape[1], shape[0]);
        GGML_ASSERT(ggml_nbytes(tensor) == data.size_bytes());
        tensor->data = reinterpret_cast<void*>(data.data());
        add_tensor(name, tensor);
    }

    Tensor named(Tensor tensor) {
        ggml_set_name(tensor, prefix.c_str());
        return tensor;
    }

    operator ggml_context*() { return graph_context; }
};

template <typename T>
struct TensorAlloc {
    Tensor tensor;
    std::unique_ptr<T[]> data;

    explicit TensorAlloc(Tensor tensor) : tensor(tensor) {
        data = std::unique_ptr<T[]>(new T[ggml_nelements(tensor)]);
    }

    void copy_to_backend_buffer() const {
        ggml_backend_tensor_set(tensor, data.get(), 0, ggml_nbytes(tensor));
    }
};

} // namespace dlimg