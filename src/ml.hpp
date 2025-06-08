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
#include <fstream>
#include <span>
#include <thread>

namespace dlimg {

using Path = std::filesystem::path;
using TensorName = FixedString<GGML_MAX_NAME>;
using Tensor = ggml_tensor*;
using Shape4 = std::array<int64_t, 4>;
using std::byte;

Shape4 nelements(Tensor t) { return {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}; }

enum class GGMLBackend { cpu = 1, vulkan = 2 };

inline bool is_gpu(GGMLBackend backend) { return backend == GGMLBackend::vulkan; }

inline bool is_float_type(ggml_type t) {
    return t != GGML_TYPE_I8 && t != GGML_TYPE_I16 && t != GGML_TYPE_I32 && t != GGML_TYPE_I64;
}

struct FloatConverter {
    ggml_type target;
    ggml_type_traits const* dst_traits = nullptr;
    std::vector<float> f32_buffer;
    std::vector<byte> dst_buffer;

    explicit FloatConverter(ggml_type target_type) : target(target_type) {
        if (target != GGML_TYPE_COUNT) {
            dst_traits = ggml_get_type_traits(target_type);
        }
    }

    ggml_type target_type(ggml_tensor const* t) const {
        if (target == GGML_TYPE_COUNT || !is_float_type(t->type)) {
            return t->type;
        }
        return target;
    }

    void const* operator()(ggml_tensor const* src, ggml_tensor const* dst) {
        if (target == GGML_TYPE_COUNT || src->type == dst->type) {
            return src->data;
        }
        ASSERT(dst->type == target);

        float const* f32_data = reinterpret_cast<float const*>(src->data);
        if (src->type != GGML_TYPE_F32) {
            if (int64_t(f32_buffer.size()) < ggml_nelements(src)) {
                f32_buffer.resize(ggml_nelements(src));
            }
            ggml_type_traits const* src_traits = ggml_get_type_traits(src->type);
            src_traits->to_float(src->data, f32_buffer.data(), ggml_nelements(src));
            f32_data = f32_buffer.data();
        }
        void const* dst_data = f32_data;
        if (target != GGML_TYPE_F32) {
            if (dst_buffer.size() < ggml_nbytes(dst)) {
                dst_buffer.resize(ggml_nbytes(dst));
            }
            dst_traits->from_float_ref(f32_data, dst_buffer.data(), ggml_nelements(dst));
            dst_data = dst_buffer.data();
        }
        return dst_data;
    }
};

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
            uint32_t nthreads = std::max(1u, std::thread::hardware_concurrency() - 2);
            ggml_backend_cpu_set_n_threads(backend.handle.get(), nthreads);
        }
        return backend;
    }

    operator ggml_backend_t() const { return handle.get(); }

    ggml_type preferred_float_type() const {
        if (kind == GGMLBackend::cpu) {
            return GGML_TYPE_F32;
        }
        return GGML_TYPE_COUNT; // use model's float type
    }
};

struct ModelLoadParams {
    ggml_type float_type = GGML_TYPE_COUNT; // use type stored in GGUF file
    int n_extra_tensors = 0;                // number of extra tensors to allocate in the context
};

struct Model {
    ggml_context_ptr context;
    ggml_backend_t backend;
    GGMLBackend backend_kind = GGMLBackend::cpu;
    ggml_backend_buffer_ptr weights_buffer;
    std::vector<ggml_backend_buffer_ptr> extra_buffers;

    static Model load(Path const& filepath, Backend_ const& backend, ModelLoadParams p = {}) {
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
        int64_t n_weights = gguf_get_n_tensors(gguf_ctx.get());

        ggml_init_params model_ctx_params{};
        model_ctx_params.mem_size = (n_weights + p.n_extra_tensors) * ggml_tensor_overhead();
        model_ctx_params.no_alloc = true;
        ggml_context_ptr model_ctx(ggml_init(model_ctx_params));

        FloatConverter convert(p.float_type);
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf_ctx.get()); ++i) {
            auto name = gguf_get_tensor_name(gguf_ctx.get(), i);
            Tensor orig = ggml_get_tensor(data_ctx, name);
            Tensor dup = ggml_new_tensor(
                model_ctx.get(), convert.target_type(orig), GGML_MAX_DIMS, orig->ne);
            ggml_set_name(dup, name);
        }

        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(model_ctx.get(), backend));

        for (ggml_tensor* t = ggml_get_first_tensor(model_ctx.get()); t != nullptr;
             t = ggml_get_next_tensor(model_ctx.get(), t)) {
            Tensor data_tensor = ggml_get_tensor(data_ctx, ggml_get_name(t));
            void const* data = convert(data_tensor, t);
            ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        }
        return Model{std::move(model_ctx), backend, backend.kind, std::move(buffer), {}};
    }

    bool allocate() {
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) {
            return false; // context contains nothing to allocate
        }
        extra_buffers.push_back(std::move(buffer));
        return true;
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
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, size, false);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return Graph{std::move(ctx_ptr), graph, ggml_gallocr_ptr(allocr)};
    }

    void allocate() {
        bool result = ggml_gallocr_alloc_graph(allocr.get(), graph);
        if (!result) {
            throw std::runtime_error("Failed to allocate buffer for graph");
        }
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

    ModelRef(Model& m)
        : weights_context(m.context.get()),
          graph_context(m.context.get()),
          graph(nullptr),
          backend(m.backend_kind) {}

    ModelRef(Model& m, Graph& g)
        : weights_context(m.context.get()),
          graph_context(g.context.get()),
          graph(g.graph),
          backend(m.backend_kind) {}

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

void set_tensor_data(Tensor tensor, std::span<float> data) {
    ASSERT(ggml_nbytes(tensor) == data.size_bytes());
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

void load_tensor_data(Tensor tensor, Path const& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error(fmt::format("Failed to open file: {}", filepath.string()));
    }
    std::vector<float> data(ggml_nelements(tensor));
    file.read(reinterpret_cast<char*>(data.data()), ggml_nbytes(tensor));
    if (!file) {
        throw std::runtime_error(
            fmt::format("Failed to read data from file: {}", filepath.string()));
    }
    set_tensor_data(tensor, data);
}

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