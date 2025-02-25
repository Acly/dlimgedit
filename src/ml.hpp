#pragma once

#include "fixed_string.hpp"

#include <ggml.h>

#include <span>

namespace dlimg {

using TensorName = FixedString<GGML_MAX_NAME>;
using Tensor = ggml_tensor*;
using Shape4 = std::array<int64_t, 4>;

struct Model {
    ggml_context* model_context = nullptr;
    ggml_context* graph_context = nullptr;
    ggml_cgraph* graph = nullptr;
    TensorName prefix;

    Model() = default;

    explicit Model(ggml_context* model_context, ggml_context* graph_context = nullptr,
                   ggml_cgraph* graph = nullptr, TensorName prefix = {})
        : model_context(model_context),
          graph_context(graph_context ? graph_context : model_context),
          graph(graph),
          prefix(prefix) {}

    Tensor find(char const* name) const {
        auto full_name = TensorName();
        if (prefix) {
            name = full_name.format("{}.{}", prefix.c_str(), name);
        }
        return ggml_get_tensor(model_context, name);
    }

    Tensor weights(char const* name) const {
        if (Tensor result = find(name)) {
            return result;
        }
        throw std::runtime_error(fmt::format("Tensor not found: {}.{}", prefix.view(), name));
    }

    Model with_prefix(TensorName prefix) const {
        return Model{model_context, graph_context, graph, prefix};
    }

    Model operator[](char const* sub_module) const {
        if (prefix) {
            return with_prefix(TensorName("{}.{}", prefix.c_str(), sub_module));
        } else {
            return with_prefix(TensorName(sub_module));
        }
    }

    Model operator[](int index) const {
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
            model_context, GGML_TYPE_F32, shape[3], shape[2], shape[1], shape[0]);
        GGML_ASSERT(ggml_nbytes(tensor) == data.size_bytes());
        tensor->data = reinterpret_cast<void*>(data.data());
        add_tensor(name, tensor);
    }

    operator ggml_context*() { return graph_context; }
};

} // namespace dlimg