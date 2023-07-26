#pragma once

#include "tensor.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <mutex>
#include <type_traits>

namespace dlimg {
using Path = std::filesystem::path;
struct SegmentationModel;

class EnvironmentImpl {
  public:
    Device device = Device::cpu;
    Path model_path;
    int thread_count = 1;
    Ort::Env onnx_env;
    Ort::MemoryInfo memory_info;

    static Path verify_path(std::string_view path);

    EnvironmentImpl(Options const&);

    SegmentationModel& segmentation();

    ~EnvironmentImpl();

  private:
    std::unique_ptr<SegmentationModel> segmentation_;
    std::mutex mutex_;
};

template <typename Tensor>
Ort::Value create_input(EnvironmentImpl const& env, Tensor const& tensor) {
    // Ort::Value doesn't support const even for input tensors
    using T = typename Tensor::Scalar;
    T* mut = const_cast<T*>(tensor.data());
    auto shape = Shape::from_eigen(tensor.dimensions());
    return Ort::Value::CreateTensor<T>(env.memory_info, mut, tensor.size(), shape.data(),
                                       shape.rank());
}

template <typename Tensor> Ort::Value create_output(EnvironmentImpl const& env, Tensor& tensor) {
    static_assert(!std::is_const_v<Tensor>);
    return create_input(env, tensor);
}

class Session {
  public:
    Session(EnvironmentImpl& env, char const* model_name, std::span<char const* const> input_names,
            std::span<char const* const> output_names);

    Shape input_shape(int index) const;
    Shape output_shape(int index) const;

    void run(std::span<Ort::Value const> inputs, std::span<Ort::Value> outputs);
    std::vector<Ort::Value> run(std::span<Ort::Value const> inputs);

    template <typename... Tensors> std::vector<Ort::Value> operator()(Tensors const&... tensors) {
        auto inputs = std::array<Ort::Value, sizeof...(Tensors)>{create_input(env_, tensors)...};
        return run(inputs);
    }

  private:
    EnvironmentImpl& env_;
    Ort::Session session_;
    std::span<char const* const> input_names_;
    std::span<char const* const> output_names_;
};

} // namespace dlimg
