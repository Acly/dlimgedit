#include "environment.hpp"
#include "segmentation.hpp"

#include <thread>

namespace dlimgedit {

Path EnvironmentImpl::verify_path(std::string_view path) {
    Path const p = std::filesystem::absolute(path);
    if (!exists(p)) {
        throw Exception(std::format("Model path {} does not exist", path));
    }
    if (!is_directory(p)) {
        throw Exception(std::format("Model path {} is not a directory", path));
    }
    return p;
}

Ort::Env init_onnx() {
    if (OrtGetApiBase()->GetApi(ORT_API_VERSION) == nullptr) {
        throw Exception("Could not load onnxruntime library, version mismatch");
    }
    return Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "dlimgedit");
}

EnvironmentImpl::EnvironmentImpl(Options const& opts)
    : device(opts.device),
      model_path(verify_path(opts.model_path)),
      thread_count(std::thread::hardware_concurrency()),
      onnx_env(init_onnx()),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

SegmentationModel& EnvironmentImpl::segmentation() {
    if (!segmentation_) {
        segmentation_ = std::make_unique<SegmentationModel>(*this);
    }
    return *segmentation_;
}

Ort::Session create_session(EnvironmentImpl& env, char const* model) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(env.thread_count);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (env.device == Device::GPU) {
        opts.AppendExecutionProvider_CUDA({});
    }
    Path model_path = env.model_path / model;
    return Ort::Session(env.onnx_env, model_path.c_str(), opts);
}

Session::Session(EnvironmentImpl& env, char const* model_name,
                 std::span<char const* const> input_names,
                 std::span<char const* const> output_names)
    : env_(env),
      session_(create_session(env, model_name)),
      input_names_(input_names),
      output_names_(output_names) {
    ASSERT(input_names.size() == session_.GetInputCount());
    ASSERT(output_names.size() == session_.GetOutputCount());
}

Shape Session::input_shape(int index) const {
    return Shape(session_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape());
}

Shape Session::output_shape(int index) const {
    return Shape(session_.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape());
}

void Session::run(std::span<Ort::Value const> inputs, std::span<Ort::Value> outputs) {
    ASSERT(inputs.size() == input_names_.size());
    ASSERT(outputs.size() == output_names_.size());
    Ort::RunOptions opts{nullptr};
    session_.Run(opts, input_names_.data(), inputs.data(), inputs.size(), output_names_.data(),
                 outputs.data(), outputs.size());
}

std::vector<Ort::Value> Session::run(std::span<Ort::Value const> inputs) {
    ASSERT(inputs.size() == input_names_.size());
    Ort::RunOptions opts{nullptr};
    return session_.Run(opts, input_names_.data(), inputs.data(), inputs.size(),
                        output_names_.data(), output_names_.size());
}

Environment::Environment(Options const& opts) : m_(std::make_unique<EnvironmentImpl>(opts)) {}
Environment::~Environment() = default;
EnvironmentImpl& Environment::impl() { return *m_; }

EnvironmentImpl::~EnvironmentImpl() = default;

} // namespace dlimgedit
