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

Environment::Environment(Options const& opts) : m_(std::make_unique<EnvironmentImpl>(opts)) {}
Environment::~Environment() = default;
EnvironmentImpl& Environment::impl() { return *m_; }

EnvironmentImpl::~EnvironmentImpl() = default;

} // namespace dlimgedit
