#include "environment.hpp"
#include "segmentation.hpp"

#include <thread>

namespace dlimg {

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

bool EnvironmentImpl::is_supported(Backend backend) {
    auto providers = Ort::GetAvailableProviders();
    switch (backend) {
    case Backend::cpu:
        return true;
    case Backend::gpu:
        return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") !=
               providers.end();
    }
    return false;
}

Ort::Env init_onnx() {
    if (OrtGetApiBase()->GetApi(ORT_API_VERSION) == nullptr) {
        throw Exception("Could not load onnxruntime library, version mismatch");
    }
    auto env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "dlimgedit");
    env.DisableTelemetryEvents();
    return env;
}

EnvironmentImpl::EnvironmentImpl(Options const& opts)
    : backend(opts.backend),
      model_directory(verify_path(opts.model_directory)),
      thread_count(std::thread::hardware_concurrency()),
      onnx_env(init_onnx()),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

SegmentationModel& EnvironmentImpl::segmentation() { return segmentation_.get_or_create(*this); }

EnvironmentImpl::~EnvironmentImpl() = default;

} // namespace dlimg
