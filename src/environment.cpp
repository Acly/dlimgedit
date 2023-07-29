#include "environment.hpp"
#include "platform.hpp"
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
    constexpr char const* cpu_provider = "CPUExecutionProvider";
    constexpr char const* gpu_provider =
        is_windows ? "DmlExecutionProvider" : "CUDAExecutionProvider";

    auto requested = backend == Backend::gpu ? gpu_provider : cpu_provider;
    auto providers = Ort::GetAvailableProviders();
    return std::find(providers.begin(), providers.end(), requested) != providers.end();
}

Ort::Env init_onnx() {
    if (OrtGetApiBase()->GetApi(ORT_API_VERSION) == nullptr) {
        if (is_windows) {
            throw Exception("Could not load onnxruntime library, version mismatch. Make sure "
                            "onnxruntime.dll is in the same directory as the executable.");
        }
        throw Exception("Could not load onnxruntime library, version mismatch.");
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
