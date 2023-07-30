#include "environment.hpp"
#include "platform.hpp"
#include "segmentation.hpp"

#include <dylib.hpp>
#include <fmt/format.h>
#ifdef DLIMG_WINDOWS
#    define WIN32_LEAN_AND_MEAN
#    include <d3d12.h>
#endif

#include <thread>

namespace dlimg {

Path EnvironmentImpl::verify_path(std::string_view path) {
    Path const p = std::filesystem::absolute(path);
    if (!exists(p)) {
        throw Exception(fmt::format("Model path {} does not exist", path));
    }
    if (!is_directory(p)) {
        throw Exception(fmt::format("Model path {} is not a directory", path));
    }
    return p;
}

// Check if a CUDA compatible GPU is available.
bool has_cuda_device() {
    try {
        auto lib = dylib(is_linux ? "cuda" : "nvcuda");
        auto cuInit = lib.get_function<int(unsigned int)>("cuInit");
        auto cuDeviceGetCount = lib.get_function<int(int*)>("cuDeviceGetCount");
        if (cuInit(0) != 0) {
            return false;
        }
        int count = 0;
        int result = cuDeviceGetCount(&count);
        return result == 0 && count > 0;
    } catch (dylib::exception const&) {
        return false;
    }
}

// Check if a DirectX 12 compatible GPU is available.
bool has_dml_device() {
#ifdef DLIMG_WINDOWS
    try {
        auto lib = dylib("d3d12");
        auto create_device = lib.get_function<decltype(D3D12CreateDevice)>("D3D12CreateDevice");
        HRESULT result =
            create_device(nullptr, D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr);
        return SUCCEEDED(result);
    } catch (dylib::exception const&) {
    }
#endif
    return false;
}

bool EnvironmentImpl::is_supported(Backend backend) {
    constexpr char const* cpu_provider = "CPUExecutionProvider";
    constexpr char const* gpu_provider =
        is_windows ? "DmlExecutionProvider" : "CUDAExecutionProvider";

    auto requested = backend == Backend::gpu ? gpu_provider : cpu_provider;
    auto providers = Ort::GetAvailableProviders();
    bool has_provider = std::find(providers.begin(), providers.end(), requested) != providers.end();
    bool has_device =
        backend == Backend::cpu || (is_windows ? has_dml_device() : has_cuda_device());
    return has_provider && has_device;
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
