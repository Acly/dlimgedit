#pragma once

#include "lazy.hpp"
#include "segmentation.hpp"
#include "tensor.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <mutex>
#include <type_traits>

namespace dlimg {
using Path = std::filesystem::path;
struct SegmentAnythingModel;
struct BiRefNetModel;

class EnvironmentImpl {
  public:
    Backend backend = Backend::cpu;
    Path model_directory;
    int thread_count = 1;
    Ort::Env onnx_env;
    Ort::MemoryInfo memory_info;

    static Path verify_path(std::string_view path);
    static bool is_supported(Backend);

    EnvironmentImpl(Options const&);

    SegmentAnythingModel& segment_anything_model();
    BiRefNetModel& birefnet_model();

    ~EnvironmentImpl();

  private:
    Lazy<SegmentAnythingModel> segment_anything_;
    Lazy<BiRefNetModel> birefnet_;
};

bool has_cuda_device();
bool has_dml_device();

} // namespace dlimg
