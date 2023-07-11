#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include <onnxruntime_cxx_api.h>

#include <filesystem>

namespace dlimgedit {
using Path = std::filesystem::path;
class SegmentationModel;

class EnvironmentImpl {
  public:
    Device device = Device::CPU;
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
};

} // namespace dlimgedit
