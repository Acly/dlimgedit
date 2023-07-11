#pragma once

#include "environment.hpp"
#include "image.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <optional>

namespace dlimgedit {

class SegmentationModel {
  public:
    Ort::Session pre_session;
    std::vector<int64_t> pre_input_shape;
    std::vector<int64_t> pre_output_shape;

    Ort::Session sam_session;

    Extent input_extent() const;
    int64_t input_size() const;
    int64_t processed_size() const;

    explicit SegmentationModel(EnvironmentImpl&);
};

struct Segmentation::Impl {
    EnvironmentImpl& env;
    SegmentationModel& model;

    Extent original_extent;
    Extent scaled_extent;
    std::vector<float> processed_image;

    explicit Impl(EnvironmentImpl& env, SegmentationModel& model);
    Image get_mask(std::optional<Point>, std::optional<Region>) const;
};

std::vector<uint8_t> create_image_tensor(ImageView const&, std::span<int64_t> const& shape);
Image create_mask_image(ImageAccess<const float> const& tensor, Extent const& extent);

} // namespace dlimgedit