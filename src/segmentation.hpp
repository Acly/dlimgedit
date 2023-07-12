#pragma once

#include "environment.hpp"
#include "image.hpp"
#include "tensor.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <optional>

namespace dlimgedit {

class SegmentationModel {
  public:
    Session image_embedder;
    Shape image_shape;
    Shape image_embedding_shape;

    Session mask_decoder;
    Tensor<float, 4> input_mask;    // always zero
    TensorArray<float, 1> has_mask; // always zero

    explicit SegmentationModel(EnvironmentImpl&);
};

struct Segmentation::Impl {
    EnvironmentImpl& env;
    SegmentationModel& model;

    Extent original_extent;
    Extent scaled_extent;
    Tensor<float, 4> image_embedding;

    explicit Impl(EnvironmentImpl& env, SegmentationModel& model);
    Image get_mask(std::optional<Point>, std::optional<Region>) const;
};

Tensor<uint8_t, 4> create_image_tensor(ImageView const&, Shape const&);
Image create_mask_image(TensorMap<float const, 4> const&, Extent const&);

} // namespace dlimgedit