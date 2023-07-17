#pragma once

#include "environment.hpp"
#include "image.hpp"
#include "tensor.hpp"
#include <dlimgedit/dlimgedit.hpp>

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

class SegmentationImpl {
  public:
    SegmentationImpl(EnvironmentImpl& env);
    void process(ImageView const&);
    void get_mask(Point const*, Region const*, uint8_t* out_mask) const;

    Extent extent() const { return original_extent_; }

  private:
    EnvironmentImpl& env_;
    SegmentationModel& model_;

    Extent original_extent_;
    Extent scaled_extent_;
    Tensor<float, 4> image_embedding_;
};

Tensor<uint8_t, 4> create_image_tensor(ImageView const&, Shape const&);
void write_mask_image(TensorMap<float const, 4> const&, Extent const&, uint8_t* out_mask);

} // namespace dlimgedit