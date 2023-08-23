#pragma once

#include "image.hpp"
#include "lazy.hpp"
#include "session.hpp"
#include "tensor.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <optional>
#include <span>

namespace dlimg {
class EnvironmentImpl;

struct SegmentationModel {
    Session image_embedder;
    Shape image_embedding_shape;

    Session& single_mask_decoder();
    Session& multi_mask_decoder();
    Tensor<float, 4> input_mask;    // always zero
    TensorArray<float, 1> has_mask; // always zero

    explicit SegmentationModel(EnvironmentImpl&);

  private:
    EnvironmentImpl& env_;
    Lazy<Session> single_mask_decoder_;
    Lazy<Session> multi_mask_decoder_;
};

struct ResizeLongestSide {
    Extent original;
    float scale = 1;

    explicit ResizeLongestSide(int max_side);
    ImageView resize(ImageView const& img);
    Point transform(Point) const;

  private:
    int max_side_ = 0;
    std::optional<Image> resized_;
};

class SegmentationImpl {
  public:
    SegmentationImpl(EnvironmentImpl& env);
    void process(ImageView const&);
    void compute_mask(Point const*, Region const*, std::span<uint8_t*, 3> out_masks,
                      std::span<float, 3> out_accuracy, std::span<float, 3> out_stability) const;
    // Segmentation::Mask compute_mask(Point const&) const;

    Extent extent() const { return image_size_.original; }

  private:
    EnvironmentImpl const& env_;
    SegmentationModel& model_;

    ResizeLongestSide image_size_;
    Tensor<float, 4> image_embedding_;
};

Tensor<float, 3> create_image_tensor(ImageView const&);

float write_mask_image(TensorMap<float const, 4> const&, int index, Extent const&,
                       uint8_t* out_mask);

struct LocalMask {
    Image image;
    Region region;
    float accuracy = 0;
};

std::vector<LocalMask> segment_image(SegmentationImpl const& seg, int point_count);

} // namespace dlimg