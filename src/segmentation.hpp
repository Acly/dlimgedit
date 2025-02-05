#pragma once

#include "image.hpp"
#include "lazy.hpp"
#include "session.hpp"
#include "tensor.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <cmath>
#include <optional>
#include <span>

namespace dlimg {
class EnvironmentImpl;
using Eigen::Array3f;

struct SegmentAnythingModel {
    Session image_embedder;
    Shape image_embedding_shape;

    Session& single_mask_decoder();
    Session& multi_mask_decoder();
    Tensor<float, 4> input_mask;    // always zero
    TensorArray<float, 1> has_mask; // always zero

    explicit SegmentAnythingModel(EnvironmentImpl&);

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
                      std::span<float, 3> out_accuracies) const;

    Extent extent() const { return image_size_.original; }

  private:
    EnvironmentImpl const& env_;
    SegmentAnythingModel& model_;

    ResizeLongestSide image_size_;
    Tensor<float, 4> image_embedding_;
};

Tensor<float, 3> create_image_tensor(ImageView const&);

void write_mask_image(TensorMap<float const, 4> const&, int index, Extent const&,
                      uint8_t* out_mask);

enum class BiRefNetModelKind { general, high_res };

struct BiRefNetModel {
    BiRefNetModelKind kind;
    Session session;
    Shape input_shape;

    explicit BiRefNetModel(EnvironmentImpl&, BiRefNetModelKind);
};

struct BiRefNet {
    static void segment(EnvironmentImpl&, ImageView const&, uint8_t* out_mask);

    static Tensor<float, 4> prepare_image(ImageView const&, Array3f mean, Array3f std);

    static Tensor<uint8_t, 2> process_mask(TensorMap<float const, 4> const& mask);
};

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace dlimg