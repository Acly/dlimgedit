#include "segmentation.hpp"
#include "assert.hpp"
#include "image.hpp"
#include "tensor.hpp"

namespace dlimgedit {
namespace {

const auto image_embedder_input_names = std::array{"input"};
const auto image_embedder_output_names = std::array{"output"};

const auto mask_decoder_input_names =
    std::array{"image_embeddings", "point_coords",   "point_labels",
               "mask_input",       "has_mask_input", "orig_im_size"};
const auto mask_decoder_output_names = std::array{"masks", "iou_predictions", "low_res_masks"};

} // namespace

SegmentationModel::SegmentationModel(EnvironmentImpl& env)
    : image_embedder(env, "mobile_sam_preprocess.onnx", image_embedder_input_names,
                     image_embedder_output_names),
      mask_decoder(env, "mobile_sam.onnx", mask_decoder_input_names, mask_decoder_output_names) {

    image_shape = image_embedder.input_shape(0);
    ASSERT(image_shape.rank() == 4);
    ASSERT(image_shape[0] == 1);
    ASSERT(image_shape[1] == 3);

    image_embedding_shape = image_embedder.output_shape(0);
    ASSERT(image_embedding_shape.rank() == 4);

    input_mask = Tensor<float, 4>(Shape{1, 1, 256, 256});
    input_mask.setZero();
    has_mask.setZero();
}

Tensor<uint8_t, 4> create_image_tensor(ImageView const& image, Shape const& shape) {
    ASSERT(image.channels == Channels::rgba || image.channels == Channels::rgb);
    ASSERT(image.extent.width <= shape[3]);
    ASSERT(image.extent.height <= shape[2]);
    ASSERT(shape[1] == 3);

    auto img = as_tensor(image);
    auto tensor = Tensor<uint8_t, 4>(shape);
    tensor.setZero();
    // image layout [H x W x C] -> tensor layout [1 x 3 x H+pad x W+pad]
    for (int i = 0; i < img.dimension(0); ++i) {
        for (int j = 0; j < img.dimension(1); ++j) {
            for (int k = 0; k < 3; ++k) {
                tensor(0, 2 - k, i, j) = img(i, j, k);
            }
        }
    }
    return tensor;
}

Image create_mask_image(TensorMap<float const, 4> const& tensor, Extent const& extent) {
    auto mask_image = Image(extent, Channels::mask);
    auto mask = as_tensor(mask_image);
    for (int i = 0; i < mask.dimension(0); ++i) {
        for (int j = 0; j < mask.dimension(1); ++j) {
            mask(i, j, 0) = uint8_t(tensor(0, 0, i, j) > 0 ? 255 : 0);
        }
    }
    return mask_image;
}

Segmentation Segmentation::process(ImageView const& input_image, Environment& env) {
    auto& model = env.impl().segmentation();
    auto image_tensor = create_image_tensor(input_image, model.image_shape);

    auto m = std::make_unique<Impl>(env.impl(), model);
    m->original_extent = input_image.extent;
    m->scaled_extent = input_image.extent;
    m->image_embedding.resize(model.image_embedding_shape);

    auto input = std::array{create_input(env.impl(), image_tensor)};
    auto output = std::array{create_output(env.impl(), m->image_embedding)};
    model.image_embedder.run(input, output);

    return Segmentation(std::move(m));
}

Image Segmentation::Impl::get_mask(std::optional<Point> point, std::optional<Region> region) const {
    ASSERT(point.has_value() || region.has_value());
    auto image_size = TensorArray<float, 2>{};
    auto points = Tensor<float, 3>{};
    auto labels = Tensor<float, 2>{};

    image_size.setValues({float(model.image_shape[2]), float(model.image_shape[3])});
    if (point.has_value()) {
        points.resize(Shape{1, 1, 2});
        points(0, 0, 0) = float(point->x);
        points(0, 0, 1) = float(point->y);
        labels.resize(Shape{1, 1});
        labels(0, 0) = 1;
    } else {
        points.resize(Shape{1, 2, 2});
        points(0, 0, 0) = float(region->origin.x);
        points(0, 0, 1) = float(region->origin.y);
        points(0, 1, 0) = float(region->origin.x + region->extent.width);
        points(0, 1, 1) = float(region->origin.y + region->extent.height);
        labels.resize(Shape{1, 2});
        labels(0, 0) = 2;
        labels(0, 1) = 3;
    }
    auto output = model.mask_decoder(image_embedding, points, labels, model.input_mask,
                                     model.has_mask, image_size);

    return create_mask_image(as_tensor<float const, 4>(output[0]), scaled_extent);
}

Image Segmentation::get_mask(Point point) const { return m_->get_mask(point, {}); }
Image Segmentation::get_mask(Region region) const { return m_->get_mask({}, region); }

Segmentation::Impl::Impl(EnvironmentImpl& env, SegmentationModel& model) : env(env), model(model) {}
Segmentation::Segmentation(std::unique_ptr<Impl>&& impl) : m_(std::move(impl)) {}
Segmentation::Segmentation(Segmentation&&) = default;
Segmentation& Segmentation::operator=(Segmentation&&) = default;
Segmentation::~Segmentation() = default;

} // namespace dlimgedit