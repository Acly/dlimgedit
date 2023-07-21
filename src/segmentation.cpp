#include "segmentation.hpp"
#include "assert.hpp"
#include "image.hpp"
#include "tensor.hpp"

namespace dlimgedit {
namespace {

const auto image_embedder_input_names = std::array{"input_image"};
const auto image_embedder_output_names = std::array{"image_embeddings"};
const auto image_input_size = 1024;

const auto mask_decoder_input_names =
    std::array{"image_embeddings", "point_coords",   "point_labels",
               "mask_input",       "has_mask_input", "orig_im_size"};
const auto mask_decoder_output_names = std::array{"masks", "iou_predictions", "low_res_masks"};

int scale_coord(int coord, float scale) { return int(coord * scale + 0.5f); }

} // namespace

SegmentationModel::SegmentationModel(EnvironmentImpl& env)
    : image_embedder(env, "mobile_sam_image_encoder.onnx", image_embedder_input_names,
                     image_embedder_output_names),
      mask_decoder(env, "mobile_sam.onnx", mask_decoder_input_names, mask_decoder_output_names) {

    auto image_shape = image_embedder.input_shape(0);
    ASSERT(image_shape.rank() == 3);
    ASSERT(image_shape[0] == image_shape[1]);
    ASSERT(image_shape[2] == 3);

    image_embedding_shape = image_embedder.output_shape(0);
    ASSERT(image_embedding_shape.rank() == 4);

    input_mask = Tensor<float, 4>(Shape{1, 1, 256, 256});
    input_mask.setZero();
    has_mask.setZero();
}

ResizeLongestSide::ResizeLongestSide(int max_side) : max_side_(max_side) {}

ImageView ResizeLongestSide::resize(ImageView const& img) {
    original = img.extent;
    scale = float(max_side_) / float(std::max(img.extent.width, img.extent.height));
    if (scale != 1) {
        auto target =
            Extent(scale_coord(img.extent.width, scale), scale_coord(img.extent.height, scale));
        resized_ = dlimgedit::resize(img, target);
        return *resized_;
    }
    return img;
}

Point ResizeLongestSide::transform(Point p) const {
    return Point(scale_coord(p.x, scale), scale_coord(p.y, scale));
}

Tensor<float, 3> create_image_tensor(ImageView const& image) {
    ASSERT(image.channels == Channels::rgba || image.channels == Channels::rgb);

    auto img = as_tensor(image);
    auto tensor = Tensor<float, 3>(Shape(image.extent, Channels::rgb));
    for (int i = 0; i < img.dimension(0); ++i) {
        for (int j = 0; j < img.dimension(1); ++j) {
            for (int k = 0; k < 3; ++k) {
                tensor(i, j, k) = float(img(i, j, k));
            }
        }
    }
    return tensor;
}

void write_mask_image(TensorMap<float const, 4> const& tensor, int index, Extent const& extent,
                      uint8_t* out_mask) {
    auto mask = TensorMap<uint8_t, 3>(out_mask, Shape(extent, Channels::mask));
    for (int i = 0; i < mask.dimension(0); ++i) {
        for (int j = 0; j < mask.dimension(1); ++j) {
            mask(i, j, 0) = uint8_t(tensor(0, index, i, j) > 0 ? 255 : 0);
        }
    }
}

SegmentationImpl::SegmentationImpl(EnvironmentImpl& env)
    : env_(env), model_(env.segmentation()), image_size_(image_input_size) {}

void SegmentationImpl::process(ImageView const& input_image) {
    auto resized_image = image_size_.resize(input_image);
    auto image_tensor = create_image_tensor(resized_image);
    image_embedding_.resize(model_.image_embedding_shape);

    auto input = std::array{create_input(env_, image_tensor)};
    auto output = std::array{create_output(env_, image_embedding_)};
    model_.image_embedder.run(input, output);
}

Point transform(Point p, Extent original, Extent scaled) {
    float scale = float(scaled.width) / float(original.width);
    return Point{int(p.x * scale + 0.5f), int(p.y * scale + 0.5f)};
}

void SegmentationImpl::get_mask(Point const* point, Region const* region,
                                std::span<uint8_t*, 3> result_masks,
                                std::span<float, 3> result_accuracy) const {
    ASSERT(point || region);
    auto image_size = TensorArray<float, 2>{};
    auto points = Tensor<float, 3>(Shape(1, 2, 2));
    auto labels = Tensor<float, 2>(Shape(1, 2));

    image_size.setValues({float(image_size_.original.height), float(image_size_.original.width)});
    auto set_point = [&](int index, Point point, int label) {
        Point transformed = image_size_.transform(point);
        points(0, index, 0) = float(transformed.x);
        points(0, index, 1) = float(transformed.y);
        labels(0, index) = float(label);
    };
    if (point) {
        set_point(0, *point, 1);
        set_point(1, Point(0, 0), -1);
    } else if (region) {
        set_point(0, region->top_left, 2);
        set_point(1, region->bottom_right, 3);
    }
    auto output = model_.mask_decoder(image_embedding_, points, labels, model_.input_mask,
                                      model_.has_mask, image_size);
    auto masks = as_tensor<float const, 4>(output[0]);
    auto iou_predictions = as_tensor<float const, 2>(output[1]);

    ASSERT(masks.dimension(1) == 4);
    ASSERT(result_masks[0] != nullptr);
    if (result_masks[1] == nullptr) {
        TensorArray<int64_t, 1> best = iou_predictions.argmax(1);
        write_mask_image(masks, int(best[0]), image_size_.original, result_masks[0]);
    } else {
        for (int i = 0; i < 3; ++i) {
            ASSERT(result_masks[i] != nullptr);
            write_mask_image(masks, i + 1, image_size_.original, result_masks[i]);
            result_accuracy[i] = iou_predictions(0, i + 1);
        }
    }
}

} // namespace dlimgedit