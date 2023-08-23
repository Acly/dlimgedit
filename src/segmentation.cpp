#include "segmentation.hpp"
#include "assert.hpp"
#include "box.hpp"
#include "environment.hpp"
#include "image.hpp"
#include "range.hpp"
#include "tensor.hpp"

#include <algorithm>

namespace dlimg {
namespace {

const auto image_embedder_model = "mobile_sam_image_encoder.onnx";
const auto image_embedder_input_names = std::array{"input_image"};
const auto image_embedder_output_names = std::array{"image_embeddings"};
const auto image_input_size = 1024;

const auto mask_decoder_single_model = "sam_mask_decoder_single.onnx";
const auto mask_decoder_multi_model = "sam_mask_decoder_multi.onnx";
const auto mask_decoder_input_names = std::array{"image_embeddings", "point_coords", "point_labels",
    "mask_input", "has_mask_input", "orig_im_size"};
const auto mask_decoder_output_names = std::array{"masks", "iou_predictions", "low_res_masks"};

int scale_coord(int coord, float scale) { return int(coord * scale + 0.5f); }

} // namespace

SegmentationModel::SegmentationModel(EnvironmentImpl& env)
    : image_embedder(env, "segmentation", image_embedder_model, image_embedder_input_names,
          image_embedder_output_names),
      env_(env) {

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

Session& SegmentationModel::single_mask_decoder() {
    return single_mask_decoder_.get_or_create(env_, "segmentation", mask_decoder_single_model,
        mask_decoder_input_names, mask_decoder_output_names);
}

Session& SegmentationModel::multi_mask_decoder() {
    return multi_mask_decoder_.get_or_create(env_, "segmentation", mask_decoder_multi_model,
        mask_decoder_input_names, mask_decoder_output_names);
}

ResizeLongestSide::ResizeLongestSide(int max_side) : max_side_(max_side) {}

ImageView ResizeLongestSide::resize(ImageView const& img) {
    original = img.extent;
    scale = float(max_side_) / float(std::max(img.extent.width, img.extent.height));
    if (scale != 1) {
        auto target =
            Extent(scale_coord(img.extent.width, scale), scale_coord(img.extent.height, scale));
        resized_ = dlimg::resize(img, target);
        return *resized_;
    }
    return img;
}

Point ResizeLongestSide::transform(Point p) const {
    return Point(scale_coord(p.x, scale), scale_coord(p.y, scale));
}

Point transform(Point p, Extent original, Extent scaled) {
    float scale = float(scaled.width) / float(original.width);
    return Point{int(p.x * scale + 0.5f), int(p.y * scale + 0.5f)};
}

Tensor<float, 3> create_image_tensor(ImageView const& image) {
    auto cmap = std::array{0, 1, 2};
    switch (image.channels) {
    case Channels::mask:
        cmap = {0, 0, 0};
        break;
    case Channels::bgra:
        cmap = {2, 1, 0};
        break;
    case Channels::argb:
        cmap = {1, 2, 3};
        break;
    default: // rgb, rgba
        break;
    }
    auto img = as_tensor(image);
    auto tensor = Tensor<float, 3>(Shape(image.extent, Channels::rgb));
    for (int i = 0; i < img.dimension(0); ++i) {
        for (int j = 0; j < img.dimension(1); ++j) {
            for (int k = 0; k < 3; ++k) {
                tensor(i, j, k) = float(img(i, j, cmap[k]));
            }
        }
    }
    return tensor;
}

float write_mask_image(
    TensorMap<float const, 4> const& tensor, int index, Extent const& extent, uint8_t* out_mask) {
    auto mask = TensorMap<uint8_t, 3>(out_mask, Shape(extent, Channels::mask));
    int64_t sum_low = 0;
    int64_t sum_high = 0;
    for (int i = 0; i < mask.dimension(0); ++i) {
        for (int j = 0; j < mask.dimension(1); ++j) {
            float v = tensor(0, index, i, j);
            sum_low += v > -1.f ? 1 : 0;
            sum_high += v > 1.f ? 1 : 0;
            mask(i, j, 0) = uint8_t(v > 0 ? 255 : 0);
        }
    }
    return float(sum_high) / float(sum_low); // stability score
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

void SegmentationImpl::compute_mask(Point const* point, Region const* region,
    std::span<uint8_t*, 3> result_masks, std::span<float, 3> result_accuracy,
    std::span<float, 3> result_stability) const {
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

    bool is_single_mask = result_masks[1] == nullptr;
    auto& mask_decoder =
        is_single_mask ? model_.single_mask_decoder() : model_.multi_mask_decoder();
    auto output = mask_decoder(
        image_embedding_, points, labels, model_.input_mask, model_.has_mask, image_size);
    auto masks = as_tensor<float const, 4>(output[0]);
    auto iou_predictions = as_tensor<float const, 2>(output[1]);

    if (is_single_mask) {
        ASSERT(result_masks[0] != nullptr);
        ASSERT(masks.dimension(1) == 1);
        write_mask_image(masks, 0, image_size_.original, result_masks[0]);
    } else {
        ASSERT(masks.dimension(1) == 4);
        for (int i = 0; i < 3; ++i) {
            ASSERT(result_masks[i] != nullptr);
            result_stability[i] =
                write_mask_image(masks, i + 1, image_size_.original, result_masks[i]);
            result_accuracy[i] = iou_predictions(0, i + 1);
        }
    }
}

struct LocalMaskImpl {
    Image image;
    AlignedBox2i box;
    float iou = 0;
};

std::vector<LocalMask> non_maximum_suppression(std::vector<LocalMaskImpl>& masks, float threshold) {
    auto indices = to_vector(std::views::iota(0, int(masks.size())));
    ranges::sort(indices, [&](int a, int b) { return masks[a].iou > masks[b].iou; }); // descending
    std::vector<LocalMask> result;
    while (!indices.empty()) {
        int i = indices.back();
        indices.pop_back();
        result.push_back(
            LocalMask{std::move(masks[i].image), to_region(masks[i].box), masks[i].iou});
        std::erase_if(indices,
            [&](int j) { return intersection_over_union(masks[i].box, masks[j].box) > threshold; });
    }
    return result;
}

std::vector<LocalMask> segment_image(SegmentationImpl const& seg, int point_count) {
    const float mask_iou_threshold = 0.88f;
    const float box_nms_threshold = 0.7f;
    const float stability_score_threshold = 0.95f;

    auto extent = to_array(seg.extent());
    float longest = float(extent.maxCoeff());
    Array2i point_counts = (Array2f(point_count) * (extent.cast<float>() / longest)).cast<int>();

    std::vector<LocalMaskImpl> masks;
    masks.reserve(point_counts.prod());

    auto temp_masks = generate_array<3>([&]() { return Image(seg.extent(), Channels::mask); });
    auto temp_pixels =
        to_array<3>(views::transform(temp_masks, [&](Image& img) { return img.pixels(); }));

    for (int i = 0; i < point_counts.x(); ++i) {
        for (int j = 0; j < point_counts.y(); ++j) {
            Array2f pointf =
                Array2f(i + 0.5f, j + 0.5f) * (extent.cast<float>() / point_counts.cast<float>());
            Point point = to_point(pointf.cast<int>());
            auto iou = std::array<float, 3>{};
            auto stability = std::array<float, 3>{};
            seg.compute_mask(&point, nullptr, temp_pixels, iou, stability);

            auto mask_indices = views::iota(0, 3);
            auto filtered = mask_indices | views::filter([&](int k) {
                return iou[k] > mask_iou_threshold && stability[k] > stability_score_threshold;
            });
            auto cropped = filtered | views::transform([&](int k) {
                auto&& [cropped, box] = crop_to_bounding_box(std::move(temp_masks[k]));
                return LocalMaskImpl{std::move(cropped), box, iou[k]};
            });
            ranges::copy(cropped, std::back_inserter(masks));
        }
    }
    auto results = non_maximum_suppression(masks, box_nms_threshold);
    return results;
}

} // namespace dlimg