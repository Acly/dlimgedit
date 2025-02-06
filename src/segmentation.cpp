#include "segmentation.hpp"
#include "assert.hpp"
#include "environment.hpp"
#include "image.hpp"
#include "tensor.hpp"

namespace dlimg {

//
// SegmentAnythingModel (SAM)

namespace {

const auto image_embedder_model = "mobile_sam_image_encoder.onnx";
const auto image_embedder_input_names = std::array{"input_image"};
const auto image_embedder_output_names = std::array{"image_embeddings"};
const auto image_input_size = 1024;

const auto mask_decoder_single_model = "sam_mask_decoder_single.onnx";
const auto mask_decoder_multi_model = "sam_mask_decoder_multi.onnx";
const auto mask_decoder_input_names =
    std::array{"image_embeddings", "point_coords",   "point_labels",
               "mask_input",       "has_mask_input", "orig_im_size"};
const auto mask_decoder_output_names = std::array{"masks", "iou_predictions", "low_res_masks"};

int scale_coord(int coord, float scale) { return int(coord * scale + 0.5f); }

} // namespace

SegmentAnythingModel::SegmentAnythingModel(EnvironmentImpl& env)
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

Session& SegmentAnythingModel::single_mask_decoder() {
    return single_mask_decoder_.get_or_create(env_, "segmentation", mask_decoder_single_model,
                                              mask_decoder_input_names, mask_decoder_output_names);
}

Session& SegmentAnythingModel::multi_mask_decoder() {
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
    : env_(env), model_(env.segment_anything_model()), image_size_(image_input_size) {}

void SegmentationImpl::process(ImageView const& input_image) {
    auto resized_image = image_size_.resize(input_image);
    auto image_tensor = create_image_tensor(resized_image);
    image_embedding_.resize(model_.image_embedding_shape);

    auto input = std::array{create_input(env_, image_tensor)};
    auto output = std::array{create_output(env_, image_embedding_)};
    model_.image_embedder.run(input, output);
}

void SegmentationImpl::compute_mask(Point const* point, Region const* region,
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

    bool is_single_mask = result_masks[1] == nullptr;
    auto& mask_decoder =
        is_single_mask ? model_.single_mask_decoder() : model_.multi_mask_decoder();
    auto output = mask_decoder(image_embedding_, points, labels, model_.input_mask, model_.has_mask,
                               image_size);
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
            write_mask_image(masks, i + 1, image_size_.original, result_masks[i]);
            result_accuracy[i] = iou_predictions(0, i + 1);
        }
    }
}

//
// BiRefNet Model

char const* select_birefnet_model(EnvironmentImpl const& env, BiRefNetModelKind kind) {
    Path models_path = env.model_directory / "segmentation";
    char const* cpu = "birefnet_cpu.onnx";
    char const* gpu = "birefnet_gpu.onnx";
    char const* hr_cpu = "birefnet_hr_cpu.onnx";
    char const* hr_gpu = "birefnet_hr_gpu.onnx";

    auto first_existing = [&](auto names) -> char const* {
        for (auto name : names) {
            if (exists(models_path / name)) {
                return name;
            }
        }
        throw Exception("Could not find any BiRefNet model in directory " + models_path.string());
    };
    if (kind == BiRefNetModelKind::general && env.backend == Backend::cpu) {
        return first_existing(std::array{cpu, gpu, hr_cpu, hr_gpu});
    }
    if (kind == BiRefNetModelKind::general && env.backend == Backend::gpu) {
        return first_existing(std::array{gpu, cpu, hr_gpu, hr_cpu});
    }
    if (kind == BiRefNetModelKind::high_res && env.backend == Backend::cpu) {
        return first_existing(std::array{hr_cpu, hr_gpu, cpu, gpu});
    }
    if (kind == BiRefNetModelKind::high_res && env.backend == Backend::gpu) {
        return first_existing(std::array{hr_gpu, hr_cpu, gpu, cpu});
    }
    throw Exception("Invalid BiRefNet model kind");
}

BiRefNetModel::BiRefNetModel(EnvironmentImpl& env, BiRefNetModelKind kind)
    : kind(kind),
      session(env, "segmentation", select_birefnet_model(env, kind)),
      input_shape(session.input_shape(0)) {
    if (input_shape[2] == -1 || input_shape[3] == -1) {
        input_shape = Shape{1, 3, 1024, 1024};
    }
    ASSERT(input_shape.rank() == 4);
    ASSERT(input_shape[1] == 3);
    ASSERT(session.output_shape(0).rank() == 4);
}

void BiRefNet::segment(EnvironmentImpl& env, ImageView const& input_image, uint8_t* out_mask) {
    auto kind = BiRefNetModelKind::general;
    if (input_image.extent.width > 1536 || input_image.extent.height > 1536) {
        kind = BiRefNetModelKind::high_res;
    }
    auto& birefnet = env.birefnet_model(kind);
    auto input_shape = birefnet.input_shape;

    auto model_resolution = Extent(int(input_shape[3]), int(input_shape[2]));
    auto resized_image = resize(input_image, model_resolution);

    const Array3f mean{0.485f, 0.456f, 0.406f};
    const Array3f std{0.229f, 0.224f, 0.225f};
    auto image_tensor = prepare_image(resized_image, mean, std);

    auto output = birefnet.session(image_tensor);

    auto mask_float = as_tensor<float const, 4>(output[0]);
    auto mask_uint8 = process_mask(mask_float);
    auto mask_img = ImageView(mask_uint8.data(), model_resolution, Channels::mask);
    resize_mask(mask_img, input_image.extent, out_mask);
}

Tensor<float, 4> BiRefNet::prepare_image(ImageView const& img, Array3f mean, Array3f std) {
    auto input = as_tensor(img);
    auto output = Tensor<float, 4>(Shape{1, 3, img.extent.height, img.extent.width});
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < img.extent.height; ++y) {
            for (int x = 0; x < img.extent.width; ++x) {
                float value = float(input(y, x, c)) / 255.f;
                output(0, c, y, x) = (value - mean[c]) / std[c];
            }
        }
    }
    return output;
}

Tensor<uint8_t, 2> BiRefNet::process_mask(TensorMap<float const, 4> const& mask) {
    auto width = mask.dimension(3);
    auto height = mask.dimension(2);

    auto output = Tensor<uint8_t, 2>(Shape{height, width});
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float value = sigmoid(mask(0, 0, y, x));
            output(y, x) = uint8_t(value * 255.f);
        }
    }
    return output;
}

} // namespace dlimg