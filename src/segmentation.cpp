#include "segmentation.hpp"
#include "assert.hpp"
#include "image.hpp"

namespace dlimgedit {
namespace {

Ort::Session create_session(EnvironmentImpl& env, Path const& model) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(env.thread_count);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (env.device == Device::GPU) {
        opts.AppendExecutionProvider_CUDA({});
    }
    Path model_path = env.model_path / model;
    return Ort::Session(env.onnx_env, model_path.c_str(), opts);
}

const auto pre_input_names = std::array{"input"};
const auto pre_output_names = std::array{"output"};

const auto sam_input_names = std::array{"image_embeddings", "point_coords",   "point_labels",
                                        "mask_input",       "has_mask_input", "orig_im_size"};
const auto sam_output_names = std::array{"masks", "iou_predictions", "low_res_masks"};

} // namespace

SegmentationModel::SegmentationModel(EnvironmentImpl& env)
    : pre_session(create_session(env, "mobile_sam_preprocess.onnx")),
      sam_session(create_session(env, "mobile_sam.onnx")) {
    ASSERT(pre_session.GetInputCount() == 1);
    ASSERT(pre_session.GetOutputCount() == 1);
    ASSERT(sam_session.GetInputCount() == 6);
    ASSERT(sam_session.GetOutputCount() == 3);

    pre_input_shape = pre_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    ASSERT(pre_input_shape.size() == 4);
    ASSERT(pre_input_shape[0] == 1);
    ASSERT(pre_input_shape[1] == 3);

    pre_output_shape = pre_session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    ASSERT(pre_output_shape.size() == 4);
}

Extent SegmentationModel::input_extent() const {
    return {static_cast<int>(pre_input_shape[3]), static_cast<int>(pre_input_shape[2])};
}

int64_t SegmentationModel::input_size() const {
    return pre_input_shape[0] * pre_input_shape[1] * pre_input_shape[2] * pre_input_shape[3];
}

int64_t SegmentationModel::processed_size() const {
    return pre_output_shape[0] * pre_output_shape[1] * pre_output_shape[2] * pre_output_shape[3];
}

std::vector<uint8_t> create_image_tensor(ImageView const& image, std::span<int64_t> const& shape) {
    ASSERT(image.channels == Channels::rgba);
    ASSERT(image.extent.width <= shape[3]);
    ASSERT(image.extent.height <= shape[2]);
    ASSERT(shape[1] == 3);

    auto tensor = std::vector<uint8_t>(shape[0] * shape[1] * shape[2] * shape[3]);
    // image layout [W x H x 4] -> tensor layout [3 x H x W]
    auto img = ImageAccess<const uint8_t>(image);
    for (int y = 0; y < img.extent.height; ++y) {
        for (int x = 0; x < img.extent.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                tensor[c * shape[2] * shape[3] + y * shape[3] + x] = img(x, y, 2 - c);
            }
        }
    }
    return tensor;
}

Image create_mask_image(ImageAccess<const float> const& tensor, Extent const& extent) {
    // auto tensor = output_tensor.GetTensorData<float>();
    // auto tensor_stride = output_tensor.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape()[2];
    auto mask_image = Image(extent, Channels::mask);
    auto mask = ImageAccess<uint8_t>(mask_image);
    for (int y = 0; y < mask.extent.height; ++y) {
        for (int x = 0; x < mask.extent.width; ++x) {
            mask(x, y) = uint8_t(tensor(y, x) * 255);
        }
    }
    return mask_image;
}

Segmentation Segmentation::process(ImageView const& input_image, Environment& env) {
    auto& model = env.impl().segmentation();
    auto image_tensor = create_image_tensor(input_image, model.pre_input_shape);

    auto m = std::make_unique<Impl>(env.impl(), model);
    m->original_extent = input_image.extent;
    m->scaled_extent = input_image.extent;
    m->processed_image.resize(model.processed_size());
    auto input_tensor = Ort::Value::CreateTensor<uint8_t>(
        env.impl().memory_info, image_tensor.data(), image_tensor.size(),
        model.pre_input_shape.data(), model.pre_input_shape.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(
        env.impl().memory_info, m->processed_image.data(), m->processed_image.size(),
        model.pre_output_shape.data(), model.pre_output_shape.size());

    model.pre_session.Run(Ort::RunOptions{nullptr}, pre_input_names.data(), &input_tensor, 1,
                          pre_output_names.data(), &output_tensor, 1);
    return Segmentation(std::move(m));
}

Image Segmentation::Impl::get_mask(std::optional<Point> point, std::optional<Region> region) const {
    ASSERT(point.has_value() || region.has_value());
    auto mask_input = std::array<float, 256 * 256>{};
    auto has_mask_input = std::array<float, 1>{0};
    auto image_size_input =
        std::array<float, 2>{float(model.input_extent().height), float(model.input_extent().width)};
    auto point_input = std::array<float, 4>{};
    auto label_input = std::array<float, 2>{};
    int64_t point_count = 0;
    int64_t label_count = 0;
    if (point.has_value()) {
        point_input[0] = float(point->x);
        point_input[1] = float(point->y);
        label_input[0] = 1;
        point_count = 2;
        label_count = 1;
    } else {
        point_input[0] = float(region->origin.x);
        point_input[1] = float(region->origin.y);
        point_input[2] = float(region->origin.x + region->extent.width);
        point_input[3] = float(region->origin.y + region->extent.height);
        label_input[0] = 2;
        label_input[1] = 3;
        point_count = 4;
        label_count = 2;
    }
    auto point_input_shape = std::array<int64_t, 3>{1, point_count / 2, 2};
    auto label_input_shape = std::array<int64_t, 2>{1, label_count};
    auto mask_input_shape = std::array<int64_t, 4>{1, 1, 256, 256};
    auto has_mask_input_shape = std::array<int64_t, 1>{1};
    auto image_size_input_shape = std::array<int64_t, 1>{2};
    auto create_tensor = [&](auto& input, auto& shape) {
        return Ort::Value::CreateTensor<float>(env.memory_info, const_cast<float*>(input.data()),
                                               int64_t(input.size()), shape.data(),
                                               int64_t(shape.size()));
    };
    const auto inputs = std::array{
        create_tensor(processed_image, model.pre_output_shape),
        create_tensor(point_input, point_input_shape),
        create_tensor(label_input, label_input_shape),
        create_tensor(mask_input, mask_input_shape),
        create_tensor(has_mask_input, has_mask_input_shape),
        create_tensor(image_size_input, image_size_input_shape),
    };
    const auto output =
        model.sam_session.Run(Ort::RunOptions{nullptr}, sam_input_names.data(), inputs.data(),
                              inputs.size(), sam_output_names.data(), 3);

    auto mask = Image(scaled_extent, Channels::mask);
    auto mask_access = ImageAccess<uint8_t>(mask);
    auto output_values = output[0].GetTensorData<float>();
    auto output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    for (int y = 0; y < mask.extent().height; ++y) {
        for (int x = 0; x < mask.extent().width; ++x) {
            float value = output_values[y * output_shape[3] + x];
            mask_access(x, y) = value > 0 ? 255 : 0;
        }
    }
    return mask;
}

Image Segmentation::get_mask(Point point) const { return m_->get_mask(point, {}); }
Image Segmentation::get_mask(Region region) const { return m_->get_mask({}, region); }

Segmentation::Impl::Impl(EnvironmentImpl& env, SegmentationModel& model) : env(env), model(model) {}
Segmentation::Segmentation(std::unique_ptr<Impl>&& impl) : m_(std::move(impl)) {}
Segmentation::Segmentation(Segmentation&&) = default;
Segmentation& Segmentation::operator=(Segmentation&&) = default;
Segmentation::~Segmentation() = default;

} // namespace dlimgedit