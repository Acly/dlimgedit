#include "tensor.hpp"

namespace dlimg {

TensorMap<uint8_t, 3> as_tensor(Image& image) {
    return TensorMap<uint8_t, 3>(image.pixels(), Shape(image.extent(), image.channels()));
}

TensorMap<const uint8_t, 3> as_tensor(ImageView image) {
    return TensorMap<const uint8_t, 3>(image.pixels, Shape(image.extent, image.channels));
}

Image copy_to_image(TensorMap<const uint8_t, 3> const& tensor) {
    auto shape = Shape::from_eigen(tensor.dimensions());
    auto image = Image(shape.extent(), shape.channels());
    ASSERT(tensor.size() == int64_t(image.size()));
    std::copy(tensor.data(), tensor.data() + tensor.size(), image.pixels());
    return image;
}

} // namespace dlimg
