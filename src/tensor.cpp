#include "tensor.hpp"

namespace dlimgedit {

TensorMap<uint8_t, 3> as_tensor(Image& image) {
    return TensorMap<uint8_t, 3>(image.pixels(), Shape(image.extent(), image.channels()));
}

TensorMap<const uint8_t, 3> as_tensor(ImageView image) {
    return TensorMap<const uint8_t, 3>(image.pixels, Shape(image.extent, image.channels));
}

} // namespace dlimgedit
