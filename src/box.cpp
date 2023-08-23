#include "box.hpp"
#include "tensor.hpp"

#include <array>

namespace dlimg {

std::tuple<Image, AlignedBox2i> crop_to_bounding_box(Image const& image) {
    auto pixels = as_tensor(image);
    auto dim = pixels.dimensions();
    auto tl = Array2i{dim[0], dim[1]};
    auto br = Array2i(0, 0);
    for (int i = 0; i < dim[0]; ++i) {
        for (int j = 0; j < dim[1]; ++j) {
            if (pixels(i, j, 0) > 0) {
                tl = tl.min(Array2i{i, j});
                br = br.max(Array2i{i, j});
            }
        }
    }
    auto out_shape = Shape{br.x() - tl.x(), br.y() - tl.y(), dim[2]};
    auto out_mask = Image(out_shape.extent(), out_shape.channels());
    auto offset = std::array<int64_t, 3>{tl.x(), tl.y(), 0};
    as_tensor(out_mask) = pixels.slice(offset, out_shape.take<3>());
    return {std::move(out_mask), AlignedBox2i{tl, br}};
};

float intersection_over_union(AlignedBox2i const& a, AlignedBox2i const& b) {
    auto intersection = a.intersection(b);
    if (intersection.isEmpty()) {
        return 0;
    }
    auto intersection_area = intersection.volume();
    auto union_area = a.volume() + b.volume() - intersection_area;
    return float(intersection_area) / float(union_area);
}

} // namespace dlimg