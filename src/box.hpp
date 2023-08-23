#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <tuple>

namespace dlimg {
using Eigen::AlignedBox2i;
using Eigen::Array2f;
using Eigen::Array2i;

Array2i to_array(Extent e) { return Array2i{e.height, e.width}; }
Point to_point(Array2i a) { return Point{a[1], a[0]}; }
Region to_region(AlignedBox2i b) { return Region{to_point(b.min()), to_point(b.max())}; }

// Discards areas in the image which are 0.
std::tuple<Image, AlignedBox2i> crop_to_bounding_box(Image const&);

float intersection_over_union(AlignedBox2i const& a, AlignedBox2i const& b);

} // namespace dlimg