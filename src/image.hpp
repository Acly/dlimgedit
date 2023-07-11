#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include "assert.hpp"

namespace dlimgedit {

template <typename PixelType> struct ImageAccess {
    Extent extent;
    int channels = 0;
    int stride = 0;
    PixelType* pixels = nullptr;

    ImageAccess() = default;
    ImageAccess(Extent extent, int channels, PixelType* pixels)
        : extent(extent), channels(channels), stride(extent.width * channels), pixels(pixels) {}
    ImageAccess(Image& img)
        : ImageAccess(img.extent(), static_cast<int>(img.channels()), img.pixels().data()) {}
    ImageAccess(ImageView const& img)
        : ImageAccess(img.extent, static_cast<int>(img.channels), img.pixels) {}

    int index(int x, int y, int c = 0) const { return y * stride + x * channels + c; }
    int checked_index(int x, int y, int c = 0) const {
        ASSERT(x >= 0 && x < extent.width);
        ASSERT(y >= 0 && y < extent.height);
        ASSERT(c >= 0 && c < channels);
        return index(x, y, c);
    }
    PixelType& operator()(int x, int y, int c = 0) { return pixels[checked_index(x, y, c)]; }
    PixelType const& operator()(int x, int y, int c = 0) const {
        return pixels[checked_index(x, y, c)];
    }
};

} // namespace dlimgedit