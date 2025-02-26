#pragma once

#include <dlimgedit/dlimgedit.hpp>

namespace dlimg {

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels);
void save_image(ImageView const& img, char const* filepath);

Image resize(ImageView const&, Extent target);
void resize_mask(ImageView const&, Extent target, uint8_t* output);

struct PixelAccessor {
    int stride_x;
    int stride_c;
    std::array<int, 3> channel_map;

    PixelAccessor(ImageView image) : PixelAccessor(image.extent, image.channels) {}

    PixelAccessor(Extent extent, Channels channels) {
        stride_c = count(channels);
        stride_x = extent.width * stride_c;
        switch (channels) {
        case Channels::bgra:
            channel_map = {2, 1, 0};
            break;
        case Channels::argb:
            channel_map = {1, 2, 3};
            break;
        case Channels::mask:
            channel_map = {0, 0, 0};
            break;
        default:
            channel_map = {0, 1, 2};
            break;
        }
    }

    uint8_t get(uint8_t const* pixels, int x, int y, int c) const {
        return pixels[y * stride_x + x * stride_c + channel_map[c]];
    }

    void set(uint8_t* pixels, int x, int y, int c, uint8_t value) {
        pixels[y * stride_x + x * stride_c + channel_map[c]] = value;
    }
};

} // namespace dlimg