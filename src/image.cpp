#include "assert.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>

#include <format>

namespace dlimg {

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels) {
    auto const pixels =
        stbi_load(filepath, &out_extent->width, &out_extent->height, out_channels, 0);
    if (!pixels) {
        throw Exception(
            std::format("Failed to load image {}: {}", filepath, stbi_failure_reason()));
    }
    if (*out_channels != 1 && *out_channels != 3 && *out_channels != 4) {
        throw Exception(
            std::format("Unsupported number of channels ({}) in {}", *out_channels, filepath));
    }
    return pixels;
}

void save_image(ImageView const& img, char const* filepath) {
    if (!(img.channels == Channels::mask || img.channels == Channels::rgb ||
          img.channels == Channels::rgba)) {
        throw Exception(std::format("Unsupported channel order [{}]", int(img.channels)));
    }
    int comp = count(img.channels);
    if (!stbi_write_png(filepath, img.extent.width, img.extent.height, comp, img.pixels,
                        img.extent.width * comp)) {
        throw Exception(std::format("Failed to save image {}", filepath));
    }
}

Image resize(ImageView const& img, Extent target) {
    ASSERT(img.stride >= img.extent.width * count(img.channels));

    auto resized = Image(target, img.channels);
    int result =
        stbir_resize_uint8_srgb(img.pixels, img.extent.width, img.extent.height, img.stride,
                                resized.pixels(), resized.extent().width, resized.extent().height,
                                0, count(img.channels), STBIR_ALPHA_CHANNEL_NONE, 0);
    if (result == 0) {
        throw Exception(std::format("Failed to resize image {}x{} to {}x{}", img.extent.width,
                                    img.extent.height, target.width, target.height));
    }
    return resized;
}

} // namespace dlimg
