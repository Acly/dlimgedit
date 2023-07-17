#include <dlimgedit/dlimgedit.hpp>

#include <stb_image.h>
#include <stb_image_write.h>

#include <format>

namespace dlimgedit {

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
    int comp = static_cast<int>(img.channels);
    if (!stbi_write_png(filepath, img.extent.width, img.extent.height, comp, img.pixels,
                        img.extent.width * comp)) {
        throw Exception(std::format("Failed to save image {}", filepath));
    }
}

} // namespace dlimgedit
