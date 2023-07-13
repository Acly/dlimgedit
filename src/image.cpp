#include <dlimgedit/dlimgedit.hpp>

#include <stb_image.h>
#include <stb_image_write.h>

#include <format>

namespace dlimgedit {

Image::Image(Extent e, Channels c)
    : Image(e, c, std::make_unique<uint8_t[]>(e.width * e.height * static_cast<int>(c))) {}

Image::Image(Extent e, Channels c, std::unique_ptr<uint8_t[]> pixels)
    : extent_(e), channels_(c), pixels_(std::move(pixels)) {}

Image Image::load(std::string_view filepath) {
    int width = 0;
    int height = 0;
    int channels = 0;
    auto const pixels = stbi_load(filepath.data(), &width, &height, &channels, 0);
    if (!pixels) {
        throw Exception(
            std::format("Failed to load image {}: {}", filepath, stbi_failure_reason()));
    }
    if (channels != 1 && channels != 3 && channels != 4) {
        throw Exception(
            std::format("Unsupported number of channels ({}) in {}", channels, filepath));
    }
    return Image(Extent{width, height}, static_cast<Channels>(channels),
                 std::unique_ptr<uint8_t[]>(pixels));
}

void Image::save(ImageView const& img, std::string_view filepath) {
    int comp = static_cast<int>(img.channels);
    if (!stbi_write_png(filepath.data(), img.extent.width, img.extent.height, comp, img.pixels,
                        img.extent.width * comp)) {
        throw Exception(std::format("Failed to save image {}", filepath));
    }
}

ImageView::ImageView(Extent e, uint8_t const* pixels, Channels c)
    : extent(e), stride(e.width), channels(c), pixels(pixels) {}
ImageView::ImageView(Extent e, std::span<uint8_t const> pixels, Channels c)
    : ImageView(e, pixels.data(), c) {}
ImageView::ImageView(Image const& img) : ImageView(img.extent(), img.pixels(), img.channels()) {}

} // namespace dlimgedit
