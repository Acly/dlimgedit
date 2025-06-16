#include "image.hpp"
#include "assert.hpp"

#include <dlimgedit/dlimgedit.hpp>

#include <fmt/format.h>
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>

#include <span>

namespace dlimg {

using std::clamp;

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels) {
    auto const pixels = stbi_load(
        filepath, &out_extent->width, &out_extent->height, out_channels, 0);
    if (!pixels) {
        throw Exception(
            fmt::format("Failed to load image {}: {}", filepath, stbi_failure_reason()));
    }
    if (*out_channels != 1 && *out_channels != 3 && *out_channels != 4) {
        throw Exception(
            fmt::format("Unsupported number of channels ({}) in {}", *out_channels, filepath));
    }
    return pixels;
}

void save_image(ImageView const& img, char const* filepath) {
    if (!(img.channels == Channels::mask || img.channels == Channels::rgb ||
          img.channels == Channels::rgba)) {
        throw Exception(fmt::format("Unsupported channel order [{}]", int(img.channels)));
    }
    int comp = count(img.channels);
    if (!stbi_write_png(filepath, img.extent.width, img.extent.height, comp, img.pixels,
                        img.extent.width * comp)) {
        throw Exception(fmt::format("Failed to save image {}", filepath));
    }
}

Image resize(ImageView const& img, Extent target) {
    ASSERT(img.stride >= img.extent.width * count(img.channels));

    auto resized = Image(target, img.channels);
    int result = stbir_resize_uint8_generic(
        img.pixels, img.extent.width, img.extent.height, img.stride, resized.pixels(),
        resized.extent().width, resized.extent().height, 0, count(img.channels),
        STBIR_ALPHA_CHANNEL_NONE, /*output_stride*/ 0, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT,
        STBIR_COLORSPACE_SRGB, nullptr);
    if (result == 0) {
        throw Exception(fmt::format("Failed to resize image {}x{} to {}x{}", img.extent.width,
                                    img.extent.height, target.width, target.height));
    }
    return resized;
}

void resize_mask(ImageView const& input, Extent target, uint8_t* output) {
    int result = stbir_resize_uint8_generic(
        input.pixels, input.extent.width, input.extent.height, input.stride, output, target.width,
        target.height, /*output_stride*/ 0, /*num_channels*/ 1, STBIR_ALPHA_CHANNEL_NONE, 0,
        STBIR_EDGE_CLAMP, STBIR_FILTER_BOX, STBIR_COLORSPACE_LINEAR, nullptr);
    if (result == 0) {
        throw Exception(fmt::format("Failed to resize mask {}x{} to {}x{}", input.extent.width,
                                    input.extent.height, target.width, target.height));
    }
}

void image_to_float(ImageView const& img, std::span<float> dst, int n_channels, float4 mean,
                    float4 std) {

    n_channels = n_channels <= 0 ? count(img.channels) : n_channels;
    // ASSERT(n_channels <= count(img.channels));
    ASSERT(dst.size() == size_t(img.extent.width * img.extent.height * n_channels));
    int channels_to_copy = std::min(n_channels, count(img.channels));

    auto input = PixelAccessor(img);
    for (int y = 0; y < img.extent.height; ++y) {
        for (int x = 0; x < img.extent.width; ++x) {
            for (int c = 0; c < channels_to_copy; ++c) {
                float value = float(input.get(img.pixels, x, y, c));
                float normalized = (value - mean[c]) / std[c];
                dst[y * img.extent.width * n_channels + x * n_channels + c] = normalized;
            }
        }
    }
}

void image_from_float(std::span<float const> src, std::span<uint8_t> dst, float scale,
                      float offset) {
    ASSERT(src.size() == dst.size());
    for (size_t i = 0; i < src.size(); ++i) {
        float value = 255.0f * std::clamp(src[i] * scale + offset, 0.0f, 1.0f);
        dst[i] = uint8_t(value);
    }
}

template <typename T>
void blur_impl(std::span<T> src, std::span<T> dst, Extent extent, int radius) {
    ASSERT(src.size() == dst.size());
    ASSERT(src.size() == size_t(extent.width * extent.height));
    ASSERT(radius > 0);
    ASSERT(radius <= extent.width / 2 && radius <= extent.height / 2);

    std::vector<T> temp(src.size());
    float weight = 1.0f / (2 * radius + 1);

    // Horizontal pass (src -> temp)
    for (int y = 0; y < extent.height; ++y) {
        int row_offset = y * extent.width;
        T sum = radius * src[row_offset];
        for (int x = 0; x <= radius; ++x) {
            sum = sum + src[row_offset + x];
        }

        temp[row_offset] = sum * weight;

        for (int x = 1; x < extent.width; ++x) {
            int left_x = std::clamp(x - radius - 1, 0, extent.width - 1);
            sum = sum - src[row_offset + left_x];

            int right_x = std::clamp(x + radius, 0, extent.width - 1);
            sum = sum + src[row_offset + right_x];

            temp[row_offset + x] = sum * weight;
        }
    }

    // Vertical pass (temp -> dst)
    for (int x = 0; x < extent.width; ++x) {
        T sum = radius * temp[x];
        for (int y = 0; y <= radius; ++y) {
            sum = sum + temp[y * extent.width + x];
        }

        dst[x] = sum * weight;

        for (int y = 1; y < extent.height; ++y) {
            int top_y = std::clamp(y - radius - 1, 0, extent.height - 1);
            sum = sum - temp[top_y * extent.width + x];

            int bottom_y = std::clamp(y + radius, 0, extent.height - 1);
            sum = sum + temp[bottom_y * extent.width + x];

            dst[y * extent.width + x] = sum * weight;
        }
    }
}

void blur(std::span<float> src, std::span<float> dst, Extent extent, int radius) {
    blur_impl(src, dst, extent, radius);
}
void blur(std::span<float4> src, std::span<float4> dst, Extent extent, int radius) {
    blur_impl(src, dst, extent, radius);
}

// Approximate Fast Foreground Colour Estimation
// https://ieeexplore.ieee.org/document/9506164
auto blur_fusion_foreground_estimator(std::span<float4> img, std::span<float4> fg,
                                      std::span<float4> bg, std::span<float> mask, Extent extent,
                                      int radius) {
    ASSERT(img.size() == fg.size() && img.size() == bg.size() && img.size() == mask.size());
    size_t n = mask.size();

    auto per_pixel = [n](auto&& f) {
        for (size_t i = 0; i < n; ++i) {
            f(i);
        }
    };

    auto blurred_mask = std::vector<float>(n);
    blur(mask, blurred_mask, extent, radius);

    auto fg_masked = std::vector<float4>(n);
    per_pixel([&](size_t i) { fg_masked[i] = fg[i] * mask[i]; });

    auto blurred_fg = std::vector<float4>(n);
    blur(fg_masked, blurred_fg, extent, radius);
    per_pixel([&](size_t i) { blurred_fg[i] = blurred_fg[i] / (blurred_mask[i] + 1e-5f); });

    auto& bg_masked = fg_masked; // Reuse fg_masked for bg
    per_pixel([&](size_t i) { bg_masked[i] = bg[i] * (1.0f - mask[i]); });

    auto blurred_bg = std::vector<float4>(n);
    blur(bg_masked, blurred_bg, extent, radius);
    per_pixel([&](size_t i) {
        blurred_bg[i] = blurred_bg[i] / ((1.0f - blurred_mask[i]) + 1e-5f);
        float4 f = blurred_fg[i] +
                   mask[i] * (img[i] - mask[i] * blurred_fg[i] - (1.0f - mask[i]) * blurred_bg[i]);
        blurred_fg[i] = clamp(f, 0.0f, 1.0f);
    });
    return std::pair{blurred_fg, blurred_bg};
}

std::vector<float4> estimate_foreground(std::span<float4> img, std::span<float> mask, Extent extent,
                                        int radius) {
    auto&& [fg, blur_bg] = blur_fusion_foreground_estimator(img, img, img, mask, extent, radius);
    return blur_fusion_foreground_estimator(img, fg, blur_bg, mask, extent, 3).first;
}

} // namespace dlimg
