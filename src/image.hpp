#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include <span>

namespace dlimg {

struct alignas(16) float4 {
    std::array<float, 4> v;

    constexpr float4() = default;
    explicit constexpr float4(float x, float y, float z, float w) : v{x, y, z, w} {}
    explicit constexpr float4(float v) : v{v, v, v, v} {}
    explicit constexpr float4(std::array<float, 4> const& arr) : v(arr) {}

    float& operator[](size_t i) { return v[i]; }
    float operator[](size_t i) const { return v[i]; }
};

constexpr float4 operator+(float4 const& a, float4 const& b) {
    return float4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}
constexpr float4 operator-(float4 const& a, float4 const& b) {
    return float4{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}
constexpr float4 operator*(float4 const& a, float4 const& b) {
    return float4{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
}
constexpr float4 operator/(float4 const& a, float4 const& b) {
    return float4{a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]};
}
constexpr float4 operator*(float4 const& a, float b) {
    return float4{a[0] * b, a[1] * b, a[2] * b, a[3] * b};
}
constexpr float4 operator*(float b, float4 const& a) {
    return float4{b * a[0], b * a[1], b * a[2], b * a[3]};
}
constexpr float4 operator/(float4 const& a, float b) {
    return float4{a[0] / b, a[1] / b, a[2] / b, a[3] / b};
}
constexpr float4 clamp(float4 const& a, float min, float max) {
    return float4{std::clamp(a[0], min, max), std::clamp(a[1], min, max),
                  std::clamp(a[2], min, max), std::clamp(a[3], min, max)};
}

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

void image_to_float(ImageView const& img, std::span<float> dst, int n_channels = -1,
                    float4 mean = float4(0.0f, 0.0f, 0.0f, 0.0f),
                    float4 std = float4(255.0f, 255.0f, 255.0f, 255.0f));

void image_from_float(std::span<float const> src, std::span<uint8_t> dst, float scale = 1,
                      float offset = 0);

void blur(std::span<float> src, std::span<float> dst, Extent extent, int radius);
void blur(std::span<float4> src, std::span<float4> dst, Extent extent, int radius);

std::vector<float4> estimate_foreground(std::span<float4> img, std::span<float> mask, Extent extent,
                                        int radius = 30);

} // namespace dlimg