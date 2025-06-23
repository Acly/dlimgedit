#pragma once

#include "assert.hpp"
#include <dlimgedit/dlimgedit.hpp>

#include <span>

namespace dlimg {

struct alignas(16) float4 {
    std::array<float, 4> v;

    static constexpr size_t size() { return 4; }

    constexpr float4() = default;
    explicit constexpr float4(float x, float y, float z, float w) : v{x, y, z, w} {}
    explicit constexpr float4(float v) : v{v, v, v, v} {}
    explicit constexpr float4(std::array<float, 4> const& arr) : v(arr) {}

    float& operator[](size_t i) { return v[i]; }
    float operator[](size_t i) const { return v[i]; }
};

constexpr float4 operator-(float4 const& a) { return float4{-a[0], -a[1], -a[2], -a[3]}; }
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
constexpr float4 operator/(float b, float4 const& a) {
    return float4{b / a[0], b / a[1], b / a[2], b / a[3]};
}
constexpr float4 clamp(float4 const& a, float min, float max) {
    return float4{std::clamp(a[0], min, max), std::clamp(a[1], min, max),
                  std::clamp(a[2], min, max), std::clamp(a[3], min, max)};
}
constexpr bool operator==(float4 const& a, float4 const& b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels);
void save_image(ImageView const& img, char const* filepath);

Image resize(ImageView const&, Extent target);
void resize_mask(ImageView const&, Extent target, uint8_t* output);

struct PixelAccessor {
    int stride_x;
    int stride_c;
    std::array<int, 4> channel_map;

    PixelAccessor(ImageView image) : PixelAccessor(image.extent, image.channels) {}

    PixelAccessor(Extent extent, Channels channels) {
        stride_c = count(channels);
        stride_x = extent.width * stride_c;
        switch (channels) {
        case Channels::bgra:
            channel_map = {2, 1, 0, 3};
            break;
        case Channels::argb:
            channel_map = {1, 2, 3, 0};
            break;
        case Channels::mask:
            channel_map = {0, 0, 0, 0};
            break;
        case Channels::rgb:
            channel_map = {0, 1, 2, 0};
        default: // rgba
            channel_map = {0, 1, 2, 3};
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

using i32x2 = std::array<int32_t, 2>;
using i64x4 = std::array<int64_t, 4>;
using rgba32_t = float4;
using rgb32_t = std::array<float, 3>;

constexpr float4 to_float4(rgb32_t const& v) { return float4{v[0], v[1], v[2], 1.0f}; }
constexpr float4 to_float4(rgba32_t const& v) { return v; }
constexpr float4 to_float4(float v) { return float4{v, v, v, v}; }

constexpr int64_t div_ceil(int64_t a, int64_t b) { return (a + b - 1) / b; }
constexpr int32_t div_ceil(int32_t a, int32_t b) { return (a + b - 1) / b; }

constexpr i32x2 div_ceil(i32x2 a, i32x2 b) { return {div_ceil(a[0], b[0]), div_ceil(a[1], b[1])}; }
constexpr i32x2 div_ceil(i32x2 a, int32_t b) { return div_ceil(a, i32x2{b, b}); }
constexpr i32x2 operator+(i32x2 a, i32x2 b) { return {a[0] + b[0], a[1] + b[1]}; }
constexpr i32x2 operator-(i32x2 a, i32x2 b) { return {a[0] - b[0], a[1] - b[1]}; }
constexpr i32x2 operator*(i32x2 a, int32_t b) { return {a[0] * b, a[1] * b}; }
constexpr i32x2 operator*(i32x2 a, i32x2 b) { return {a[0] * b[0], a[1] * b[1]}; }
constexpr i32x2 operator/(i32x2 a, int32_t b) { return {a[0] / b, a[1] / b}; }
constexpr i32x2 min(i32x2 a, i32x2 b) { return {std::min(a[0], b[0]), std::min(a[1], b[1])}; }

template <typename T>
struct image_span {
    using value_type = std::remove_cv_t<T>;

    constexpr static int n_channels = sizeof(T) / sizeof(float);

    Extent extent;
    size_t row_stride;
    T* data;

    image_span(Extent extent, T* data) : extent(extent), row_stride(extent.width), data(data) {}

    image_span(Extent extent, std::span<float> data)
        : image_span(extent, reinterpret_cast<T*>(data.data())) {
        ASSERT(data.size() == size_t(extent.width * extent.height * n_channels));
    }
    image_span(image_span<value_type> const& other)
        : extent(other.extent), row_stride(other.row_stride), data(other.data) {}

    float4 get(int x, int y) const { return to_float4(data[y * row_stride + x]); }

    void set(int x, int y, float4 value) {
        if constexpr (std::is_same_v<value_type, rgba32_t>) {
            data[y * row_stride + x] = value;
        } else if constexpr (std::is_same_v<value_type, rgb32_t>) {
            data[y * row_stride + x] = {value[0], value[1], value[2]};
        } else {
            static_assert(std::is_same_v<value_type, float>, "Unsupported type for image_span");
            data[y * row_stride + x] = value[0];
        }
    }

    float4 get(i32x2 coord) const { return get(coord[0], coord[1]); }
    void set(i32x2 coord, float4 value) { set(coord[0], coord[1], value); }
    float4 operator[](size_t i) const { return to_float4(data[i]); }

    int n_elements() const { return extent.width * extent.height; }

    std::span<float> as_float() {
        return std::span(reinterpret_cast<float*>(data), n_elements() * n_channels);
    }
};

template <typename T>
void image_to_float(ImageView const& src, image_span<T> dst, float4 offset = float4(0),
                    float4 scale = float4(1), i32x2 tile_offset = {0, 0});

void image_from_float(std::span<float const> src, std::span<uint8_t> dst, float scale = 1,
                      float offset = 0);

void blur(std::span<float> src, std::span<float> dst, Extent extent, int radius);
void blur(std::span<float4> src, std::span<float4> dst, Extent extent, int radius);

std::vector<float4> estimate_foreground(std::span<float4> img, std::span<float> mask, Extent extent,
                                        int radius = 30);

void alpha_composite(ImageView const& fg, ImageView const& bg, ImageView const& mask, uint8_t* dst);

struct tile_layout {
    i32x2 image_extent;
    i32x2 overlap;
    i32x2 n_tiles;
    i32x2 tile_size;

    tile_layout() = default;

    tile_layout(Extent extent, int max_tile_size, int overlap, int align = 16)
        : image_extent{extent.width, extent.height}, overlap{overlap, overlap} {

        n_tiles = div_ceil(image_extent, max_tile_size);
        i32x2 img_extent_overlap = image_extent + (n_tiles - i32x2{1, 1}) * overlap;
        tile_size = div_ceil(img_extent_overlap, n_tiles);
        tile_size = div_ceil(tile_size, align) * align;
    }

    static tile_layout scale(tile_layout const& o, int scale) {
        tile_layout scaled;
        scaled.image_extent = o.image_extent * scale;
        scaled.overlap = o.overlap * scale;
        scaled.n_tiles = o.n_tiles;
        scaled.tile_size = o.tile_size * scale;
        return scaled;
    }

    i32x2 start(i32x2 coord, i32x2 pad = {0, 0}) const {
        i32x2 offset = coord * (tile_size - overlap);
        return offset + i32x2{coord[0] == 0 ? 0 : pad[0], coord[1] == 0 ? 0 : pad[1]};
    }

    i32x2 end(i32x2 coord, i32x2 pad = {0, 0}) const {
        i32x2 offset = start(coord) + tile_size;
        offset = offset - i32x2{coord[0] == n_tiles[0] - 1 ? 0 : pad[0],
                                coord[1] == n_tiles[1] - 1 ? 0 : pad[1]};
        return min(offset, image_extent);
    }

    i32x2 size(i32x2 coord) const { return end(coord) - start(coord); }

    int total() const { return n_tiles[0] * n_tiles[1]; }

    i32x2 coord(int index) const {
        ASSERT(index >= 0 && index < total());
        return {index % n_tiles[0], index / n_tiles[0]};
    }

    Extent tile_extent() const { return {tile_size[0], tile_size[1]}; }
};

void merge_tile(image_span<const rgb32_t> tile, image_span<rgb32_t> dst, i32x2 tile_coord,
                tile_layout const& layout);

} // namespace dlimg