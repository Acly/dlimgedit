#pragma once

#include <dlimgedit/detail/dlimgedit.h>
#include <dlimgedit/detail/handle.hpp>

#include <array>
#include <memory>
#include <string>
#ifndef DLIMGEDIT_NO_FILESYSTEM
#    include <filesystem>
#endif

namespace dlimg {
class Image;

// Image

struct Extent {
    int width = 0;
    int height = 0;
};

enum class Channels { mask = 1, rgb = 3, rgba = 4, bgra, argb };
constexpr int count(Channels);

struct ImageView {
    Extent extent;
    Channels channels = Channels::rgba;
    int stride = 0;
    uint8_t const* pixels = nullptr;

    ImageView() = default;
    ImageView(uint8_t const* pixels, Extent, Channels = Channels::rgba);
    ImageView(Image const&);
};

class Image {
  public:
    explicit Image(Extent, Channels = Channels::rgba);

    Extent extent() const { return extent_; }
    Channels channels() const { return channels_; }
    uint8_t* pixels() { return pixels_; }
    uint8_t const* pixels() const { return pixels_; }
    size_t size() const { return extent_.width * extent_.height * count(channels_); }

    static Image load(char const* filepath);
    static void save(ImageView const& img, char const* filepath);

#ifndef DLIMGEDIT_NO_FILESYSTEM
    static Image load(std::filesystem::path const& filepath);
    static void save(ImageView const& img, std::filesystem::path const& filepath);
#endif

    ~Image();
    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;

  private:
    Image(Extent, Channels, uint8_t*);

    Extent extent_;
    Channels channels_;
    uint8_t* pixels_ = nullptr;
};

// Environment

void initialize(dlimg_Api const* = dlimg_init());

enum class Backend { cpu, gpu };

struct Options {
    Backend backend = Backend::cpu;
    char const* model_path = "models";
};

class Environment : public Handle<dlimg_Environment_> {
  public:
    static bool is_supported(Backend);

    explicit Environment(Options const& = {});

    Environment(std::nullptr_t) noexcept;
};

// Segmentation

struct Point {
    int x = 0;
    int y = 0;
};

struct Region {
    Point top_left;
    Point bottom_right;

    Region() = default;
    Region(Point top_left, Point bottom_right);
    Region(Point origin, Extent extent);
};

class Segmentation : public Handle<dlimg_Segmentation_> {
  public:
    struct Mask {
        Image image;
        float accuracy = 0.0f;
    };

    static Segmentation process(ImageView const& img, Environment const&);

    Image get_mask(Point) const;
    void get_mask(Point, uint8_t* result_mask) const;

    std::array<Mask, 3> get_masks(Point) const;

    Image get_mask(Region) const;
    void get_mask(Region, uint8_t* result_mask) const;

    Extent extent() const;

    Segmentation(std::nullptr_t) noexcept;
};

// Error handling

class Exception : public std::exception {
  public:
    explicit Exception(std::string msg) : msg_(std::move(msg)) {}
    char const* what() const noexcept override { return msg_.c_str(); }

  private:
    std::string msg_;
};

//
// Implementation

inline void initialize(dlimg_Api const* api) { detail::Global<void>::api_ = api; }

inline void throw_on_error(dlimg_Result result) {
    if (result == dlimg_error) {
        throw Exception(api().last_error());
    }
}

// ImageView

constexpr int count(Channels c) { return int(c) > 4 ? 4 : int(c); }

inline ImageView::ImageView(uint8_t const* pixels, Extent extent, Channels channels)
    : extent(extent), channels(channels), stride(extent.width * count(channels)), pixels(pixels) {}

inline ImageView::ImageView(Image const& img)
    : extent(img.extent()),
      channels(img.channels()),
      stride(extent.width * count(channels)),
      pixels(img.pixels()) {}

inline dlimg_ImageView const* to_api(ImageView const& i) {
    return reinterpret_cast<dlimg_ImageView const*>(&i);
}

// Environment

inline bool Environment::is_supported(Backend d) {
    return api().is_backend_supported(dlimg_Backend(int(d))) != 0;
}

inline dlimg_Options const* to_api(Options const& o) {
    return reinterpret_cast<dlimg_Options const*>(&o);
}

inline Environment::Environment(Options const& options) {
    throw_on_error(api().create_environment(&emplace(), to_api(options)));
}

inline Environment::Environment(std::nullptr_t) noexcept {}

// Point & Region

inline bool operator==(Point a, Point b) { return a.x == b.x && a.y == b.y; }
inline bool operator!=(Point a, Point b) { return !(a == b); }

inline Region::Region(Point top_left, Point bottom_right)
    : top_left(top_left), bottom_right(bottom_right) {}

inline Region::Region(Point origin, Extent extent)
    : top_left(origin), bottom_right(Point{origin.x + extent.width, origin.y + extent.height}) {}

inline bool operator==(Region a, Region b) {
    return a.top_left == b.top_left && a.bottom_right == b.bottom_right;
}
inline bool operator!=(Region a, Region b) { return !(a == b); }

// Segmentation

inline Segmentation::Segmentation(std::nullptr_t) noexcept {}

inline Segmentation Segmentation::process(ImageView const& img, Environment const& env) {
    auto result = Segmentation(nullptr);
    throw_on_error(
        api().process_image_for_segmentation(&result.emplace(), to_api(img), env.handle()));
    return result;
}

inline void Segmentation::get_mask(Point point, uint8_t* result_mask) const {
    auto masks = std::array<uint8_t*, 3>{result_mask, nullptr, nullptr};
    auto ious = std::array<float, 3>{0.0f, 0.0f, 0.0f};
    throw_on_error(
        api().get_segmentation_mask(handle(), &point.x, nullptr, masks.data(), ious.data()));
}

inline Image Segmentation::get_mask(Point point) const {
    auto result = Image(extent(), Channels::mask);
    get_mask(point, result.pixels());
    return result;
}

inline std::array<Segmentation::Mask, 3> Segmentation::get_masks(Point point) const {
    auto result = std::array<Mask, 3>{Mask{Image(extent(), Channels::mask), 0.f},
                                      Mask{Image(extent(), Channels::mask), 0.f},
                                      Mask{Image(extent(), Channels::mask), 0.f}};
    auto masks = std::array<uint8_t*, 3>{result[0].image.pixels(), result[1].image.pixels(),
                                         result[2].image.pixels()};
    auto ious = std::array<float, 3>{0.0f, 0.0f, 0.0f};
    throw_on_error(
        api().get_segmentation_mask(handle(), &point.x, nullptr, masks.data(), ious.data()));
    result[0].accuracy = ious[0];
    result[1].accuracy = ious[1];
    result[2].accuracy = ious[2];
    return result;
}

inline void Segmentation::get_mask(Region region, uint8_t* result_mask) const {
    auto masks = std::array<uint8_t*, 3>{result_mask, nullptr, nullptr};
    auto ious = std::array<float, 3>{0.0f, 0.0f, 0.0f};
    throw_on_error(api().get_segmentation_mask(handle(), nullptr, &region.top_left.x, masks.data(),
                                               ious.data()));
}

inline Image Segmentation::get_mask(Region region) const {
    auto result = Image(extent(), Channels::mask);
    get_mask(region, result.pixels());
    return result;
}

inline Extent Segmentation::extent() const {
    Extent result;
    api().get_segmentation_extent(handle(), &result.width);
    return result;
}

// Image

inline Image::Image(Extent extent, Channels channels)
    : extent_(extent),
      channels_(channels),
      pixels_(api().create_image(extent.width, extent.height, count(channels))) {}

inline Image::Image(Extent extent, Channels channels, uint8_t* pixels)
    : extent_(extent), channels_(channels), pixels_(pixels) {}

inline Image::~Image() { api().destroy_image(pixels_); }

inline Image::Image(Image&& other) noexcept : Image(other.extent_, other.channels_, other.pixels_) {
    other.pixels_ = nullptr;
}

inline Image& Image::operator=(Image&& other) noexcept {
    std::swap(extent_, other.extent_);
    std::swap(channels_, other.channels_);
    std::swap(pixels_, other.pixels_);
    return *this;
}

inline Image Image::load(char const* filepath) {
    uint8_t* pixels = nullptr;
    Extent extent;
    int channels = 0;
    throw_on_error(api().load_image(filepath, &extent.width, &channels, &pixels));
    return Image(extent, static_cast<Channels>(channels), pixels);
}

inline void Image::save(ImageView const& img, char const* filepath) {
    throw_on_error(api().save_image(to_api(img), filepath));
}

#ifndef DLIMGEDIT_NO_FILESYSTEM

inline Image Image::load(std::filesystem::path const& filepath) {
    return load(filepath.string().c_str());
}

inline void Image::save(ImageView const& img, std::filesystem::path const& filepath) {
    save(img, filepath.string().c_str());
}

#endif

constexpr bool operator==(Extent a, Extent b) { return a.width == b.width && a.height == b.height; }
constexpr bool operator!=(Extent a, Extent b) { return !(a == b); }

} // namespace dlimg