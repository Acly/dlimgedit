#pragma once

#include <dlimgedit/detail/dlimgedit.h>
#include <dlimgedit/detail/handle.hpp>

#include <memory>
#include <string>
#ifndef DLIMGEDIT_NO_FILESYSTEM
#    include <filesystem>
#endif

namespace dlimgedit {
class Image;

struct Point {
    int x = 0;
    int y = 0;
};

struct Extent {
    int width = 0;
    int height = 0;
};

struct Region {
    Point origin;
    Extent extent;
};

enum class Channels { mask = 1, rgb = 3, rgba = 4 };

struct ImageView {
    Extent extent;
    Channels channels = Channels::rgba;
    int stride = 0;
    uint8_t const* pixels = nullptr;

    ImageView() = default;
    ImageView(Extent, uint8_t const* pixels, Channels = Channels::rgba);
    ImageView(Image const&);
};

enum class Device { cpu, gpu };

struct Options {
    Device device = Device::cpu;
    char const* model_path = "models";
};

void initialize(dlimg_Api const* = dlimg_init());

class Environment : public Handle<dlimg_Environment_> {
  public:
    explicit Environment(Options const& = {});
};

// Segmentation

class Segmentation : public Handle<dlimg_Segmentation_> {
  public:
    static Segmentation process(ImageView const& img, Environment&);

    Image get_mask(Point) const;
    void get_mask(Point, uint8_t* result_mask) const;

    Image get_mask(Region) const;
    void get_mask(Region, uint8_t* result_mask) const;

    Extent extent() const;

  private:
    Segmentation();
};

// Error handling

class Exception : public std::exception {
  public:
    explicit Exception(std::string msg) : msg_(std::move(msg)) {}
    char const* what() const noexcept override { return msg_.c_str(); }

  private:
    std::string msg_;
};

// Tools

class Image {
  public:
    explicit Image(Extent, Channels = Channels::rgba);

    Extent extent() const { return extent_; }
    Channels channels() const { return channels_; }
    uint8_t* pixels() { return pixels_; }
    uint8_t const* pixels() const { return pixels_; }
    size_t size() const { return extent_.width * extent_.height * static_cast<int>(channels_); }

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

//
// Implementation

inline void initialize(dlimg_Api const* api) { detail::Global<void>::api_ = api; }

inline void throw_on_error(dlimg_Result result) {
    if (result == dlimg_error) {
        throw Exception(api().last_error());
    }
}

// ImageView

inline ImageView::ImageView(Extent extent, uint8_t const* pixels, Channels channels)
    : extent(extent),
      channels(channels),
      stride(extent.width * static_cast<int>(channels)),
      pixels(pixels) {}

inline ImageView::ImageView(Image const& img)
    : extent(img.extent()),
      channels(img.channels()),
      stride(extent.width * static_cast<int>(channels)),
      pixels(img.pixels()) {}

inline dlimg_ImageView const* to_api(ImageView const& i) {
    return reinterpret_cast<dlimg_ImageView const*>(&i);
}

// Environment

inline dlimg_Options const* to_api(Options const& o) {
    return reinterpret_cast<dlimg_Options const*>(&o);
}

inline Environment::Environment(Options const& options) {
    throw_on_error(api().create_environment(&emplace(), to_api(options)));
}

// Segmentation

inline Segmentation::Segmentation() {}

inline Segmentation Segmentation::process(ImageView const& img, Environment& env) {
    auto result = Segmentation();
    throw_on_error(
        api().process_image_for_segmentation(&result.emplace(), to_api(img), env.handle()));
    return result;
}

inline void Segmentation::get_mask(Point point, uint8_t* result_mask) const {
    throw_on_error(api().get_segmentation_mask(handle(), &point.x, nullptr, result_mask));
}

inline Image Segmentation::get_mask(Point point) const {
    auto result = Image(extent(), Channels::mask);
    get_mask(point, result.pixels());
    return result;
}

inline void Segmentation::get_mask(Region region, uint8_t* result_mask) const {
    throw_on_error(api().get_segmentation_mask(handle(), nullptr, &region.origin.x, result_mask));
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
      pixels_(api().create_image(extent.width, extent.height, static_cast<int>(channels))) {}

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

} // namespace dlimgedit