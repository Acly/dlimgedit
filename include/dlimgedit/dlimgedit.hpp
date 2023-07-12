#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace dlimgedit {
class EnvironmentImpl;
class Image;
struct ImageView;

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
    ImageView(Extent, std::span<uint8_t const> pixels, Channels = Channels::rgba);
    ImageView(Image const&);
};

enum class Device { cpu, gpu };

struct Options {
    Device device = Device::cpu;
    std::string_view model_path = "models";
};

class Environment {
  public:
    explicit Environment(Options const& = {});
    ~Environment();

    static Environment& global();

    EnvironmentImpl& impl();

  private:
    std::unique_ptr<EnvironmentImpl> m_;
};

// Segmentation

class Segmentation {
  public:
    static Segmentation process(ImageView const& img, Environment& = Environment::global());

    Image get_mask(Point) const;
    Image get_mask(Region) const;

    ~Segmentation();
    Segmentation(Segmentation&&);
    Segmentation& operator=(Segmentation&&);

  private:
    struct Impl;
    Segmentation(std::unique_ptr<Impl>&&);
    std::unique_ptr<Impl> m_;
};

// Upscaling

enum class Upscaler { esrgan };

Image upscale(ImageView const&, Extent target, Upscaler, Environment& env = Environment::global());

// Error handling

class Exception : public std::exception {
  public:
    explicit Exception(std::string_view msg) : msg_(msg) {}
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
    std::span<uint8_t> pixels() { return {pixels_.get(), size()}; }
    std::span<uint8_t const> pixels() const { return {pixels_.get(), size()}; }
    size_t size() const { return extent_.width * extent_.height * static_cast<int>(channels_); }

    static Image load(std::string_view filepath);
    static void save(ImageView const& img, std::string_view filepath);

  private:
    Image(Extent, Channels, std::unique_ptr<uint8_t[]> pixels);

    Extent extent_;
    Channels channels_;
    std::unique_ptr<uint8_t[]> pixels_;
};

} // namespace dlimgedit