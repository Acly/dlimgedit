// dlimgedit.hpp
// C++ library for image painting and editing workflows which make use of deep learning

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

//
// Image handling

// Resolution of an image or size of an image region.
struct Extent {
    int width = 0;
    int height = 0;
};

// Channel order of image pixels. Each channel is 1 byte (uint8).
enum class Channels { mask = 1, rgb = 3, rgba = 4, bgra, argb };

// Returns number of channels for a pixel.
constexpr int count(Channels);

// Read-only view of an image. Does not own the pixel data.
// Pixel data is expected to be row-major, with the origin in the top left corner.
struct ImageView {
    Extent extent;
    Channels channels = Channels::rgba;
    int stride = 0; // size of one row of pixels, in bytes
    uint8_t const* pixels = nullptr;

    ImageView() noexcept = default;
    ImageView(uint8_t const* pixels, Extent, Channels = Channels::rgba) noexcept;
    ImageView(Image const&) noexcept;
};

// Represents an image made up of packed pixel data.
class Image {
  public:
    // Allocate memory for a new image. Pixel data is uninitialized.
    explicit Image(Extent, Channels = Channels::rgba);

    Extent extent() const noexcept { return extent_; }
    Channels channels() const noexcept { return channels_; }
    uint8_t* pixels() noexcept { return pixels_; }
    uint8_t const* pixels() const noexcept { return pixels_; }
    size_t size() const noexcept; // in bytes

    // Read an image from a file. Supported formats are PNG, JPEG, BMP, TGA.
    static Image load(char const* filepath);

    // Store an image as a PNG file.
    static void save(ImageView const& img, char const* filepath);

#ifndef DLIMGEDIT_NO_FILESYSTEM
    static Image load(std::filesystem::path const& filepath);
    static void save(ImageView const& img, std::filesystem::path const& filepath);
#endif

    ~Image();
    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;
    Image(Image const&) = delete;
    Image& operator=(Image const&) = delete;

  private:
    Image(Extent, Channels, uint8_t*);

    Extent extent_;
    Channels channels_;
    uint8_t* pixels_ = nullptr;
};

//
// Neural network inference environment

// The hardware to use for inference. The GPU backend requires a CUDA-capable NVIDIA graphics card.
enum class Backend { cpu, gpu };

// Deep learning model search paths and inference options.
struct Options {
    Backend backend = Backend::cpu;

    // Path to the directory where models (.onnx files) are stored.
    char const* model_directory = "models";
};

// The environment holds common infrastructure for neural network inference, and caches loaded
// models after they are first used. An instance of this object _must_ outlive all other objects
// which it is passed to.
// Environment objects are safe to use from multiple threads.
class Environment : public Handle<dlimg_Environment_> {
  public:
    // Check if the given backend is supported on the current system. This checks if provider
    // libraries are installed and basic hardware requirements are met. This is a good indicator
    // that features will work with this backend, but not a guarantee.
    static bool is_supported(Backend) noexcept;

    // Initialize common infrastructure. Models are only loaded on demand.
    explicit Environment(Options const& = {});

    Environment(std::nullptr_t) noexcept;
};

//
// Segmentation

// A point in image pixel coordinates, with the origin in the top left corner.
struct Point {
    int x = 0;
    int y = 0;
};

// A rectangular region in image pixel coordinates.
struct Region {
    Point top_left;
    Point bottom_right;

    constexpr Region() = default;
    constexpr Region(Point top_left, Point bottom_right);
    constexpr Region(Point origin, Extent extent);
};

// Stores an image embedding that has been processed for segmentation, and can be used to query
// masks for multiple objects.
class Segmentation : public Handle<dlimg_Segmentation_> {
  public:
    // A binary mask for a single object in the image.
    struct Mask {
        Image image;           // always uses Channels::mask with values 0 or 255
        float accuracy = 0.0f; // confidence value
    };

    // Process a color image for segmentation. This is a comparatively expensive operation.
    // The returned object can be used to cheaply query multiple masks.
    static Segmentation process(ImageView const& img, Environment const&);

    // Compute a mask from a single point in the image. If the result is ambiguous, the best mask
    // (highest accuracy) is returned.
    Image compute_mask(Point) const;
    void compute_mask(Point, uint8_t* result_mask) const;

    // Compute a mask from a single point in the image. Returns multiple masks with varying
    // confidence. This can be useful when the point is ambiguous, e.g. pointing at a chimney may
    // return masks for the chimeny, the roof, and the entire house.
    std::array<Mask, 3> compute_masks(Point) const;

    // Compute a mask for the largest object contained in the given bounding box.
    Image compute_mask(Region) const;
    void compute_mask(Region, uint8_t* result_mask) const;

    // The resolution of the input image.
    Extent extent() const noexcept;

    Segmentation(std::nullptr_t) noexcept;
};

//
// API and error handling

// Initialize the C API. This is called automatically when linking at compile time.
// To load the library dynamically at runtime, define DLIMGEDIT_LOAD_DYNAMIC, resolve
// the symbol for the dlimg_init function, and pass its result to this function.
void initialize(dlimg_Api const* = dlimg_init());

// The exception type thrown by functions in this library (unless marked noexcept).
class Exception : public std::exception {
  public:
    explicit Exception(std::string msg) : msg_(std::move(msg)) {}
    char const* what() const noexcept override { return msg_.c_str(); }

  private:
    std::string msg_;
};

} // namespace dlimg

#include <dlimgedit/detail/dlimgedit.impl.hpp>
