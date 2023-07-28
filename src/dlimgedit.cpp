#include "environment.hpp"
#include "image.hpp"
#include "segmentation.hpp"
#include <dlimgedit/detail/dlimgedit.h>
#include <dlimgedit/detail/handle.hpp>
#include <dlimgedit/dlimgedit.hpp>

namespace dlimg {
using detail::bit_cast;

dlimg_Api api_;
std::string last_error_;

static_assert(sizeof(dlimg_Options) == sizeof(Options));
static_assert(sizeof(dlimg_ImageView) == sizeof(ImageView));

dlimg_Environment to_handle(EnvironmentImpl const* env) { return bit_cast<dlimg_Environment>(env); }
dlimg_Segmentation to_handle(SegmentationImpl const* seg) {
    return bit_cast<dlimg_Segmentation>(seg);
}

EnvironmentImpl& to_impl(dlimg_Environment env) { return *bit_cast<EnvironmentImpl*>(env); }
SegmentationImpl& to_impl(dlimg_Segmentation seg) { return *bit_cast<SegmentationImpl*>(seg); }
ImageView const& to_impl(dlimg_ImageView const* img) { return *bit_cast<ImageView const*>(img); }

dlimg_Result set_last_error(std::exception const& e) {
    last_error_ = e.what();
    return dlimg_error;
}

template <typename F> dlimg_Result try_(F const& f) {
    try {
        f();
    } catch (std::exception const& e) {
        return set_last_error(e);
    } catch (...) {
        return set_last_error(std::runtime_error("Unknown error"));
    }
    return dlimg_success;
}

int is_backend_supported(dlimg_Backend backend) {
    return EnvironmentImpl::is_supported(Backend(backend)) ? 1 : 0;
}

dlimg_Result create_environment(dlimg_Environment* handle, dlimg_Options const* options) {
    return try_([=] { *handle = to_handle(new EnvironmentImpl(bit_cast<Options>(*options))); });
}

void destroy_environment(dlimg_Environment handle) { delete &to_impl(handle); }

dlimg_Result process_image_for_segmentation(dlimg_Segmentation* handle, dlimg_ImageView const* img,
                                            dlimg_Environment env) {
    return try_([=] {
        auto seg = new SegmentationImpl(to_impl(env));
        *handle = to_handle(seg);
        seg->process(to_impl(img));
    });
}

dlimg_Result get_segmentation_mask(dlimg_Segmentation handle, int const* point, int const* region,
                                   uint8_t** result_masks, float* result_accuracy) {
    return try_([=] {
        to_impl(handle).compute_mask(
            reinterpret_cast<Point const*>(point), reinterpret_cast<Region const*>(region),
            std::span<uint8_t*, 3>(result_masks, 3), std::span<float, 3>(result_accuracy, 3));
    });
}

void get_segmentation_extent(dlimg_Segmentation handle, int* out_extent) {
    out_extent[0] = to_impl(handle).extent().width;
    out_extent[1] = to_impl(handle).extent().height;
}

void destroy_segmentation(dlimg_Segmentation handle) { delete &to_impl(handle); }

dlimg_Result load_image_api(char const* filepath, int* out_extent, int* out_channels,
                            uint8_t** out_pixels) {
    return try_([=] {
        *out_pixels = load_image(filepath, reinterpret_cast<Extent*>(out_extent), out_channels);
    });
}

dlimg_Result save_image_api(dlimg_ImageView const* img, char const* filepath) {
    return try_([=] { save_image(to_impl(img), filepath); });
}

uint8_t* create_image(int w, int h, int channels) { return new uint8_t[w * h * channels]; }

void destroy_image(uint8_t const* pixels) { delete[] pixels; }

char const* last_error() { return last_error_.c_str(); }

} // namespace dlimg

extern "C" {

dlimg_Api const* dlimg_init() {
    dlimg::api_.is_backend_supported = dlimg::is_backend_supported;
    dlimg::api_.create_environment = dlimg::create_environment;
    dlimg::api_.destroy_environment = dlimg::destroy_environment;
    dlimg::api_.process_image_for_segmentation = dlimg::process_image_for_segmentation;
    dlimg::api_.get_segmentation_mask = dlimg::get_segmentation_mask;
    dlimg::api_.get_segmentation_extent = dlimg::get_segmentation_extent;
    dlimg::api_.destroy_segmentation = dlimg::destroy_segmentation;
    dlimg::api_.load_image = dlimg::load_image_api;
    dlimg::api_.save_image = dlimg::save_image_api;
    dlimg::api_.create_image = dlimg::create_image;
    dlimg::api_.destroy_image = dlimg::destroy_image;
    dlimg::api_.last_error = dlimg::last_error;
    return &dlimg::api_;
}

} // extern "C"
