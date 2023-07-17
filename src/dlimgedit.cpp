#include "environment.hpp"
#include "image.hpp"
#include "segmentation.hpp"
#include <dlimgedit/detail/dlimgedit.h>
#include <dlimgedit/detail/handle.hpp>
#include <dlimgedit/dlimgedit.hpp>

namespace dlimgedit {
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
    }
    return dlimg_success;
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
                                   uint8_t* result_mask) {
    return try_([=] {
        to_impl(handle).get_mask(reinterpret_cast<Point const*>(point),
                                 reinterpret_cast<Region const*>(region), result_mask);
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

} // namespace dlimgedit

extern "C" {

dlimg_Api const* dlimg_init() {
    dlimgedit::api_.create_environment = dlimgedit::create_environment;
    dlimgedit::api_.destroy_environment = dlimgedit::destroy_environment;
    dlimgedit::api_.process_image_for_segmentation = dlimgedit::process_image_for_segmentation;
    dlimgedit::api_.get_segmentation_mask = dlimgedit::get_segmentation_mask;
    dlimgedit::api_.get_segmentation_extent = dlimgedit::get_segmentation_extent;
    dlimgedit::api_.destroy_segmentation = dlimgedit::destroy_segmentation;
    dlimgedit::api_.load_image = dlimgedit::load_image_api;
    dlimgedit::api_.save_image = dlimgedit::save_image_api;
    dlimgedit::api_.create_image = dlimgedit::create_image;
    dlimgedit::api_.destroy_image = dlimgedit::destroy_image;
    dlimgedit::api_.last_error = dlimgedit::last_error;
    return &dlimgedit::api_;
}

} // extern "C"
