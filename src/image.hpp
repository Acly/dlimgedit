#pragma once

#include <dlimgedit/dlimgedit.hpp>

namespace dlimg {

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels);
void save_image(ImageView const& img, char const* filepath);

Image resize(ImageView const&, Extent target);
void resize_mask(ImageView const&, Extent target, uint8_t* output);

} // namespace dlimg