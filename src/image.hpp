#pragma once

#include <dlimgedit/dlimgedit.hpp>

namespace dlimg {

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels);
void save_image(ImageView const& img, char const* filepath);

Image resize(ImageView const&, Extent target);

} // namespace dlimg