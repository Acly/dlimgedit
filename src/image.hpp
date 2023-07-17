#pragma once

#include <dlimgedit/dlimgedit.hpp>

namespace dlimgedit {

uint8_t* load_image(char const* filepath, Extent* out_extent, int* out_channels);
void save_image(ImageView const& img, char const* filepath);

} // namespace dlimgedit