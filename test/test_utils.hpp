#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include <filesystem>

namespace dlimg {
using Path = std::filesystem::path;

Path const& root_dir();
Path const& model_dir();
Path const& test_dir();

Environment default_env();

float rmse(ImageView const&, ImageView const&);
void check_image_matches(ImageView const&, std::string_view reference, float max_rmse = 0.002f);

} // namespace dlimg
