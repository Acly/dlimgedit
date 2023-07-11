#pragma once

#include <dlimgedit/dlimgedit.hpp>

#include <filesystem>

namespace dlimgedit {
using Path = std::filesystem::path;

Path const& root_dir();
Path const& model_dir();
Path const& test_dir();

Environment default_env();

} // namespace dlimgedit
