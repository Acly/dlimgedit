#pragma once

namespace dlimg {

#ifdef _WIN32
#    define DLIMG_WINDOWS
constexpr bool is_windows = true;
#else
#    define DLIMG_LINUX
constexpr bool is_windows = false;
#endif
constexpr bool is_linux = !is_windows;

} // namespace dlimg
