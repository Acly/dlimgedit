#pragma once

namespace dlimg {

#ifdef _WIN32
constexpr bool is_windows = true;
#else
constexpr bool is_windows = false;
#endif
constexpr bool is_linux = !is_windows;

} // namespace dlimg
