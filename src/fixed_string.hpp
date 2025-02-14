#pragma once

#include <fmt/format.h>

#include <array>
#include <string_view>

namespace dlimg {

template <size_t N>
struct FixedString {
    std::array<char, N> data{};
    size_t length = 0;

    constexpr FixedString() {}

    FixedString(char const* str) {
        auto view = std::string_view(str);
        length = std::min(view.size(), N - 1);
        std::copy(view.begin(), view.begin() + length, data.begin());
    }

    template <typename... Args>
    FixedString(char const* fmt, Args&&... args) {
        format(fmt, std::forward<Args>(args)...);
    }

    char const* c_str() const { return data.data(); }

    std::string_view view() const { return {data.data(), length}; }

    template <typename... Args>
    char const* format(char const* fmt, Args&&... args) {
        fmt::vformat_to_n(data.data(), N, fmt, fmt::make_format_args(args...));
        data[N - 1] = 0;
        length = strlen(data.data());
        return data.data();
    }

    explicit operator bool() const { return length > 0; }
};

} // namespace dlimg