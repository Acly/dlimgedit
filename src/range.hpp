#pragma once

#include <ranges>

namespace dlimg {
namespace ranges = std::ranges;
namespace views = std::views;

// Copies a range into a vector.
template <ranges::range R> constexpr auto to_vector(R&& r) {
    using elem_t = std::decay_t<ranges::range_value_t<R>>;
    return std::vector<elem_t>{r.begin(), r.end()};
}

// Copies a range into an array. Elements must be default-constructible.
template <size_t N, ranges::input_range R> constexpr auto to_array(R&& r) {
    using elem_t = std::decay_t<ranges::range_value_t<R>>;
    auto result = std::array<elem_t, N>{};
    ranges::copy(r, result.begin());
    return result;
}

template <class F, std::size_t... Is>
auto generate_array_impl(F&& f, std::index_sequence<Is...>)
    -> std::array<decltype(f()), sizeof...(Is)> {
    return std::array<decltype(f()), sizeof...(Is)>{{(void(Is), f())...}};
}

// Generates an array of size N by calling f() N times.
template <std::size_t N, class F> auto generate_array(F f) -> std::array<decltype(f()), N> {
    return generate_array_impl(f, std::make_index_sequence<N>());
}

} // namespace dlimg