#pragma once

#include "ml.hpp"

#include <array>
#include <cmath>

namespace dlimg::migan {

struct MIGANParams {
    int resolution = 256;
    bool invert_mask = false;

    static MIGANParams detect(ModelRef m);
};

Tensor generate(ModelRef m, Tensor image, MIGANParams const& p);

std::vector<float> preprocess(ImageView image, ImageView mask, MIGANParams const& p);
Image postprocess(std::span<float> data, Extent extent, MIGANParams const& p);

template <typename E>
struct flags {
    using enum_type = E;

    constexpr flags() = default;
    constexpr flags(E value) : value(uint32_t(value)) {}
    explicit constexpr flags(uint32_t value) : value(value) {}

    flags& operator|=(E other) {
        value |= other;
        return *this;
    }

    uint32_t value = 0;
};

template <typename E>
constexpr bool operator&(flags<E> lhs, E rhs) {
    return (lhs.value & uint32_t(rhs)) != 0;
}

template <typename E>
constexpr flags<E> operator|(flags<E> lhs, E rhs) {
    return flags<E>(lhs.value | uint32_t(rhs));
}

enum class conv {
    none = 0,
    upsample = 1 << 0,
    downsample = 1 << 1,
    noise = 1 << 2,
    activation = 1 << 3,
};

constexpr flags<conv> operator|(conv lhs, conv rhs) {
    return flags<conv>(uint32_t(lhs) | uint32_t(rhs));
}

using Features = std::array<Tensor, 9>;

Tensor lrelu_agc(ModelRef m, Tensor x, float alpha = 0.2f, float gain = 1, float clamp = 0);
Tensor downsample_2d(ModelRef m, Tensor x);
Tensor upsample_2d(ModelRef m, Tensor x);
Tensor separable_conv_2d(ModelRef m, Tensor x, flags<conv> flags = {});
Tensor from_rgb(ModelRef m, Tensor x);

std::pair<Tensor, Tensor> encoder_block(ModelRef m, Tensor x, conv flag = conv::none);
std::pair<Tensor, Features> encode(ModelRef m, Tensor x, int res);

std::pair<Tensor, Tensor> synthesis_block(ModelRef m, Tensor x, Tensor feat, Tensor img,
                                          conv up_flag = conv::none, conv noise_flag = conv::none);
Tensor synthesis(ModelRef m, Tensor x_in, Features feats, int res);

} // namespace dlimg::migan