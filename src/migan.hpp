#pragma once

#include "ml.hpp"
#include "mobile_sam.hpp"

namespace dlimg::migan {
using sam::conv_2d;
using sam::depthwise_conv_2d;

constexpr float sqrt2 = 1.4142135623f;

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

Tensor lrelu_agc(ModelRef m, Tensor x, float alpha = 0.2f, float gain = 1, float clamp = 0) {
    x = ggml_leaky_relu(m, x, alpha, true);
    if (gain != 1) {
        x = ggml_scale_inplace(m, x, gain);
    }
    if (clamp != 0) {
        x = ggml_clamp(m, x, -clamp, clamp);
    }
    return m.named(x);
}

Tensor downsample_2d(ModelRef m, Tensor x) { return depthwise_conv_2d(m["filter"], x, 2, 1); }

Tensor upsample_2d(ModelRef m, Tensor x) {
    Tensor filter_const = m.weights("filter_const");
    filter_const = ggml_reshape_4d(m, filter_const, 1, filter_const->ne[0], filter_const->ne[1], 1);

    x = ggml_upscale_ext(
        m, x, x->ne[0], x->ne[1] * 2, x->ne[2] * 2, x->ne[3], GGML_SCALE_MODE_NEAREST);
    x = ggml_mul_inplace(m, x, filter_const);
    x = depthwise_conv_2d(m["filter"], x, 1, 2);
    x = ggml_view_4d(
        m, x, x->ne[0], x->ne[1] - 1, x->ne[2] - 1, x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
    return m.named(x);
}

enum class conv {
    upsample = 1 << 0,
    downsample = 1 << 1,
    noise = 1 << 2,
    activation = 1 << 3,
};

constexpr flags<conv> operator|(conv lhs, conv rhs) {
    return flags<conv>(uint32_t(lhs) | uint32_t(rhs));
}

Tensor separable_conv_2d(ModelRef m, Tensor x, flags<conv> flags = {}) {
    int pad = m["conv1"].weights("weight")->ne[2] / 2;
    x = depthwise_conv_2d(m["conv1"], x, 1, pad);
    if (flags & conv::activation) {
        x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    }

    if (flags & conv::downsample) {
        x = downsample_2d(m["downsample"], x);
    }
    x = conv_2d(m["conv2"], x);
    if (flags & conv::upsample) {
        x = upsample_2d(m["upsample"], x);
    }

    if (flags & conv::noise) {
        Tensor noise = m.weights("noise_const");
        noise = ggml_mul_inplace(m, noise, m.weights("noise_strength"));
        noise = ggml_reshape_4d(m, noise, 1, noise->ne[0], noise->ne[1], 1);
        x = ggml_add_inplace(m, x, noise);
    }
    if (flags & conv::activation) {
        x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    }
    return m.named(x);
}

} // namespace dlimg::migan