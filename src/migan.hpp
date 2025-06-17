#pragma once

#include "ml.hpp"
#include "mobile_sam.hpp"

#include <array>
#include <cmath>

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

template <typename E>
constexpr flags<E> operator|(flags<E> lhs, E rhs) {
    return flags<E>(lhs.value | uint32_t(rhs));
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

    auto [c, w, h, b] = nelements(x);
    x = ggml_upscale_ext(m, x, c, w * 2, h * 2, b, GGML_SCALE_MODE_NEAREST);
    x = ggml_mul_inplace(m, x, filter_const);
    x = depthwise_conv_2d(m["filter"], x, 1, 2); // 4x4 filter
    x = slice(m, x, {}, {0, -1}, {0, -1}, {});   // remove padding from right and bottom
    x = ggml_cont(m, x); // required by subsequent ggml_scale for some reason
    return m.named(x);
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

Tensor from_rgb(ModelRef m, Tensor x) {
    x = conv_2d(m["fromrgb"], x);
    x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    return m.named(x);
}

std::pair<Tensor, Tensor> encoder_block(ModelRef m, Tensor x, conv flag = conv::none) {
    Tensor feat = separable_conv_2d(m["conv1"], x, conv::activation);
    x = separable_conv_2d(m["conv2"], feat, conv::activation | flag);
    return {x, feat};
}

using Features = std::array<Tensor, 9>;

std::pair<Tensor, Features> encode(ModelRef m, Tensor x, int res) {
    ASSERT(res == int(x->ne[1]));
    int n = int(log2f(res)) - 1;
    ASSERT((1 << (n + 1)) == res);

    x = from_rgb(m[TensorName("b{}", res)], x);
    Features feats{};
    for (int i = 0; i < n - 1; ++i) {
        ModelRef block = m[TensorName("b{}", res >> i)];
        std::tie(x, feats[i]) = encoder_block(block, x, conv::downsample);
    }
    std::tie(x, feats[n - 1]) = encoder_block(m["b4"], x);
    return {x, feats};
}

std::pair<Tensor, Tensor> synthesis_block(ModelRef m, Tensor x, Tensor feat, Tensor img,
                                          conv up_flag = conv::none, conv noise_flag = conv::none) {
    x = separable_conv_2d(m["conv1"], x, conv::activation | noise_flag | up_flag);
    x = ggml_add_inplace(m, x, feat);
    x = separable_conv_2d(m["conv2"], x, conv::activation | noise_flag);

    if (img) {
        img = upsample_2d(m["upsample"], img);
    }
    Tensor y = conv_2d(m["torgb"], x);
    img = img ? ggml_add_inplace(m, img, y) : y;

    return {x, img};
}

Tensor synthesis(ModelRef m, Tensor x_in, Features feats, int res) {
    int n = int(log2f(res)) - 1;
    ASSERT((1 << (n + 1)) == res);

    auto [x, img] = synthesis_block(m["b4"], x_in, feats[n - 1], nullptr);
    for (int i = n - 2; i >= 0; --i) {
        ModelRef block = m[TensorName("b{}", res >> i)];
        std::tie(x, img) = synthesis_block(block, x, feats[i], img, conv::upsample, conv::noise);
    }
    return img;
}

Tensor run(ModelRef m, Tensor image, int res) {
    auto [x, feats] = encode(m["encoder"], image, res);
    return synthesis(m["synthesis"], x, feats, res);
}

std::vector<float> preprocess(ImageView image, ImageView mask, int res, bool invert_mask = false) {
    std::optional<Image> resized_image;
    if (image.extent.width != res || image.extent.height != res) {
        resized_image = resize(image, Extent(res, res));
        image = ImageView(*resized_image);
    }
    std::optional<Image> resized_mask;
    if (mask.extent.width != res || mask.extent.height != res) {
        resized_mask = resize(mask, Extent(res, res));
        mask = ImageView(*resized_mask);
    }
    PixelAccessor rgb(image);
    PixelAccessor alpha(mask);
    const float scale = 2.0f / 255.0f;
    const uint8_t no_fill = invert_mask ? 0 : 255;
    std::vector<float> result(4 * res * res);

    for (int y = 0; y < res; ++y) {
        for (int x = 0; x < res; ++x) {
            int i = y * res * 4 + x * 4;
            float a = alpha.get(mask.pixels, x, y, 0) == no_fill ? 1.0f : 0.0f;
            result[i + 0] = a - 0.5f;
            result[i + 1] = a * (rgb.get(image.pixels, x, y, 0) * scale - 1.0f);
            result[i + 2] = a * (rgb.get(image.pixels, x, y, 1) * scale - 1.0f);
            result[i + 3] = a * (rgb.get(image.pixels, x, y, 2) * scale - 1.0f);
        }
    }
    return result;
}

Image postprocess(std::span<float> data, Extent extent) {
    auto image = Image(Extent(512, 512), Channels::rgb);
    image_from_float(data, std::span(image.pixels(), data.size()), 0.5f, 0.5f);
    if (extent.width != 512 || extent.height != 512) {
        return resize(image, extent);
    }
    return image;
}

} // namespace dlimg::migan