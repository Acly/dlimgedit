#pragma once

#include "ml.hpp"

#include <ggml.h>

#include <fmt/format.h>

namespace dlimg {

inline Tensor linear(Model m, Tensor x) {
    x = ggml_mul_mat(m, m.weights("weight"), x);
    if (Tensor bias = m.find("bias")) {
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

inline Tensor conv_2d(Model m, Tensor x, int stride = 1, int pad = 0) {
    x = ggml_conv_2d(m, m.weights("weight"), x, stride, stride, pad, pad, 1, 1);
    if (Tensor bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

inline Tensor conv_2d_depth_wise(Model m, Tensor x, int stride = 1, int pad = 0) {
    return ggml_depthwise_conv_2d(m, m.weights("weight"), x, stride, stride, pad, pad, GGML_NCHW);
}

inline Tensor conv_2d_depth_wise_channels(Model m, Tensor x, int stride = 1, int pad = 0) {
    return ggml_depthwise_conv_2d(m, m.weights("weight"), x, stride, stride, pad, pad, GGML_NHWC);
}

inline Tensor conv_2d_channels(Model m, Tensor x, int stride = 1, int pad = 0) {
    x = ggml_conv_2d_ext(m, m.weights("weight"), x, stride, stride, pad, pad, 1, 1, GGML_NHWC);
    if (Tensor bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, bias->ne[0], 1, 1, 1);
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

inline Tensor layer_norm(Model m, Tensor x, float eps = 1e-5f) {
    x = ggml_norm(m, x, eps);
    x = ggml_mul_inplace(m, x, m.weights("weight"));
    x = ggml_add_inplace(m, x, m.weights("bias"));
    return x;
}

inline Tensor batch_norm_2d(Model m, Tensor x) {
    Tensor var = m.weights("running_var"); // = sqrt(var + eps)
    Tensor mean = m.weights("running_mean");
    Tensor weight = m.weights("weight");
    Tensor bias = m.weights("bias");
    x = ggml_batch_norm_2d_inplace(m, x, mean, var, weight, bias);
    return m.named(x);
}

} // namespace dlimg