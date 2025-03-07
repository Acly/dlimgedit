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
    return ggml_conv_2d_depthwise_cont_channels(m, m.weights("weight"), x, stride, stride, pad, GGML_NCHW);
}

inline Tensor conv_2d_depth_wise_channels(Model m, Tensor x, int stride = 1, int pad = 0) {
    Tensor kernel = m.weights("weight");
    // TODO: reshape flips input-depth and depth-multiplier
    // weights are arranged wrong to make regular 1x1 conv2d work as ggml_out_prod
    int64_t c = kernel->ne[0];
    GGML_ASSERT(kernel->ne[1] == 1);
    kernel = ggml_reshape_4d(m, kernel, 1, c, kernel->ne[2], kernel->ne[3]);
    return ggml_conv_2d_depthwise_cont_channels(m, kernel, x, stride, stride, pad, GGML_NHWC);
}

inline Tensor conv_2d_channels(Model m, Tensor x, int stride = 1, int pad = 0) {
    Tensor kernel = m.weights("weight");
    int64_t c = x->ne[0];
    int64_t w = x->ne[1];
    int64_t h = x->ne[2];
    int64_t batch = x->ne[3];
    int64_t kernel_w = kernel->ne[2];
    int64_t kernel_h = kernel->ne[3];
    int64_t kernel_c = kernel->ne[1];
    int64_t kernel_count = kernel->ne[0];

    if (kernel_w == 1 && kernel_h == 1 && stride == 1) {
        kernel = ggml_reshape_3d(m, kernel, kernel_count, kernel_c * kernel_w * kernel_h, 1);
        x = ggml_reshape_3d(m, x, c, w * h, batch);
        x = ggml_transpose(m, x);
        x = ggml_out_prod(m, kernel, x);
        x = ggml_reshape_4d(m, x, kernel_count, w, h, batch);
    } else {
        x = ggml_conv_2d_cont_channels(m, kernel, x, stride, stride, pad);
    }
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

inline Tensor batch_norm_2d(Model m, Tensor x, float eps = 1e-5f) {
    Tensor var = m.weights("running_var"); // = sqrt(var + eps)
    Tensor mean = m.weights("running_mean");
    Tensor weight = m.weights("weight");
    Tensor bias = m.weights("bias");

    var = ggml_reshape_4d(m, var, var->ne[0], 1, 1, 1);
    mean = ggml_reshape_4d(m, mean, mean->ne[0], 1, 1, 1);
    weight = ggml_reshape_4d(m, weight, weight->ne[0], 1, 1, 1);
    bias = ggml_reshape_4d(m, bias, bias->ne[0], 1, 1, 1);

    x = ggml_sub_inplace(m, x, mean);
    x = ggml_div_inplace(m, x, var);
    x = ggml_mul_inplace(m, x, weight);
    x = ggml_add_inplace(m, x, bias);
    return m.named(x);
}

} // namespace dlimg