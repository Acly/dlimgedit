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

inline Tensor conv_2d(Model m, Tensor x, int stride = 1, int pad = 0, int dilation = 1) {
    x = ggml_conv_2d(m, m.weights("weight"), x, stride, stride, pad, pad, dilation, dilation);
    if (Tensor bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        x = ggml_add_inplace(m, x, bias);
    }
    return x;
}

inline Tensor conv_2d_depth_wise(Model m, Tensor x, int stride = 1, int pad = 0, int dilation = 1) {
    auto ctx = m.graph_context;
    auto a = m.weights("weight");
    auto b = x;
    int s0 = stride;
    int s1 = stride;
    int p0 = pad;
    int p1 = pad;
    int d0 = dilation;
    int d1 = dilation;

    // Copied from ggml.c, fixed hardcoded GGML_TYPE_F16
    Tensor new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    Tensor im2col =
        ggml_im2col(ctx, new_a, ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                    s0, s1, p0, p1, d0, d1, true, b->type); // [N * IC, OH, OW, KH * KW]
    Tensor new_b =
        ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2],
                        b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3],
                            1); // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    Tensor result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2],
                             b->ne[3]); // [N, OC, OH, OW]

    return result;
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

    var = ggml_reshape_4d(m, var, 1, 1, var->ne[0], 1);
    mean = ggml_reshape_4d(m, mean, 1, 1, mean->ne[0], 1);
    weight = ggml_reshape_4d(m, weight, 1, 1, weight->ne[0], 1);
    bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
    x = ggml_sub(m, x, mean);
    x = ggml_div(m, x, var);
    x = ggml_mul(m, x, weight);
    x = ggml_add(m, x, bias);
    return x;
}

} // namespace dlimg