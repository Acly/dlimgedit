#include <ggml.h>

#include <fmt/format.h>

namespace dlimg {

inline ggml_tensor* conv_2d(ggml_context* c, ggml_tensor* x, ggml_tensor* weight, int stride = 1,
                            int pad = 0, int dilation = 1, ggml_tensor* bias = nullptr) {

    x = ggml_conv_2d(c, weight, x, stride, stride, pad, pad, dilation, dilation);
    if (bias) {
        bias = ggml_reshape_4d(c, bias, 1, 1, bias->ne[0], 1);
        x = ggml_add_inplace(c, x, bias);
    }
    return x;
}

inline ggml_tensor* conv_2d_depth_wise(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight,
                                       int stride = 1, int pad = 0, int dilation = 1) {
    auto a = weight;
    auto b = x;
    int s0 = stride;
    int s1 = stride;
    int p0 = pad;
    int p1 = pad;
    int d0 = dilation;
    int d1 = dilation;

    // Copied from ggml.c, fixed hardcoded GGML_TYPE_F16
    ggml_tensor* new_a = ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    ggml_tensor* im2col =
        ggml_im2col(ctx, new_a, ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                    s0, s1, p0, p1, d0, d1, true, b->type); // [N * IC, OH, OW, KH * KW]
    ggml_tensor* new_b =
        ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2],
                        b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2], new_a->ne[3],
                            1); // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    ggml_tensor* result = ggml_mul_mat(ctx, new_a, new_b);
    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2],
                             b->ne[3]); // [N, OC, OH, OW]

    return result;
}

inline ggml_tensor* batch_norm_2d(ggml_context* c, ggml_tensor* x, ggml_tensor* weight,
                                  ggml_tensor* bias, ggml_tensor* mean, ggml_tensor* var,
                                  float eps = 1e-5) {
    var = ggml_sqrt(c, var);
    var = ggml_reshape_4d(c, var, 1, 1, var->ne[0], 1);
    // var = ggml_add_inplace(c, var, eps);
    mean = ggml_reshape_4d(c, mean, 1, 1, mean->ne[0], 1);
    weight = ggml_reshape_4d(c, weight, 1, 1, weight->ne[0], 1);
    bias = ggml_reshape_4d(c, bias, 1, 1, bias->ne[0], 1);
    x = ggml_sub(c, x, mean);
    x = ggml_div(c, x, var);
    x = ggml_mul(c, x, weight);
    x = ggml_add(c, x, bias);
    return x;
}

} // namespace dlimg