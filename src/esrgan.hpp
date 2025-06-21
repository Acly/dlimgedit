#pragma once

#include "birefnet.hpp"
#include "ml.hpp"
#include "mobile_sam.hpp"

namespace dlimg::esrgan {
using sam::conv_2d;

int log2(int x) {
    int result = 0;
    for (; x > 1; x >>= 1) {
        ++result;
    }
    return result;
}

Tensor upsample(ModelRef m, Tensor x) {
    auto [c, w, h, n] = nelements(x);
    x = ggml_upscale_ext(m, x, c, w * 2, h * 2, n, GGML_SCALE_MODE_NEAREST);
    x = conv_2d(m, x, 1, 1);
    x = ggml_leaky_relu(m, x, 0.2f, true);
    return m.named(x);
}

Tensor conv_block(ModelRef m, Tensor x) {
    x = conv_2d(m[0], x, 1, 1);
    x = ggml_leaky_relu(m, x, 0.2f, true);
    return x;
}

Tensor risidual_dense_block(ModelRef m, Tensor x) {
    Tensor x1 = conv_block(m["conv1"], x);
    Tensor c1 = concat(m, {x, x1}, 0);
    Tensor x2 = conv_block(m["conv2"], c1);
    Tensor c2 = concat(m, {c1, x2}, 0);
    Tensor x3 = conv_block(m["conv3"], c2);
    Tensor c3 = concat(m, {c2, x3}, 0);
    Tensor x4 = conv_block(m["conv4"], c3);
    Tensor c4 = concat(m, {c3, x4}, 0);
    Tensor x5 = conv_2d(m["conv5.0"], c4, 1, 1);
    x5 = ggml_scale_inplace(m, x5, 0.2f);
    x = ggml_add(m, x, x5);
    return m.named(x);
}

Tensor rrdb(ModelRef m, Tensor x) {
    Tensor x_in = x;
    x = risidual_dense_block(m["RDB1"], x);
    x = risidual_dense_block(m["RDB2"], x);
    x = risidual_dense_block(m["RDB3"], x);
    x = ggml_scale_inplace(m, x, 0.2f);
    x = ggml_add(m, x, x_in);
    return m.named(x);
}

struct ESRGANParams {
    int scale = 4;
    int n_blocks = 23;
};

Tensor upscale(ModelRef m, Tensor x, ESRGANParams const& p) {
    m = m["model"];
    x = conv_2d(m[0], x, 1, 1);

    Tensor sub = x;
    ModelRef block = m[1]["sub"];
    for (int i = 0; i < p.n_blocks; ++i) {
        sub = rrdb(block[i], sub);
    }
    sub = conv_2d(block[p.n_blocks], sub, 1, 1);
    x = ggml_add(m, x, sub);

    int seq = 2;
    for (int i = 0; i < log2(p.scale); ++i) {
        x = upsample(m[seq + 1], x);
        seq += 3;
    }
    x = conv_2d(m[seq], x, 1, 1);
    x = ggml_leaky_relu(m, x, 0.2f, true);
    x = conv_2d(m[seq + 2], x, 1, 1);
    
    return mark_output(m, x, "result");
}

} // namespace dlimg::esrgan