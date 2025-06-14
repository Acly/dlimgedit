#pragma once

#include "ml.hpp"

namespace dlimg::migan {

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

} // dlimg::migan