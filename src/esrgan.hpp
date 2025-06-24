#pragma once

#include "ml.hpp"

namespace dlimg::esrgan {

struct ESRGANParams {
    int scale = 4;
    int n_blocks = 23;

    static ESRGANParams detect(ModelRef m);
};

Tensor upscale(ModelRef m, Tensor x, ESRGANParams const& p);

Tensor upsample(ModelRef m, Tensor x);
Tensor conv_block(ModelRef m, Tensor x);
Tensor risidual_dense_block(ModelRef m, Tensor x);
Tensor rrdb(ModelRef m, Tensor x);

} // namespace dlimg::esrgan