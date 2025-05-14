# Convert BiRefNet model checkpoint to gguf format
#

import itertools
import sys
import torch
import safetensors
import numpy as np
from pathlib import Path
from torch import Tensor

import gguf


def conv_2d_kernel_to_nhwc(kernel: torch.Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


fname_model = sys.argv[1]
dir_out = sys.argv[2]
fname_out = Path(dir_out) / Path(fname_model).name.replace(".safetensors", ".gguf")


model: dict[str, torch.Tensor] = safetensors.safe_open(fname_model, "pt")

writer = gguf.GGUFWriter(fname_out, "birefnet")
writer.add_name("BiRefNet")

window_size = 12  # swin-l: 12, swin-t: 7
batch_norm_eps = 1e-5

conv_2d_names = ["bb.patch_embed.proj.weight"]

pbt = None

for name in model.keys():
    tensor = model.get_tensor(name)

    # Shorten some names to fit into 64 chars
    name = name.replace("decoder_block", "block")
    name = name.replace("atrous_conv", "conv")
    name = name.replace("modulator_conv", "modulator")
    name = name.replace("offset_conv", "offset")
    name = name.replace("regular_conv", "conv")

    if len(name) >= 64:
        print("Warning: name too long", len(name), name)

    if name.endswith("relative_position_index"):
        continue  # precomputed in c++ code

    # BatchNorm2d: precompute sqrt(var + eps)
    if name.endswith("running_var"):
        tensor = torch.sqrt(tensor + batch_norm_eps)

    # Conv2d: convert to NHWC format
    if name in conv_2d_names:
        assert tensor.shape[2] == tensor.shape[3] and tensor.shape[2] <= 4
        tensor = conv_2d_kernel_to_nhwc(tensor)

    if tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)

    tensor_data = tensor.numpy()
    print(name, tensor.shape, tensor_data.dtype)
    writer.add_tensor(name, tensor_data)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()

print("")
