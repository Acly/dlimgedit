# Convert ESRGAN model checkpoint to gguf format
#

import sys
import torch
from pathlib import Path

import gguf


def conv_2d_kernel_to_nhwc(kernel: torch.Tensor):
    c_in = kernel.shape[1]
    if c_in == 1:  # depthwise
        return kernel.permute(2, 3, 1, 0)  # H W 1 C_out
    else:
        return kernel.permute(0, 2, 3, 1)  # C_out H W C_in


in_filepath = sys.argv[1]
out_dir = sys.argv[2]
convert_f16 = len(sys.argv) > 3 and sys.argv[3] == "f16"

out_filename = Path(in_filepath).name
out_filename = out_filename.replace(".pt", "").replace(".pth", "")
if convert_f16:
    out_filename += "-F16"
out_filename += ".gguf"
out_filepath = Path(out_dir) / out_filename


model: dict[str, torch.Tensor] = torch.load(in_filepath, weights_only=True)

writer = gguf.GGUFWriter(out_filepath, "esrgan")
writer.add_name("ESRGAN")

for name, tensor in model.items():
    if len(name) >= 64:
        print("Warning: name too long", len(name), name)

    # Conv2d: convert to NHWC format
    is_conv = (
        tensor.ndim == 4
        and tensor.shape[2] == tensor.shape[3]
        and tensor.shape[2] in (1, 3, 4, 7)
        and name.endswith("weight")
    )
    if is_conv:
        tensor = conv_2d_kernel_to_nhwc(tensor)

    if convert_f16 and tensor.dtype == torch.float32:
        tensor = tensor.to(torch.float16)

    tensor_data = tensor.numpy()
    print("⇄" if is_conv else "○", name, tensor.shape, tensor_data.dtype)
    writer.add_tensor(name, tensor_data)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()

print("")
