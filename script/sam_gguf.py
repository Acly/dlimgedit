# Convert MobileSAM (Tiny-ViT) model checkpoint to a gguf format
#

import itertools
import sys
import torch
from pathlib import Path

import gguf


def build_attention_bias_indices(resolution: int):
    points = list(itertools.product(range(resolution), range(resolution)))
    N = len(points)
    attention_offsets = {}
    idxs = []
    for p1 in points:
        for p2 in points:
            offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
            if offset not in attention_offsets:
                attention_offsets[offset] = len(attention_offsets)
            idxs.append(attention_offsets[offset])

    return torch.LongTensor(idxs).view(N, N)


if len(sys.argv) < 3:
    print("Usage: sam_gguf.py file-model dir-output\n")
    sys.exit(1)

fname_model = sys.argv[1]
dir_out = sys.argv[2]
fname_out = Path(dir_out) / Path(fname_model).name.replace(".pt", ".gguf")

ftype = 0  # fp32

model: dict[str, torch.Tensor] = torch.load(
    fname_model, map_location="cpu", weights_only=True
)

writer = gguf.GGUFWriter(fname_out, "sam")
writer.add_name("MobileSAM")

batch_norm_eps = 1e-5

for name, tensor in model.items():
    name = name.replace("image_encoder.", "enc.")
    name = name.replace("mask_decoder.", "dec.")
    name = name.replace("cross_attn_image_to_token.", "cross_attn_i2t.")
    name = name.replace("cross_attn_token_to_image.", "cross_attn_t2i.")

    if name.endswith("attention_biases"):
        num_heads = tensor.shape[0]
        resolution = {4: 7, 5: 14, 10: 7}[num_heads]
        attention_bias_idxs = build_attention_bias_indices(resolution)
        name = name + "_indexed"
        tensor = tensor[:, attention_bias_idxs]

    if name.endswith("running_var"):
        tensor = tensor + batch_norm_eps

    tensor_data = tensor.numpy()
    print(name, tensor.shape, tensor_data.dtype)
    writer.add_tensor(name, tensor_data)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()

print("")
