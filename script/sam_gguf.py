# Convert MobileSAM (Tiny-ViT) model checkpoint to a gguf format
#

import itertools
import sys
import torch
import numpy as np
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


def build_dense_positional_embeddings(
    positional_encoding_gaussian_matrix: torch.Tensor, image_embedding_size=64
):
    # from sam/modeling/prompt_encoder.py - PositionEmbeddingRandom
    h, w = image_embedding_size, image_embedding_size
    grid = torch.ones((h, w), dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    coords = torch.stack((x_embed, y_embed), dim=-1)
    coords = 2 * coords - 1
    coords = coords @ positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # outputs d_1 x ... x d_n x C shape
    pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    pe = pe.permute(2, 0, 1)
    return pe


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
    name = name.replace("_image_to_token.", "_i2t.")
    name = name.replace("_token_to_image.", "_t2i.")

    if name.endswith("attention_biases"):
        num_heads = tensor.shape[0]
        resolution = {4: 7, 5: 14, 10: 7}[num_heads]
        attention_bias_idxs = build_attention_bias_indices(resolution)
        name = name + "_indexed"
        tensor = tensor[:, attention_bias_idxs]

    if name.endswith("running_var"):
        tensor = torch.sqrt(tensor + batch_norm_eps)

    # Precompute dense positional embeddings from random matrix stored in the model
    if name == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
        pe = build_dense_positional_embeddings(tensor)
        print(pe.unsqueeze(0))
        writer.add_tensor("dec.dense_positional_embedding", pe.numpy())

    tensor_data = tensor.numpy()
    print(name, tensor.shape, tensor_data.dtype)
    writer.add_tensor(name, tensor_data)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file(progress=True)
writer.close()

print("")
