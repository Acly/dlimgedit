import ctypes
from functools import reduce
import torch
import os

from pathlib import Path

float_ptr = ctypes.POINTER(ctypes.c_float)


class RawTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("ne", ctypes.c_int * 4),
    ]


tensor_types = {
    torch.float32: 0,  # GGML_TYPE_F32
    torch.int32: 26,  # GGML_TYPE_I32
}


def to_raw_tensor(name: str, tensor: torch.Tensor):
    t = tensor.contiguous()
    if tensor.dtype not in tensor_types:
        print(
            f"Warning: tensor {name} has unsupported dtype {tensor.dtype}, converting to float"
        )
        t = t.to(torch.float32)
    while t.dim() < 4:
        t = t.unsqueeze(0)
    assert t.dim() == 4
    raw_tensor = RawTensor()
    raw_tensor.name = name.encode()
    raw_tensor.data = ctypes.cast(t.data_ptr(), ctypes.c_void_p)
    raw_tensor.type = tensor_types[t.dtype]
    raw_tensor.ne[0] = t.size(3)
    raw_tensor.ne[1] = t.size(2)
    raw_tensor.ne[2] = t.size(1)
    raw_tensor.ne[3] = t.size(0)
    return (raw_tensor, t)


os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "TRUE"  # Some python lib (numpy? torch?) loads OpenMP
)

root_dir = Path(__file__).parent.parent
lib = ctypes.CDLL(str(root_dir / "build" / "bin" / "dlimgedit_workbench.dll"))
lib.dlimg_workbench.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.POINTER(RawTensor),
    ctypes.POINTER(RawTensor),
    ctypes.c_int32,
]
lib.dlimg_workbench.restype = ctypes.c_int32


def invoke_test(
    test_case: str,
    input: torch.Tensor,
    output: torch.Tensor,
    state: dict[str, torch.Tensor],
    backend: str = "cpu",
    **kwargs: dict[str, torch.Tensor],
):
    state.update(kwargs)
    raw_inputs = [to_raw_tensor("input", input)]
    raw_inputs += [to_raw_tensor(name, tensor) for name, tensor in state.items()]
    input_tensors = [t for _, t in raw_inputs]
    input_tensors  # keep the tensors alive
    raw_inputs = [t for t, _ in raw_inputs]
    raw_output, output_tensor = to_raw_tensor("output", output)
    result = lib.dlimg_workbench(
        test_case.encode(),
        len(raw_inputs),
        (RawTensor * len(raw_inputs))(*raw_inputs),
        ctypes.byref(raw_output),
        1 if backend == "cpu" else 2,
    )
    assert result == 0, f"Test case {test_case} failed"
    return output_tensor


def input_tensor(*shape: tuple[int]):
    end = reduce(lambda x, y: x * y, shape, 1)
    return torch.arange(0, end).reshape(*shape) / end


def input_like(tensor: torch.Tensor):
    return input_tensor(*tensor.shape)


def generate_state(state_dict: dict[str, torch.Tensor]):
    return {
        k: input_like(v) if v.dtype.is_floating_point else v
        for k, v in state_dict.items()
    }


def randomize(state_dict: dict[str, torch.Tensor]):
    return {
        k: torch.rand_like(v)
        for k, v in state_dict.items()
        if v.dtype.is_floating_point
    }


def to_channel_last(tensor: torch.Tensor):
    return tensor.permute(0, 2, 3, 1).contiguous()


def revert_channel_last(tensor: torch.Tensor):
    return tensor.permute(0, 3, 1, 2).contiguous()


def convert_to_channel_last(state: dict[str, torch.Tensor], key="c.weight"):
    for k, v in state.items():
        if k.endswith(key):
            if v.shape[1] == 1:  # depthwise
                state[k] = v.permute(2, 3, 1, 0).contiguous()
            else:
                state[k] = v.permute(0, 2, 3, 1).contiguous()
    return state


def print_results(result: torch.Tensor, expected: torch.Tensor):
    print("\ntorch seed:", torch.initial_seed())
    print("\nresult -----", result, sep="\n")
    print("\nexpected ---", expected, sep="\n")
