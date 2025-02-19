import ctypes
import torch

from pathlib import Path

float_ptr = ctypes.POINTER(ctypes.c_float)


class RawTensor(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", float_ptr),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("c", ctypes.c_int),
        ("n", ctypes.c_int),
    ]


def to_raw_tensor(name: str, tensor: torch.Tensor):
    assert tensor.is_contiguous()
    assert tensor.dtype == torch.float32
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 4
    raw_tensor = RawTensor()
    raw_tensor.name = name.encode()
    raw_tensor.data = ctypes.cast(tensor.float().data_ptr(), float_ptr)
    raw_tensor.w = tensor.size(3)
    raw_tensor.h = tensor.size(2)
    raw_tensor.c = tensor.size(1)
    raw_tensor.n = tensor.size(0)
    return raw_tensor


root_dir = Path(__file__).parent.parent
lib = ctypes.CDLL(str(root_dir / "build" / "bin" / "dlimgedit_workbench.dll"))
lib.dlimg_workbench.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int32,
    ctypes.POINTER(RawTensor),
    ctypes.POINTER(RawTensor),
]
lib.dlimg_workbench.restype = ctypes.c_int32


def invoke_test(
    test_case: str,
    input: torch.Tensor,
    output: torch.Tensor,
    state: dict[str, torch.Tensor],
    **kwargs: dict[str, torch.Tensor],
):
    state.update(kwargs)
    raw_inputs = [to_raw_tensor("input", input)]
    raw_inputs += [to_raw_tensor(name, tensor) for name, tensor in state.items()]
    raw_output = to_raw_tensor("output", output)
    result = lib.dlimg_workbench(
        test_case.encode(),
        len(raw_inputs),
        (RawTensor * len(raw_inputs))(*raw_inputs),
        ctypes.byref(raw_output),
    )
    assert result == 0, f"Test case {test_case} failed"


def randomize(state_dict: dict[str, torch.Tensor]):
    return {
        k: torch.rand_like(v)
        for k, v in state_dict.items()
        if v.dtype.is_floating_point
    }


def print_results(result: torch.Tensor, expected: torch.Tensor):
    print("\nresult -----", result, sep="\n")
    print("\nexpected ---", expected, sep="\n")
