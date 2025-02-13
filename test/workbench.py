import ctypes
import torch

from pathlib import Path

float_ptr = ctypes.POINTER(ctypes.c_float)


class RawTensor(ctypes.Structure):
    _fields_ = [
        ("data", float_ptr),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("c", ctypes.c_int),
        ("n", ctypes.c_int),
    ]


def to_raw_tensor(tensor: torch.Tensor):
    assert tensor.is_contiguous()
    assert tensor.dtype == torch.float32
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 4
    raw_tensor = RawTensor()
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


def invoke_test(test_case: str, inputs: list[torch.Tensor], output: torch.Tensor):
    raw_inputs = (to_raw_tensor(tensor) for tensor in inputs)
    raw_output = to_raw_tensor(output)
    result = lib.dlimg_workbench(
        test_case.encode(),
        len(inputs),
        (RawTensor * len(inputs))(*raw_inputs),
        ctypes.byref(raw_output),
    )
    assert result == 0, f"Test case {test_case} failed"


def print_results(result: torch.Tensor, expected: torch.Tensor):
    print("\nresult -----", result, sep="\n")
    print("\nexpected ---", expected, sep="\n")
