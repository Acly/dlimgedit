import torch
import torchvision
import numpy as np

from torch import Tensor, nn
from torch.nn import functional as F

from . import workbench
from .workbench import to_channel_last, revert_channel_last, convert_to_channel_last
from .workbench import input_tensor, generate_state

torch.set_printoptions(precision=3, sci_mode=False)

class lrelu_agc:
    def __init__(self, alpha=0.2, gain=1, clamp=None):
        self.alpha = alpha
        if gain == 'sqrt_2':
            self.gain = np.sqrt(2)
        else:
            self.gain = gain
        self.clamp = clamp

    def __call__(self, x, gain=1):
        print("call lrelu_agc")
        x = F.leaky_relu(x, negative_slope=self.alpha, inplace=True)
        act_gain = self.gain * gain
        act_clamp = self.clamp * gain if self.clamp is not None else None
        if act_gain != 1:
            x = x * act_gain
        if act_clamp is not None:
            x = x.clamp(-act_clamp, act_clamp)
        return x

def test_lrelu_agc():
    x = input_tensor(1, 3, 8, 8) * 2.0 - 1.0
    lrelu = lrelu_agc(alpha=0.2, gain='sqrt_2', clamp=1.0)
    expected = lrelu(x.clone())

    result = torch.zeros_like(x)
    result = workbench.invoke_test("migan_lrelu_agc", x, result, {})

    assert torch.allclose(result, expected)