import cv2
import numpy as np
import torch

from . import workbench


## Foreground estimation reference
# from https://github.com/Photoroom/fast-foreground-estimation


def FB_blur_fusion_foreground_estimator_1(image, alpha, r=90):
    alpha = alpha[:, :, None]
    return FB_blur_fusion_foreground_estimator(
        image, F=image, B=image, alpha=alpha, r=r
    )[0]


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=7)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    blurred_alpha = cv2.blur(alpha, (r, r), borderType=cv2.BORDER_REPLICATE)[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r), borderType=cv2.BORDER_REPLICATE)
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r), borderType=cv2.BORDER_REPLICATE)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


##


def test_blur():
    n = 1024
    r = 30
    image = np.random.rand(n, n, 4).astype(np.float32)

    expected = cv2.blur(image, (r * 2 + 1, r * 2 + 1), borderType=cv2.BORDER_REPLICATE)

    result = torch.zeros(1, n, n, 4, dtype=torch.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    result = workbench.invoke_test("blur", image, result, {})
    result = result.squeeze(0)

    expected = torch.from_numpy(expected)
    assert torch.allclose(result, expected, atol=1e-4)


def test_estimate_foreground():
    image = np.random.rand(256, 256, 4).astype(np.float32)
    alpha = np.random.rand(256, 256).astype(np.float32)

    expected = FB_blur_fusion_foreground_estimator_2(image, alpha, r=61)

    result = torch.zeros(1, 256, 256, 4, dtype=torch.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    alpha = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0)
    result = workbench.invoke_test(
        "estimate_foreground", image, result, {"mask": alpha}
    )
    result = result.squeeze(0)

    expected = torch.from_numpy(expected)
    assert torch.allclose(result, expected, atol=1e-4)
