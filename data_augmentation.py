import random

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Allows dependency-light policy/augmentation tests without PyTorch.
    torch = None
    F = None


def _is_torch_tensor(value):
    return torch is not None and torch.is_tensor(value)


def _flip(value, axis):
    if _is_torch_tensor(value):
        return torch.flip(value, dims=[axis])
    return np.flip(value, axis=axis)


def _transpose_last_two_axes(value):
    if _is_torch_tensor(value):
        return value.transpose(-2, -1)
    return np.swapaxes(value, -2, -1)


def _resize_bilinear_numpy(image, out_h, out_w):
    """Bilinearly resize a 2-D array using PyTorch's align_corners=False grid."""
    in_h, in_w = image.shape
    y = (np.arange(out_h, dtype=np.float64) + 0.5) * in_h / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float64) + 0.5) * in_w / out_w - 0.5
    y = np.clip(y, 0, in_h - 1)
    x = np.clip(x, 0, in_w - 1)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = image[y0[:, None], x0[None, :]] * (1 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1 - wx) + image[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def resize_sai_center_zoom(value, ang_res, factor=4):
    """Replace each angular view by a centered crop enlarged to its original size."""
    if not isinstance(ang_res, int) or ang_res < 1:
        raise ValueError("ang_res must be a positive integer")
    if not isinstance(factor, int) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if value.ndim < 2:
        raise ValueError("value must have at least two spatial dimensions")
    height, width = value.shape[-2:]
    if height % ang_res or width % ang_res:
        raise ValueError("SAI height and width must be divisible by ang_res")
    patch_h, patch_w = height // ang_res, width // ang_res
    crop_h, crop_w = max(1, patch_h // factor), max(1, patch_w // factor)
    start_h, start_w = (patch_h - crop_h) // 2, (patch_w - crop_w) // 2

    if _is_torch_tensor(value):
        original_dtype = value.dtype
        flat = value.reshape(-1, height, width)
        output = torch.empty_like(flat)
        for sample_idx in range(flat.shape[0]):
            for u in range(ang_res):
                for v in range(ang_res):
                    y0, y1 = u * patch_h, (u + 1) * patch_h
                    x0, x1 = v * patch_w, (v + 1) * patch_w
                    crop = flat[sample_idx, y0 + start_h:y0 + start_h + crop_h,
                                 x0 + start_w:x0 + start_w + crop_w]
                    resized = F.interpolate(
                        crop.to(torch.float32)[None, None],
                        size=(patch_h, patch_w), mode="bilinear", align_corners=False,
                    )[0, 0]
                    output[sample_idx, y0:y1, x0:x1] = resized.to(original_dtype)
        return output.reshape(value.shape)

    flat = np.asarray(value).reshape(-1, height, width)
    output = np.empty_like(flat)
    for sample_idx in range(flat.shape[0]):
        for u in range(ang_res):
            for v in range(ang_res):
                y0, y1 = u * patch_h, (u + 1) * patch_h
                x0, x1 = v * patch_w, (v + 1) * patch_w
                crop = flat[sample_idx, y0 + start_h:y0 + start_h + crop_h,
                             x0 + start_w:x0 + start_w + crop_w]
                output[sample_idx, y0:y1, x0:x1] = _resize_bilinear_numpy(
                    crop, patch_h, patch_w
                ).astype(output.dtype, copy=False)
    return output.reshape(value.shape)


def augmentation(data, label, rng=None):
    """Apply synchronized geometric augmentation over the last two SAI axes."""
    rng = random if rng is None else rng
    if rng.random() < 0.5:
        data = _flip(data, axis=-1)
        label = _flip(label, axis=-1)
    if rng.random() < 0.5:
        data = _flip(data, axis=-2)
        label = _flip(label, axis=-2)
    if rng.random() < 0.5:
        data = _transpose_last_two_axes(data)
        label = _transpose_last_two_axes(label)
    return data, label


def augmentation_with_resize(data, label, rng=None, resize_prob=0.5):
    """Apply 50%-probability center zoom, followed by geometric augmentation."""
    rng = random if rng is None else rng
    if rng.random() < resize_prob:
        # Current training uses a 2x2 input and angout x angout label.
        data_ang_res = 2
        view_h = data.shape[-2] // data_ang_res
        if view_h < 1 or label.shape[-2] % view_h:
            raise ValueError("data and label must contain equally sized spatial views")
        label_ang_res = label.shape[-2] // view_h
        data = resize_sai_center_zoom(data, data_ang_res)
        label = resize_sai_center_zoom(label, label_ang_res)
    return augmentation(data, label, rng=rng)
