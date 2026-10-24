# =============================================================================
# 训练数据增强
#
# 所有几何变换都同时作用于 data（2x2 输入 SAI）和 label（angout x angout
# 标签 SAI），否则输入与监督的角/空间对应关系会被破坏。
#
# 支持两种后端：
#   - torch.Tensor：训练时的主路径；
#   - numpy.ndarray：便于不安装 PyTorch 的单元测试/策略测试。
# 最后两个维度始终被解释为单视角内的 (s, t) 空间轴，前面的轴（视角网格、
# 通道等）原样保留。
# =============================================================================
import random

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Allows dependency-light policy/augmentation tests without PyTorch.
    torch = None
    F = None


def _is_torch_tensor(value):
    # 判断输入是否为 torch.Tensor；torch 未安装时固定返回 False。
    return torch is not None and torch.is_tensor(value)


def _flip(value, axis):
    # 按指定轴翻转，自动分派到 torch.flip 或 np.flip。
    if _is_torch_tensor(value):
        return torch.flip(value, dims=[axis])
    return np.flip(value, axis=axis)


def _transpose_last_two_axes(value):
    # 交换最后两个空间轴（s,t），对应把每张视角图转置。
    if _is_torch_tensor(value):
        return value.transpose(-2, -1)
    return np.swapaxes(value, -2, -1)


def _resize_bilinear_numpy(image, out_h, out_w):
    """Bilinearly resize a 2-D array using PyTorch's align_corners=False grid."""
    # Bilinearly resize a 2-D array using PyTorch's align_corners=False grid.
    #
    #     NumPy 后端没有 F.interpolate，这里手动实现与 PyTorch
    #     ``mode='bilinear', align_corners=False`` 一致的采样：
    #       1. 输出像素中心映射回输入坐标：(i + 0.5) * in / out - 0.5；
    #       2. 坐标夹到 [0, in-1]，再取 floor/ceil 四个邻点；
    #       3. 先横向、后纵向做两次线性插值。
    #     仅用于二维单通道图，且 out_h/out_w 均为正数。
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
    # Replace each angular view by a centered crop enlarged to its original size.
    #
    #     这是“视角内部”的空间增强，不是角度插值：
    #       - 把 SAI 按 ang_res x ang_res 切成视角网格；
    #       - 每个视角取中心 1/factor 区域（crop_h/crop_w 至少为 1）；
    #       - 双线性放大回原视角尺寸 patch_h x patch_w；
    #       - 输入/输出形状完全一致，因此可以直接替换 data 或 label。
    #
    #     参数 value 的最后两维是 SAI 的空间尺寸，必须能被 ang_res 整除。
    #     torch 路径逐视角调用 F.interpolate；numpy 路径调用
    #     ``_resize_bilinear_numpy``，两者使用同一套坐标约定。
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
    # Apply synchronized geometric augmentation over the last two SAI axes.
    #
    #     依次以 0.5 概率执行：水平翻转（最后一维 t）、垂直翻转（倒数第二维 s）、
    #     转置（交换 s/t）。三种变换彼此独立，且 data 与 label 使用同一组随机
    #     判定，保证监督像素仍然一一对应。
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
    # Apply 50%-probability center zoom, followed by geometric augmentation.
    #
    #     先用 data 的空间高和固定的输入角度数 2 推出一张输入视角的高 view_h，
    #     再由 label 的高推出标签角度数（因此训练侧必须保持 data 为 2x2 输入）。
    #     若 label 的空间块与 data 不等高，说明两者空间分辨率不一致，直接报错，
    #     避免在不匹配的监督对上继续训练。
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
