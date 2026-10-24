# =============================================================================
# 训练/测试数据、损失、分块推理与指标工具
#
# 本文件是 train.py 的“数据与评测底座”，主要包含六类功能：
#   1. Dataset/Loader：从 HDF5 读取原始光场或已切好的固定测试对；
#   2. 在线角度裁切：从 source_ang_res x source_ang_res 的原始光场中裁出
#      本轮所需的 angout x angout 标签，再抽取四角拼成 2x2 输入；
#   3. 缺失视角损失：只监督四个角以外的视角，含 L1 和 Sobel 梯度损失；
#   4. 分块推理：LFdivide 切块（带镜像边界）、LFintegrate 拼回；
#   5. PSNR/SSIM 与逐视角评测；
#   6. 一些备用损失（目前主训练循环未调用）。
#
# 统一记法：SAI 张量为 [B,C,U*H,V*W]，拆开角度后为 [B,C,U,V,H,W]；
# 其中 (u,v) 是视角坐标，(s,t) 是单视角内的空间坐标。
# =============================================================================
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics


class TrainSetLoaderRawLF(Dataset):
    # 训练集：每个 .h5 文件存一整幅 source_ang_res x source_ang_res 的原始 SAI。
    #
    #     这里刻意不在 Dataset 里做角度裁切/四角抽取：同一份 9x9 光场在任意尺度
    #     训练时需要按当前随机 angout 现场生成不同监督对，因此裁切放到
    #     make_online_crop_pair 中完成。__getitem__ 返回形状 [1, U*H, V*W]。

    def __init__(self, dataset_dir, raw_lf_key='label'):
        super(TrainSetLoaderRawLF, self).__init__()
        self.dataset_dir = dataset_dir
        self.raw_lf_key = raw_lf_key
        self.file_list = os.listdir(dataset_dir)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = os.path.join(self.dataset_dir, self.file_list[index])
        with h5py.File(file_name, 'r') as hf:
            raw_lf = np.array(hf.get(self.raw_lf_key))
            raw_lf = ToTensor()(raw_lf.copy())
        return raw_lf

    def __len__(self):
        return self.item_num


class TestSetDataLoaderRawLF(Dataset):
    # 测试集的原始光场版本：按 args.raw_lf_key 读取，不做输入/标签配对。
    #
    #     当前入口 MultiTestSetDataLoader 实际使用的是下面的 TestSetDataLoader
    #     （固定 data/label 配对），本类属于保留的旧接口，便于需要在线裁切的
    #     评测脚本复用。

    def __init__(self, args, data_name, Lr_Info=None):
        super(TestSetDataLoaderRawLF, self).__init__()
        self.dataset_dir = os.path.join(args.testset_dir, data_name)
        self.raw_lf_key = args.raw_lf_key
        self.file_list = os.listdir(self.dataset_dir)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = os.path.join(self.dataset_dir, self.file_list[index])
        with h5py.File(file_name, 'r') as hf:
            raw_lf = np.array(hf.get(self.raw_lf_key))
            raw_lf = ToTensor()(raw_lf.copy())
        return raw_lf

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # 为测试根目录下的每个子目录各建一个 batch=1、不打乱的 DataLoader。
    #
    #     返回 (子目录名列表, loader 列表, 样本总数)。子目录通常对应一个测试场景/
    #     数据集（如 HCI 的不同类别），验证时按该顺序逐个数据集统计 PSNR/SSIM。
    #     batch 固定为 1，因为 valid() 的分块推理和逐样本指标都是按单张 SAI 组织的。
    # get testdataloader of every test dataset
    dataset_dir = args.testset_dir
    data_list = os.listdir(dataset_dir)
    print("Test data mode: fixed")

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=None)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    # 当前固定测试模式使用的 Dataset：每个 .h5 直接存一对 data/label。
    #
    #     读取后对两个数组都做 (1,0) 转置，这是该项目测试 HDF5 的存储约定：
    #     文件里按 [W,H] 存，读入后转成 [H,W] 再 ToTensor。
    #     __getitem__ 返回：
    #       data  -- [1, angin*H, angin*W] 的 2x2 输入 SAI；
    #       label -- [1, angout*H, angout*W] 的目标 SAI。

    def __init__(self, args, data_name, Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.dataset_dir = os.path.join(args.testset_dir, data_name)
        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = tmp_list[index]

        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
            data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.item_num

class TestSetLoader(Dataset):
    # 旧版 YCbCr 测试接口（当前 train.py 未使用）。
    #
    #     它读取 data_SAI_y / data_SAI_cbcr / label_SAI_ycbcr 三个键，并返回亮度、
    #     色度与标签；目录名按 angRes、scale_factor 自动拼出。保留它是为了兼容
    #     早期的彩色超分评测脚本，当前主流程只走 TestSetDataLoader。

    def __init__(self, cfg, data_name = 'ALL', Lr_Info=None):
        super(TestSetLoader, self).__init__()
        self.angRes = cfg.angRes
        self.dataset_dir = cfg.data_for_test + str(cfg.angRes) + 'x' + str(cfg.angRes) + '_' + str(cfg.scale_factor) + 'xSR/'
        data_list = [data_name]

        self.Lr_Info = self.angRes

        self.file_list = []
        for data_name in data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('data_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('data_SAI_cbcr'))
            Hr_SAI_ycbcr = np.array(hf.get('label_SAI_ycbcr'))
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_ycbcr = np.transpose(Hr_SAI_ycbcr, (0, 2, 1)).transpose(1, 2, 0)
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (0, 2, 1)).transpose(1, 2, 0)

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_ycbcr = ToTensor()(Hr_SAI_ycbcr.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        return Lr_SAI_y, Hr_SAI_ycbcr, Sr_SAI_cbcr, self.Lr_Info

    def __len__(self):
        return self.item_num


def crop_lf_sai(raw_lf, source_ang_res, angout, crop_policy='center_floor'):
    # 从原始光场 SAI 中按“视角网格”居中裁出 angout x angout 个视角。
    #
    #     注意裁的是角度维（整块视角），不是把每个视角内部裁小：
    #       - raw_lf 形状 [..., source_ang_res*H, source_ang_res*W]；
    #       - patch_h/patch_w 是单个视角的空间尺寸；
    #       - start=(source-angout)//2 取中心，向下取整（center_floor）；
    #       - 返回形状 [..., angout*H, angout*W]，空间分辨率不变。
    #
    #     例如 9x9、每视角 64x64、angout=7 时，从 576x576 中心裁出 448x448。
    if angout > source_ang_res:
        raise ValueError('angout must be <= source_ang_res')
    if crop_policy != 'center_floor':
        raise ValueError("unsupported crop_policy: {}".format(crop_policy))

    height, width = raw_lf.shape[-2:]
    if height % source_ang_res != 0 or width % source_ang_res != 0:
        raise ValueError('raw_lf spatial shape must be divisible by source_ang_res')

    patch_h = height // source_ang_res
    patch_w = width // source_ang_res
    start = (source_ang_res - angout) // 2
    end = start + angout
    return raw_lf[..., start * patch_h:end * patch_h, start * patch_w:end * patch_w]


def build_input_from_cropped_label(label, angout, angin=2):
    # 从 angout x angout 的标签 SAI 中抽四个角视角，拼成 2x2 输入 SAI。
    #
    #     label 先按视角切成 angout x angout 块，取左上、右上、左下、右下四块：
    #       top    = [左上, 右上] 横向拼接
    #       bottom = [左下, 右下] 横向拼接
    #       data   = [top; bottom] 纵向拼接
    #     输出空间尺寸为 [2*H, 2*W]，即 2x2 个视角，供网络作为唯一观测输入。
    if angin != 2:
        raise ValueError('only 2x2 corner input is supported')
    height, width = label.shape[-2:]
    if height % angout != 0 or width % angout != 0:
        raise ValueError('label spatial shape must be divisible by angout')

    patch_h = height // angout
    patch_w = width // angout
    top = torch.cat([label[..., :patch_h, :patch_w], label[..., :patch_h, -patch_w:]], dim=-1)
    bottom = torch.cat([label[..., -patch_h:, :patch_w], label[..., -patch_h:, -patch_w:]], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def make_online_crop_pair(raw_lf, source_ang_res, angout, angin=2, crop_policy='center_floor'):
    # 训练样本的核心转换：原始光场 -> (2x2 输入 data, angout x angout 标签)。
    #
    #     组合 crop_lf_sai 与 build_input_from_cropped_label。因为标签的四角就是
    #     输入的四角，训练损失会用 mask 排除这四个角，只监督中间缺失视角。
    label = crop_lf_sai(raw_lf, source_ang_res, angout, crop_policy=crop_policy)
    data = build_input_from_cropped_label(label, angout, angin=angin)
    return data, label


def missing_view_l1_loss(pred, target, angout):
    # 只计算“缺失视角”的平均绝对误差（L1）。
    #
    #     pred/target 是拼接 SAI，先由 missing_view_lf_tensors 拆成 [B,C,U,V,H,W]
    #     并生成四角为 False 的 mask，再只取 mask 为 True 的视角做 L1。
    #     四角被网络原样拷贝，若也计入损失会得到虚假的零误差，掩盖中间视角的质量。
    pred_lf, target_lf, mask = missing_view_lf_tensors(pred, target, angout)
    return torch.nn.functional.l1_loss(pred_lf[:, :, mask], target_lf[:, :, mask])


def missing_view_lf_tensors(pred, target, angout):
    # 把 SAI 拆成光场视角张量，并返回“缺失视角”布尔掩码。
    #
    #     输入 [B,C,angout*H,angout*W] 先 reshape 成
    #     [B,C,U,H,V,W]，再 permute 成 [B,C,U,V,H,W]。
    #     返回 mask[U,V]：四角为 False，其余为 True。
    #     同时校验 pred/target 形状一致、空间尺寸能被 angout 整除。
    if pred.shape != target.shape:
        raise ValueError('pred and target must have the same shape')
    if pred.shape[-2] % angout != 0 or pred.shape[-1] % angout != 0:
        raise ValueError('pred spatial shape must be divisible by angout')

    patch_h = pred.shape[-2] // angout
    patch_w = pred.shape[-1] // angout
    pred_lf = pred.reshape(
        pred.shape[0], pred.shape[1], angout, patch_h, angout, patch_w
    ).permute(0, 1, 2, 4, 3, 5)
    target_lf = target.reshape(
        target.shape[0], target.shape[1], angout, patch_h, angout, patch_w
    ).permute(0, 1, 2, 4, 3, 5)

    mask = torch.ones((angout, angout), device=pred.device, dtype=torch.bool)
    mask[0, 0] = False
    mask[0, -1] = False
    mask[-1, 0] = False
    mask[-1, -1] = False
    return pred_lf, target_lf, mask


def missing_view_l1_loss_with_weights(pred, target, angout, weights):
    # 带逐像素权重的缺失视角 L1（备用损失，当前主循环未调用）。
    #
    #     权重与 pred 同形状，先拆成视角张量再取同一 mask；最终用
    #     sum(w*|err|)/sum(w) 归一化，避免权重整体尺度改变损失量级。
    pred_lf, target_lf, mask = missing_view_lf_tensors(pred, target, angout)
    weight_lf, _, _ = missing_view_lf_tensors(weights, weights, angout)
    abs_error = torch.abs(pred_lf[:, :, mask] - target_lf[:, :, mask])
    masked_weights = weight_lf[:, :, mask]
    return torch.sum(masked_weights * abs_error) / torch.clamp(torch.sum(masked_weights), min=1e-12)


def sobel_xy(x):
    # 逐通道计算 Sobel 水平/垂直一阶梯度。
    #
    #     x 形状 [N,C,H,W]。两个 3x3 卷积核分别复制成 groups=C 的 depthwise 卷积，
    #     因此各通道独立、不跨通道混合。padding=1 保持空间尺寸不变。
    #     返回 grad_x, grad_y，形状均与 x 相同。
    channels = x.shape[1]
    sobel_x = x.new_tensor([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    sobel_y = x.new_tensor([
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ]).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    grad_x = torch.nn.functional.conv2d(x, sobel_x, padding=1, groups=channels)
    grad_y = torch.nn.functional.conv2d(x, sobel_y, padding=1, groups=channels)
    return grad_x, grad_y


def sobel_edge_map(x):
    # Sobel 梯度幅值 sqrt(gx^2+gy^2)，加 1e-12 防止 sqrt 在零梯度处求导出 NaN。
    grad_x, grad_y = sobel_xy(x)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def missing_view_sobel_gradient_loss(pred, target, angout):
    # 在缺失视角上比较预测与标签的 Sobel 梯度（Smooth L1）。
    #
    #     动机：单纯 L1 容易把边缘训得偏软；约束 gx/gy 可让预测保留与 GT 一致的
    #     局部边缘方向和纹理变化。流程：
    #       1. 拆视角并取缺失视角，展平成 [N,C,H,W]；
    #       2. 对预测/标签分别算 Sobel gx、gy；
    #       3. 裁掉最外一圈像素，避免 padding 造成的边界伪梯度；
    #       4. 对 gx、gy 各算 smooth_l1_loss，取平均。
    pred_lf, target_lf, mask = missing_view_lf_tensors(pred, target, angout)
    pred_views = pred_lf[:, :, mask].reshape(
        -1, pred.shape[1], pred_lf.shape[-2], pred_lf.shape[-1]
    )
    target_views = target_lf[:, :, mask].reshape(
        -1, target.shape[1], target_lf.shape[-2], target_lf.shape[-1]
    )

    pred_grad_x, pred_grad_y = sobel_xy(pred_views)
    target_grad_x, target_grad_y = sobel_xy(target_views)

    pred_grad_x = pred_grad_x[..., 1:-1, 1:-1]
    pred_grad_y = pred_grad_y[..., 1:-1, 1:-1]
    target_grad_x = target_grad_x[..., 1:-1, 1:-1]
    target_grad_y = target_grad_y[..., 1:-1, 1:-1]

    grad_x_loss = torch.nn.functional.smooth_l1_loss(pred_grad_x, target_grad_x)
    grad_y_loss = torch.nn.functional.smooth_l1_loss(pred_grad_y, target_grad_y)
    return 0.5 * (grad_x_loss + grad_y_loss)


def edge_weighted_missing_view_l1_loss(pred, target, angout, alpha=0.0):
    # 边缘加权 L1（备用损失，当前主训练循环未调用）。
    #
    #     先用 GT 的 Sobel 梯度幅值构造逐像素权重 1 + alpha*edge：
    #       - 每个视角内部按自身最大梯度归一化，避免不同视角亮度/对比度影响权重；
    #       - alpha=0 时退化为普通缺失视角 L1，并返回全 1 的统计量。
    #     返回 (loss, stats)，stats 仅用于日志观察权重范围。
    if alpha == 0.0:
        plain_loss = missing_view_l1_loss(pred, target, angout)
        one = plain_loss.new_tensor(1.0)
        return plain_loss, {'weight_mean': one, 'weight_max': one}

    pred_lf, target_lf, mask = missing_view_lf_tensors(pred, target, angout)
    missing_view_count = int(mask.sum().item())
    target_views = target_lf[:, :, mask].reshape(-1, target.shape[1], target_lf.shape[-2], target_lf.shape[-1])
    edge = sobel_edge_map(target_views)
    edge_max = torch.amax(edge.reshape(edge.shape[0], -1), dim=1).view(-1, 1, 1, 1)
    edge = edge / torch.clamp(edge_max, min=1e-12)
    weights = (1.0 + alpha * edge).view(
        target.shape[0],
        target.shape[1],
        missing_view_count,
        target_lf.shape[-2],
        target_lf.shape[-1]
    )
    abs_error = torch.abs(pred_lf[:, :, mask] - target_lf[:, :, mask])
    loss = torch.sum(weights * abs_error) / torch.clamp(torch.sum(weights), min=1e-12)
    return loss, {
        'weight_mean': weights.detach().mean(),
        'weight_max': weights.detach().max()
    }


def LFdivide(data, angRes, patch_size, stride):
    # 把整幅 SAI 切成带重叠的空间 patch，供验证时逐块推理。
    #
    #     参数：
    #       data       -- [angRes*H, angRes*W] 的二维 SAI；
    #       angRes     -- 角度边长（测试时通常等于输入/输出角度，块内保持所有视角）；
    #       patch_size -- 每个视角内切出的空间块边长（像素）；
    #       stride     -- 相邻块步长（像素），bdr=(patch_size-stride)//2 为重叠半边。
    #     返回 subLF：[numU,numV,angRes*patch_size,angRes*patch_size]。
    #     每个 subLF 把同一空间位置的 angRes x angRes 个视角纵向堆叠，正好是网络
    #     期望的 SAI 输入布局。先对每个视角做镜像边界扩展，再按滑动窗口取块，
    #     因此图像边缘块也有完整上下文；块间重叠由 LFintegrate 取中心有效区消除。
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes, device=data.device, dtype=data.dtype)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes, device=data.device, dtype=data.dtype)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    # 对单张视角图做镜像 padding，给分块推理提供边界上下文。
    #
    #     构造 3x3 的镜像拼贴：中间为原图，左右为水平翻转，上下为垂直翻转，
    #     四角为双向翻转；然后从拼贴中心裁出 [h+2*bdr, w+2*bdr]。
    #     相比补零，镜像不会在图像边缘引入强人工梯度。
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])
    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    # 把 LFdivide 切出的重叠 patch 结果拼回完整光场。
    #
    #     subLF 形状 [numU,numV,angRes*pz,angRes*pz]，其中每个 patch 的外圈
    #     bdr=(pz-stride)//2 是推理时的重叠上下文。这里对每个视角只取中心
    #     stride x stride 的有效区写回 temp，重叠区不会被重复平均；最后裁到
    #     原始空间尺寸 [angRes,h0,w0]。
    numU, numV, pH, pW = subLF.shape
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV, device=subLF.device, dtype=subLF.dtype)
    outLF = torch.zeros(angRes, angRes, h0, w0, device=subLF.device, dtype=subLF.dtype)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]
            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def cal_psnr(img1, img2):
    # 用 skimage 计算单张图 PSNR；完全相同时返回 inf，避免 log10(0) 警告。
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()
    if np.mean((img1_np - img2_np) ** 2) == 0:
        return float('inf')

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)

def cal_ssim(img1, img2):
    # 用 skimage 计算 SSIM，采用高斯窗（sigma=1.5）且不做样本协方差校正。
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    out = metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    return out

def cal_metrics_RE(img1, img2, angRes_in, angRes_out):
    # 计算光场缺视角的平均 PSNR/SSIM（当前验证主指标）。
    #
    #     输入可以是二维 SAI [U*H,V*W]，也可以是已拆好的四维张量 [U,V,H,W]。
    #     评测约定：
    #       - 只统计四个角以外的视角（角点由输入硬拷贝，不反映生成能力）；
    #       - 每个视角裁掉四周 22 像素，避免分块/镜像 padding 的边缘影响；
    #       - 对有效视角取算术平均。
    #     angRes_in 参数当前保留用于接口兼容，实际视角网格由 angRes_out 决定。
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    [U2, V2, h, w] = img2.size()
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    valid_mask = np.ones(shape=(U, V), dtype='bool')
    valid_mask[0, 0] = False
    valid_mask[0, V - 1] = False
    valid_mask[U - 1, 0] = False
    valid_mask[U - 1, V - 1] = False
    bd = 22
    for u in range(U):
        for v in range(V):
            if not valid_mask[u, v]:
                continue
            PSNR[u, v] = cal_psnr(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
            SSIM[u, v] = cal_ssim(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
            pass
        pass

    psnr_mean = PSNR[valid_mask].mean()
    ssim_mean = SSIM[valid_mask].mean()

    return psnr_mean, ssim_mean
