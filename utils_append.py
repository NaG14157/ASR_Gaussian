import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics

class TrainSetLoaderRawLF(Dataset):
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
    label = crop_lf_sai(raw_lf, source_ang_res, angout, crop_policy=crop_policy)
    data = build_input_from_cropped_label(label, angout, angin=angin)
    return data, label


def missing_view_l1_loss(pred, target, angout):
    pred_lf, target_lf, mask = missing_view_lf_tensors(pred, target, angout)
    return torch.nn.functional.l1_loss(pred_lf[:, :, mask], target_lf[:, :, mask])


def missing_view_lf_tensors(pred, target, angout):
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
    pred_lf, target_lf, mask = missing_view_lf_tensors(pred, target, angout)
    weight_lf, _, _ = missing_view_lf_tensors(weights, weights, angout)
    abs_error = torch.abs(pred_lf[:, :, mask] - target_lf[:, :, mask])
    masked_weights = weight_lf[:, :, mask]
    return torch.sum(masked_weights * abs_error) / torch.clamp(torch.sum(masked_weights), min=1e-12)


def sobel_xy(x):
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
    grad_x, grad_y = sobel_xy(x)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)


def missing_view_sobel_gradient_loss(pred, target, angout):
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
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()
    if np.mean((img1_np - img2_np) ** 2) == 0:
        return float('inf')

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    out = metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    return out

def cal_metrics_RE(img1, img2, angRes_in, angRes_out):
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
