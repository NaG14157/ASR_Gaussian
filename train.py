# =============================================================================
# ASR_Gaussian 训练 / 验证入口
#
# 本文件负责把“数据 -> 网络 -> 损失 -> 反传 -> 存储 -> 评测”串起来，网络结构
# 本身在 model_crossAttention_anyAng_HCI.py，数据裁切/损失/指标在
# utils_append.py，训练模式策略在 training_policy.py。
#
# 一次训练 batch 的主链路：
#   原始 9x9 光场 HDF5
#     -> crop_lf_sai：居中裁出本轮 angout x angout 标签
#     -> build_input_from_cropped_label：取标签四角拼成 2x2 输入
#     -> 同步数据增强（仅固定尺度模式）
#     -> Net_CrossAttention：
#          特征提取 -> 四体积构建 -> Transformer 交互
#          -> 坐标条件角度场 -> 高斯参数解码 -> 高斯泼溅 -> SAI 拼装
#     -> 缺失视角 L1 + Sobel 梯度损失
#     -> Adam 更新
# 验证时：SAI 分块 -> 逐块前向 -> 拼回整图 -> 缺视角 PSNR/SSIM/诊断 CSV。
# =============================================================================
import time
import argparse
import sys
import os
import csv
import shutil
import math

# 注意：CUDA_VISIBLE_DEVICES 必须在 import torch 之前设置才生效。
# 这里保留一个默认值，随后 apply_cuda_visible_devices_from_argv() 会允许
# 命令行 --cuda_visible_devices 覆盖它。
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
HALVE_RESUME_LR = False
TRAIN_PROGRESS_NCOLS_RATIO = 0.75

def get_train_progress_ncols():
    # 按终端宽度计算 tqdm 进度条宽度，至少 60 列，避免窄终端换行刷屏。
    terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
    return max(60, int(terminal_width * TRAIN_PROGRESS_NCOLS_RATIO))


def apply_cuda_visible_devices_from_argv():
    # 在 import torch 之前预解析 --cuda_visible_devices。
    #
    #     正式参数解析在 parse_args()，但 CUDA 设备可见性必须在初始化 CUDA 前写入
    #     环境变量，所以这里先扫一遍 sys.argv；值非空时才覆盖默认的 '3'。
    if "--cuda_visible_devices" not in sys.argv:
        return
    index = sys.argv.index("--cuda_visible_devices")
    if index + 1 < len(sys.argv):
        value = sys.argv[index + 1]
        if value != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = value


apply_cuda_visible_devices_from_argv()

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from checkpoint_utils import build_training_checkpoint, filter_pretrain_state_dict, restore_scheduler_state, str2bool
from data_augmentation import augmentation_with_resize
from model_crossAttention_anyAng_HCI import Net_CrossAttention
from utils_append import (
    LFdivide,
    LFintegrate,
    MultiTestSetDataLoader,
    TrainSetLoaderRawLF,
    cal_metrics_RE,
    cal_psnr,
    cal_ssim,
    make_online_crop_pair,
    missing_view_l1_loss,
    missing_view_sobel_gradient_loss,
)
from training_policy import (
    ARBITRARY_SCALE_MODE,
    TEST_DATA_MODE,
    TRAIN_DATA_MODE,
    apply_training_augmentation,
    resolve_training_angout,
    training_augmentation_enabled,
    training_mode_name,
    training_scale_tag,
    validate_training_policy,
)
# from model_crossAttention_anyAng_Lytro import Net_CrossAttention

from tensorboardX import SummaryWriter

# Settings
def parse_args():
    # 定义并解析命令行参数。
    #
    #     参数可分成六组：
    #       1. 角度/数据协议：angin、angout、train_mode、source_ang_res、angout_min/max；
    #       2. 数据路径与 HDF5 键：trainset_dir、testset_dir、raw_lf_key、crop_policy；
    #       3. 优化器：lr、n_epochs、n_steps、gamma、batch_size；
    #       4. 损失：grad_l1_lambda 及 warmup/ramp；
    #       5. 验证与输出：patchsize/stride、保存图片/CSV、诊断开关、tag/model_path；
    #       6. 模型/解码器：模板、偏移上限、自适应偏移、置信门控。
    #     训练数据是原始 9x9 光场在线裁切，测试数据是预先切好的固定配对。
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=2, help="angular resolution")
    parser.add_argument("--angout", type=int, default=7, help="angular resolution")
    # 1 = 任意尺度预训练（随机 angout、无增强）；2 = 固定尺度微调（固定 angout、开增强）
    parser.add_argument("--train_mode", type=int, default=2, choices=[1, 2],
                        help="1: arbitrary-scale training; 2: fixed-scale fine-tuning")
    parser.add_argument("--source_ang_res", type=int, default=9, help="source angular resolution in raw LF labels")
    parser.add_argument("--angout_min", type=int, default=7, help="minimum sampled output angular resolution")
    parser.add_argument("--angout_max", type=int, default=7, help="maximum sampled output angular resolution")
    parser.add_argument("--eval_angouts", type=str, default="7", help="comma-separated eval angular resolutions; defaults to --angout")
    parser.add_argument("--raw_lf_key", type=str, default="label", help="h5 key for raw full-angular LF SAI")
    parser.add_argument("--crop_policy", type=str, default="center_floor", help="online angular crop policy")
    parser.add_argument("--cuda_visible_devices", type=str, default="", help="optional CUDA_VISIBLE_DEVICES override")
    parser.add_argument("--upscale_factor", type=int, default=1, help="upscale factor")
    parser.add_argument('--model_name', type=str, default='GILF_ASR')
    #训练集目录（要根据实际情况修改）
    parser.add_argument('--trainset_dir', type=str, default='/home/vision/work1/ywj/data/ASR_data/train/HCI/TrainData_HCI_2x2_9x9_64')
   #测试集目录（要根据实际情况修改）
    parser.add_argument('--testset_dir', type=str, default='/home/vision/work1/ywj/data/ASR_data/test/HCI/test_2x2_sx1SR_7x7')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--grad_l1_lambda', type=float, default=0.05,
                        help='weight for per-view missing-view Sobel gx/gy smooth L1 loss; 0 disables it')
    parser.add_argument('--grad_l1_warmup_epochs', type=int, default=30,
                        help='number of initial epochs before Sobel gradient loss ramps in')
    parser.add_argument('--grad_l1_ramp_epochs', type=int, default=20,
                        help='number of epochs used to linearly ramp Sobel gradient loss weight')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    #多少个epoch更新一次学习率
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument("--stride", type=int, default=32, help="stride for patch cropping")
    #是否加载模型（没有模型时写False，已有模型写True）
    parser.add_argument('--load_pretrain', type=str2bool, default=True)
    #训练完后模型的路径（要根据实际情况修改）
    parser.add_argument('--model_path', type=str, default='/home/vision/work1/ywj/traiplane/save/gs_checkpoint_ASR_R16_HCI_σ/GILF_ASR_1xSR_2x2_fixed7x7_epoch_42.pth.tar')
    parser.add_argument('--weights_only_pretrain', type=str2bool, default=False,
                        help='load all compatible model weights but never restore optimizer/scheduler state; '
                             'this is the default, pass false to resume optimizer and scheduler state')
    #测试集的标签，每次训练一个模型时都要改变
    parser.add_argument('--tag', type=str, default='ASR_R16_HCI_σ_test')
    parser.add_argument('--save_test_images', type=str2bool, default=True,
                        help='save generated LF image and GT during validation')
    parser.add_argument('--save_test_image_limit', type=int, default=-1,
                        help='max saved validation samples per dataset/angout/epoch; -1 means save all')
    parser.add_argument('--save_test_metrics', type=str2bool, default=True,
                        help='save per-sample validation PSNR/SSIM metrics to CSV')
    parser.add_argument('--print_test_sample_metrics', type=str2bool, default=True,
                        help='print per-sample validation PSNR/SSIM metrics')
    parser.add_argument('--save_test_diagnostics', type=str2bool, default=True,
                        help='save per-view PSNR/SSIM and Gaussian offset diagnostics during validation')
    parser.add_argument('--print_test_diagnostics', type=str2bool, default=True,
                        help='print per-sample Gaussian diagnostic summary during validation')
    # 消融实验：把指定体积支路置零，仅用于对比 uvs/uvt/ust/vst 的贡献
    parser.add_argument('--ablate_planes', type=str, default='',
                        help='comma/space-separated volumes for branch-off ablation during forward passes, e.g. uvs, uvt, ust, vst, or uvs,vst')
    parser.add_argument('--use_volume_template', type=str2bool, default=True,
                        help='enable learnable fixed-size templates added to uvs/uvt/ust/vst after encoding')
    parser.add_argument('--max_offset_px', type=float, default=12.0,
                        help='fallback maximum free Gaussian center offset in pixels')
    parser.add_argument('--use_adaptive_offset', type=str2bool, default=True,
                        help='enable angular-distance adaptive Gaussian offset limit')
    parser.add_argument('--adaptive_offset_min_px', type=float, default=8.0,
                        help='minimum Gaussian offset limit in pixels for near anchor-target pairs')
    parser.add_argument('--adaptive_offset_max_px', type=float, default=16.0,
                        help='maximum Gaussian offset limit in pixels for far anchor-target pairs')
    parser.add_argument('--use_confidence_gate', type=str2bool, default=True,
                        help='enable per-anchor confidence gate multiplied into Gaussian weights')
    parser.add_argument('--confidence_gate_blend', type=float, default=1.0,
                        help='blend confidence gate with uniform anchor weights; 1 uses learned gate, 0 disables its effect')
    parser.add_argument('--confidence_temperature', type=float, default=0.7,
                        help='softmax temperature for confidence gate; lower makes anchor weights sharper')


    return parser.parse_args()

def parse_ablate_planes(value):
    # 把 --ablate_planes 的逗号/空格分隔字符串解析成集合。
    #
    #     合法名称固定为 {'uvs','uvt','ust','vst'}；出现未知名称直接报错，防止
    #     消融实验因拼写错误而实际没有关闭任何支路。
    if value is None:
        return set()
    items = str(value).replace(',', ' ').split()
    valid_planes = {'uvs', 'uvt', 'ust', 'vst'}
    planes = {item.strip().lower() for item in items if item.strip()}
    invalid_planes = planes - valid_planes
    if invalid_planes:
        raise ValueError("invalid --ablate_planes {}. Expected subset of {}".format(
            sorted(invalid_planes), sorted(valid_planes)))
    return planes


def get_plain_net(net):
    # 剥掉 DataParallel 外壳，返回真正的 Net_CrossAttention。
    #
    #     访问自定义属性（如 epiFeatureRebuild、current_gaussian_stats）时必须用
    #     未包装的模型，否则属性会落在 DataParallel 层而不是内部模块上。
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def configure_plane_ablation(net, planes):
    # 把消融支路集合写入高斯解码器，forward 时据此零化对应体积。
    plain_net = get_plain_net(net)
    decoder = plain_net.epiFeatureRebuild.implicit_decoder
    decoder.ablate_planes = set(planes)
    print("Volume branch-off ablation active: {}".format(sorted(decoder.ablate_planes)))


def train(cfg, train_loader, test_Names, test_loaders):
    # 完整训练流程：建模 -> 恢复权重 -> 训练前验证 -> 逐 epoch 优化/记录/存档。
    #
    #     参数：
    #       cfg          -- parse_args() 得到的配置；
    #       train_loader -- 原始光场 Dataset 的 DataLoader，batch 形状 [B,1,9H,9W]；
    #       test_Names   -- 测试子目录名列表；
    #       test_loaders -- 与 test_Names 一一对应的固定测试 DataLoader。
    # -------------------------------------------------------------------------
    # 1) 建模：特征提取 + 四体积重建 + 高斯解码器都在 Net_CrossAttention 内组装。
    #    DataParallel 只影响并行执行，下面的自定义属性访问统一先剥壳。
    # -------------------------------------------------------------------------
    net = Net_CrossAttention(
        cfg.angin,
        cfg.angout,
        use_volume_template=cfg.use_volume_template,
        max_offset_px=cfg.max_offset_px,
        use_adaptive_offset=cfg.use_adaptive_offset,
        adaptive_offset_min_px=cfg.adaptive_offset_min_px,
        adaptive_offset_max_px=cfg.adaptive_offset_max_px,
        use_confidence_gate=cfg.use_confidence_gate,
        confidence_gate_blend=cfg.confidence_gate_blend,
        confidence_temperature=cfg.confidence_temperature
    )
    # net = torch.nn.DataParallel(net.to(cfg.device))
    if torch.cuda.device_count() > 1:
        print("Lets use", torch.cuda.device_count(), 'GPUs!')
        net = torch.nn.DataParallel(net.to(cfg.device))
    net.to(cfg.device)
    cudnn.benchmark = True
    epoch_state = 0
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 19, 34, 49, 64, 79], gamma=cfg.gamma)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 70, 80, 85, 90, 95, 100], gamma=cfg.gamma)

    # 消融配置必须在任何前向之前写入解码器。
    ablate_planes = parse_ablate_planes(cfg.ablate_planes)
    configure_plane_ablation(net, ablate_planes)

    # -------------------------------------------------------------------------
    # 2) 恢复预训练/断点。兼容加载策略：
    #    filter_pretrain_state_dict 只保留键名和形状都匹配的参数；
    #    只有“没有新增 confidence_net 缺失”且未指定 weights-only 时，
    #    才恢复 optimizer/scheduler/epoch，否则从 epoch 0 重新开始。
    # -------------------------------------------------------------------------
    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            # net.load_state_dict(model['state_dict'])
            state_dict = model['state_dict']
            load_target = net.module if isinstance(net, torch.nn.DataParallel) else net
            current_state = load_target.state_dict()
            filtered_state, skipped_keys = filter_pretrain_state_dict(
                state_dict,
                current_state
            )
            missing_keys, unexpected_keys = load_target.load_state_dict(filtered_state, strict=False)
            print("loaded compatible pretrain weights including decoder: loaded {}, skipped {}".format(
                len(filtered_state), len(skipped_keys)
            ))
            if missing_keys:
                print("missing keys after compatible load: {}".format(missing_keys[:20]))
            if unexpected_keys:
                print("unexpected keys after compatible load: {}".format(unexpected_keys[:20]))
            missing_confidence_net = any('confidence_net' in key for key in missing_keys)
            can_resume_optimizer = (
                (not missing_confidence_net)
                and (not cfg.weights_only_pretrain)
            )
            if 'optimizer' in model and can_resume_optimizer:
                optimizer.load_state_dict(model['optimizer'])
                if HALVE_RESUME_LR:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    print("halve resume optimizer lr")
                epoch_state = model.get("epoch", 0)
                if restore_scheduler_state(scheduler, model, epoch_state):
                    print("resume scheduler state")
                else:
                    print("no scheduler state in checkpoint; align scheduler last_epoch to {}".format(epoch_state))
                print("resume training at epoch {}".format(epoch_state))
            else:
                epoch_state = 0
                if cfg.weights_only_pretrain:
                    print("weights-only pretrain: loaded model weights, skipped optimizer and scheduler, start training from epoch 0")
                elif missing_confidence_net:
                    print("old checkpoint has no confidence_net optimizer state; load weights only and start training from epoch 0")
                else:
                    print("load pre-trained weights only, start training from epoch 0")
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    # net = torch.nn.DataParallel(net, device_ids= list(eval(cfg.device_ids)) )

    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    loss_epoch = []
    base_l1_epoch = []
    grad_loss_epoch = []
    grad_lambda_epoch = []
    volume_ratio_epoch = []
    loss_list = []
    # -------------------------------------------------------------------------
    # 3) 训练前验证：用恢复后的初始权重先跑一遍测试集，记录起始 PSNR/SSIM。
    #    当前逐 epoch 验证代码在下方被注释，因此训练过程中不会自动刷新指标。
    # -------------------------------------------------------------------------
    #验证集（在训练模型前要注释掉这一段）
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        num_testset = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_loaders[index]
            psnr_epoch_test, ssim_epoch_test, num_epoch_test = valid(test_loader, net, cfg.angout, test_name, epoch_state)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            num_testset.append(num_epoch_test)

            print(time.ctime()[4:-5] + ' Valid----%15s,\t test Number---%d, PSNR---%f, SSIM---%f' % (
            test_name, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
            txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
            txtfile.write('Dataset----%10s,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (
            test_name, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
            txtfile.close()

            # writer.add_scalars('psnr', {test_name:psnr_epoch_test} ,idx_epoch)
            # writer.add_scalars('ssim', {test_name:ssim_epoch_test} ,idx_epoch)
            # tensorboard

            pass

    # -----------------------------------------------------------------------------
    # 4) 主训练循环。每个 batch 现场决定 angout，并完成
    #    原始光场 -> 标签 -> 2x2 输入 -> 增强 -> 前向 -> 缺视角损失。
    # -----------------------------------------------------------------------------
    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader), ncols=get_train_progress_ncols()):
            raw_lf = batch
            # 固定尺度模式返回 cfg.angout；任意尺度模式在 [min,max] 随机采样。
            angout = resolve_training_angout(
                train_mode=cfg.train_mode,
                angout=cfg.angout,
                angout_min=cfg.angout_min,
                angout_max=cfg.angout_max,
                source_ang_res=cfg.source_ang_res,
            )
            # 标签 = 原始光场中心裁出的 angout x angout；
            # 输入 = 标签四个角拼成的 2x2 SAI，空间分辨率不变。
            data, label = make_online_crop_pair(
                raw_lf,
                cfg.source_ang_res,
                angout,
                angin=cfg.angin,
                crop_policy=cfg.crop_policy
            )
            data, label = apply_training_augmentation(
                cfg.train_mode,
                data,
                label,
                augmentation_with_resize,
            )
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            # 前向：内部依次经过特征提取、四体积构建、Transformer、场合成、
            # 高斯解码与泼溅，返回完整 angout x angout SAI（四角已硬拷贝回输入）。
            out = net(data, angout).to(cfg.device)
            # print(out.shape)
            # print(label.shape)
            # 基础监督只覆盖缺失视角（四角被 mask 排除）。
            base_l1 = missing_view_l1_loss(out, label, angout)

            # Sobel 梯度损失权重：前 warmup 个 epoch 为 0，随后线性 ramp 到
            # grad_l1_lambda；权重为 0 时不算梯度图，直接构造同设备零标量。
            if idx_epoch < cfg.grad_l1_warmup_epochs or cfg.grad_l1_lambda <= 0.0:
                grad_lambda_current = 0.0
            elif cfg.grad_l1_ramp_epochs <= 0:
                grad_lambda_current = cfg.grad_l1_lambda
            else:
                ramp_progress = float(idx_epoch - cfg.grad_l1_warmup_epochs + 1) / float(cfg.grad_l1_ramp_epochs)
                grad_lambda_current = cfg.grad_l1_lambda * min(1.0, max(0.0, ramp_progress))

            if grad_lambda_current > 0.0:
                grad_loss = missing_view_sobel_gradient_loss(out, label, angout)
            else:
                grad_loss = base_l1.detach().new_tensor(0.0)
            loss = base_l1 + grad_lambda_current * grad_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())
            base_l1_epoch.append(base_l1.data.cpu())
            grad_loss_epoch.append(grad_loss.data.cpu())
            grad_lambda_epoch.append(loss.detach().new_tensor(grad_lambda_current).data.cpu())
            decoder = get_plain_net(net).epiFeatureRebuild.implicit_decoder
            volume_ratio_epoch.append(decoder.last_volume_ratios.detach().cpu())

        # -------------------------------------------------------------------------
        # 5) epoch 汇总：平均各项损失、四体积的实际贡献比例，写日志/TensorBoard，
        #    推进学习率，并保存一个包含完整训练状态的 checkpoint。
        # -------------------------------------------------------------------------
        if idx_epoch % 1 == 0:
            loss_mean = float(np.array(loss_epoch).mean())
            base_l1_mean = float(np.array(base_l1_epoch).mean())
            grad_loss_mean = float(np.array(grad_loss_epoch).mean())
            grad_lambda_mean = float(np.array(grad_lambda_epoch).mean())
            if volume_ratio_epoch:
                volume_ratio_mean = torch.stack(volume_ratio_epoch).mean(dim=0)
            else:
                volume_ratio_mean = torch.tensor([0.25, 0.25, 0.25, 0.25])
            vol_uvs_mean, vol_uvt_mean, vol_ust_mean, vol_vst_mean = [float(v) for v in volume_ratio_mean]
            loss_list.append(loss_mean)
            log_line = (
                time.ctime()[4:-5] +
                ' Epoch----%5d, total_loss---%f, base_l1---%f, grad_loss---%f, grad_lambda---%f, vol_uvs---%.4f, vol_uvt---%.4f, vol_ust---%.4f, vol_vst---%.4f'
                % (
                    idx_epoch + 1,
                    loss_mean,
                    base_l1_mean,
                    grad_loss_mean,
                    grad_lambda_mean,
                    vol_uvs_mean,
                    vol_uvt_mean,
                    vol_ust_mean,
                    vol_vst_mean
                )
            )
            print(log_line)
            txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
            txtfile.write(log_line + '\n')
            txtfile.close()
            writer.add_scalars('loss', {
                'total': loss_mean,
                'base_l1': base_l1_mean,
                'grad': grad_loss_mean
            }, idx_epoch)
            writer.add_scalars('grad_l1', {
                'lambda': grad_lambda_mean,
                'loss': grad_loss_mean
            }, idx_epoch)
            writer.add_scalars('volume_gate', {
                'uvs': vol_uvs_mean,
                'uvt': vol_uvt_mean,
                'ust': vol_ust_mean,
                'vst': vol_vst_mean
            }, idx_epoch)
            writer.flush()
            scheduler.step()
            save_ckpt(build_training_checkpoint(
                epoch=idx_epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler,
                net=net,
                loss_list=loss_list,
            ),
                save_path=savepath, filename=cfg.model_name + '_' + str(cfg.upscale_factor) + 'xSR_' +
                            str(cfg.angin) + 'x' + str(cfg.angin) + '_' + training_scale_tag(
                                cfg.train_mode, cfg.angout, cfg.angout_min, cfg.angout_max
                            ) + '_epoch_' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []
            base_l1_epoch = []
            grad_loss_epoch = []
            grad_lambda_epoch = []
            volume_ratio_epoch = []

        # ''' evaluation '''
        # with torch.no_grad():
        #     for eval_angout in cfg.eval_angouts_list:
        #         psnr_testset = []
        #         ssim_testset = []
        #         num_testset = []
        #         for index, test_name in enumerate(test_Names):
        #             test_loader = test_loaders[index]
        #             psnr_epoch_test, ssim_epoch_test, num_epoch_test = valid(test_loader, net, eval_angout, test_name, idx_epoch + 1)
        #             psnr_testset.append(psnr_epoch_test)
        #             ssim_testset.append(ssim_epoch_test)
        #             num_testset.append(num_epoch_test)
        #
        #             print(time.ctime()[4:-5] + ' Valid----%15s,\t AngOut---%d,\t test Number---%d, PSNR---%f, SSIM---%f' % (test_name, eval_angout, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
        #             txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
        #             txtfile.write('Dataset----%10s,\t AngOut---%d,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (test_name, eval_angout, num_epoch_test, psnr_epoch_test, ssim_epoch_test))
        #             txtfile.close()
        #
        #             pass
        #         psnr_avg = sum([psnr_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)
        #         ssim_avg = sum([ssim_testset[ii]*num_testset[ii] for ii in range(len(num_testset))]) / sum(num_testset)
        #
        #         txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
        #         txtfile.write('Total testset,\t AngOut---%d,\t test Number---%d ,\t PSNR---%f,\t SSIM---%f\n' % (eval_angout, sum(num_testset), psnr_avg, ssim_avg))
        #         txtfile.close()
        #         writer.add_scalars('psnr', {'testset_%dx%d' % (eval_angout, eval_angout):psnr_avg}, idx_epoch)
        #         writer.add_scalars('ssim', {'testset_%dx%d' % (eval_angout, eval_angout):ssim_avg}, idx_epoch)
        #
        #     writer.flush()
        #     pass


        pass

GAUSSIAN_DIAGNOSTIC_MEAN_FIELDS = [
    'delta_px_mean',
    'delta_px_p95',
    'delta_s_px_mean',
    'delta_t_px_mean',
    'sigma_s_mean',
    'sigma_t_mean',
    'opacity_mean',
    'offset_limit_px_mean',
    'confidence_mean',
    'confidence_max_mean',
    'confidence_entropy',
    'confidence_anchor_00_mean',
    'confidence_anchor_01_mean',
    'confidence_anchor_10_mean',
    'confidence_anchor_11_mean',
    'tanh_sat_ratio',
    'tanh_sat_s_ratio',
    'tanh_sat_t_ratio',
    'center_clamp_ratio',
]

GAUSSIAN_DIAGNOSTIC_MAX_FIELDS = [
    'delta_px_max',
    'delta_s_px_max',
    'delta_t_px_max',
    'sigma_s_max',
    'sigma_t_max',
    'opacity_max',
    'offset_limit_px_max',
]


def get_gaussian_decoder(net):
    # 定位高斯解码器：Net -> FourVolumeFeatureRebuild -> implicit_decoder。
    return get_plain_net(net).epiFeatureRebuild.implicit_decoder


def set_gaussian_diagnostics_enabled(net, enabled):
    # 开关解码器的高斯统计收集，并清空上次残留记录。
    #
    #     训练/普通推理时关闭可避免额外的统计开销；验证需要 CSV 诊断时临时打开，
    #     返回旧值以便调用结束后恢复。
    decoder = get_gaussian_decoder(net)
    previous = getattr(decoder, 'enable_gaussian_stats', False)
    decoder.enable_gaussian_stats = bool(enabled)
    decoder.current_gaussian_stats = []
    return previous


def update_gaussian_diagnostic_accumulator(accumulator, records):
    # 把一张测试图多个 patch 的高斯统计按视角累加。
    #
    #     均值类字段先存“值 x 高斯数”的加权和，最后除以总 count；最大值类字段
    #     直接逐 patch 取 max。这样分块推理不会把不同 patch 的统计简单平均而失真。
    for record in records:
        view_u = int(record.get('view_u', -1))
        view_v = int(record.get('view_v', -1))
        if view_u < 0 or view_v < 0:
            continue
        key = (view_u, view_v)
        item = accumulator.setdefault(key, {'view_u': view_u, 'view_v': view_v, 'count': 0, 'patches': 0})
        count = int(record.get('count', 0))
        if count <= 0:
            continue
        item['count'] += count
        item['patches'] += 1
        for field in GAUSSIAN_DIAGNOSTIC_MEAN_FIELDS:
            item[field + '_weighted_sum'] = item.get(field + '_weighted_sum', 0.0) + float(record.get(field, 0.0)) * count
        for field in GAUSSIAN_DIAGNOSTIC_MAX_FIELDS:
            item[field] = max(float(item.get(field, 0.0)), float(record.get(field, 0.0)))


def finalize_gaussian_diagnostic_accumulator(accumulator):
    # 结束累加：均值字段除以 count，最大值字段保留，返回按视角排序的行。
    finalized = []
    for key in sorted(accumulator.keys()):
        item = accumulator[key]
        count = max(int(item.get('count', 0)), 1)
        row = {
            'view_u': item['view_u'],
            'view_v': item['view_v'],
            'count': int(item.get('count', 0)),
            'patches': int(item.get('patches', 0)),
        }
        for field in GAUSSIAN_DIAGNOSTIC_MEAN_FIELDS:
            row[field] = float(item.get(field + '_weighted_sum', 0.0)) / float(count)
        for field in GAUSSIAN_DIAGNOSTIC_MAX_FIELDS:
            row[field] = float(item.get(field, 0.0))
        finalized.append(row)
    return finalized


def summarize_gaussian_diagnostics(view_diagnostics):
    # 把逐视角诊断汇总成单张测试图的标量，供日志和 per-sample CSV 使用。
    #
    #     均值字段按 count 加权（每个视角高斯数量不同）；最大值字段跨视角取 max。
    #     sigma 的汇总取 sigma_s/sigma_t 两个均值的平均。
    if not view_diagnostics:
        return {
            'gaussian_delta_px_mean': 0.0,
            'gaussian_delta_px_p95': 0.0,
            'gaussian_delta_px_max': 0.0,
            'gaussian_delta_s_px_max': 0.0,
            'gaussian_delta_t_px_max': 0.0,
            'gaussian_tanh_sat_ratio': 0.0,
            'gaussian_center_clamp_ratio': 0.0,
            'gaussian_sigma_mean': 0.0,
            'gaussian_opacity_mean': 0.0,
            'gaussian_offset_limit_px_mean': 0.0,
            'gaussian_offset_limit_px_max': 0.0,
            'gaussian_confidence_max_mean': 0.0,
            'gaussian_confidence_entropy': 0.0,
        }
    total_count = sum(max(int(item.get('count', 0)), 0) for item in view_diagnostics)
    total_count = max(total_count, 1)

    def weighted_mean(field):
        return sum(float(item.get(field, 0.0)) * max(int(item.get('count', 0)), 0) for item in view_diagnostics) / float(total_count)

    return {
        'gaussian_delta_px_mean': weighted_mean('delta_px_mean'),
        'gaussian_delta_px_p95': weighted_mean('delta_px_p95'),
        'gaussian_delta_px_max': max(float(item.get('delta_px_max', 0.0)) for item in view_diagnostics),
        'gaussian_delta_s_px_max': max(float(item.get('delta_s_px_max', 0.0)) for item in view_diagnostics),
        'gaussian_delta_t_px_max': max(float(item.get('delta_t_px_max', 0.0)) for item in view_diagnostics),
        'gaussian_tanh_sat_ratio': weighted_mean('tanh_sat_ratio'),
        'gaussian_center_clamp_ratio': weighted_mean('center_clamp_ratio'),
        'gaussian_sigma_mean': 0.5 * (weighted_mean('sigma_s_mean') + weighted_mean('sigma_t_mean')),
        'gaussian_opacity_mean': weighted_mean('opacity_mean'),
        'gaussian_offset_limit_px_mean': weighted_mean('offset_limit_px_mean'),
        'gaussian_offset_limit_px_max': max(float(item.get('offset_limit_px_max', 0.0)) for item in view_diagnostics),
        'gaussian_confidence_max_mean': weighted_mean('confidence_max_mean'),
        'gaussian_confidence_entropy': weighted_mean('confidence_entropy'),
    }


TAIL_METRIC_FIELDS = (
    'err_gt60_ratio',
    'top0_1pct_mse_share',
    'psnr_trim_0_1pct',
)
TAIL_ERROR_FRACTION = 60.0 / 255.0
TAIL_EXCLUDE_FRACTION = 0.001
IMAGE_PEAK_FLOAT = 1.0
IMAGE_PEAK_UINT8 = 255.0


def infer_image_peak(gt, pred):
    """Mirror the convention used by cal_psnr: non-negative float data lives in [0, 1]."""
    # 推断指标使用的数据峰值，与 cal_psnr 的约定保持一致。
    #
    #     非负且最大值不超过 1.5 视为 [0,1] 浮点图，返回 1；否则视为 [0,255]。
    #     出现负值时返回动态范围 max-min（用于归一化误差阈值）。
    highest = max(float(gt.max()), float(pred.max()))
    lowest = min(float(gt.min()), float(pred.min()))
    if lowest >= 0.0:
        return IMAGE_PEAK_FLOAT if highest <= 1.5 else IMAGE_PEAK_UINT8
    return highest - lowest


def tail_error_metrics(pred, gt, err_fraction=TAIL_ERROR_FRACTION,
                       exclude_fraction=TAIL_EXCLUDE_FRACTION):
    """Tail-sensitive metrics that expose localized failures which mean PSNR hides."""
    # 尾部敏感指标：暴露均值 PSNR 掩盖的局部严重失败。
    #
    #     返回三个标量：
    #       err_gt60_ratio       -- 绝对误差超过 60/255*peak 的像素占比；
    #       top0_1pct_mse_share  -- 最差 0.1% 像素贡献了多少总平方误差；
    #       psnr_trim_0_1pct     -- 去掉最差 0.1% 像素后的 PSNR。
    #     计算固定在 CPU float32 上，避免 GPU/标签设备不一致，也便于逐视角汇总。
    # Validation labels stay on CPU while predictions are produced on
    # ``cfg.device``. Match the existing PSNR/SSIM helpers by calculating
    # these scalar diagnostics on CPU before subtracting the tensors.
    pred = pred.detach().to(device='cpu', dtype=torch.float32)
    gt = gt.detach().to(device='cpu', dtype=torch.float32)
    error = (pred - gt).abs().reshape(-1)
    peak = infer_image_peak(gt, pred)
    err_threshold = err_fraction * peak
    numel = int(error.numel())
    if numel == 0:
        return {field: 0.0 for field in TAIL_METRIC_FIELDS}
    squared = error.square()
    total = float(squared.sum().item())
    excluded = max(1, int(math.ceil(exclude_fraction * numel)))
    if total <= 0.0:
        share = 0.0
        trimmed_mse = 0.0
    else:
        worst_sum = float(torch.topk(squared, excluded, largest=True).values.sum().item())
        share = worst_sum / total
        remaining = numel - excluded
        trimmed_mse = (total - worst_sum) / float(remaining) if remaining > 0 else 0.0
    trimmed_psnr = 10.0 * math.log10(peak * peak / trimmed_mse) if trimmed_mse > 0.0 else float('inf')
    return {
        'err_gt60_ratio': float((error > err_threshold).to(torch.float32).mean().item()),
        'top0_1pct_mse_share': float(share),
        'psnr_trim_0_1pct': float(trimmed_psnr),
    }


def summarize_tail_metrics(view_records):
    # 对逐视角尾部指标取有限值的算术平均；无有效记录时返回 0。
    summary = {}
    for field in TAIL_METRIC_FIELDS:
        values = [
            float(item[field]) for item in view_records
            if field in item and np.isfinite(float(item[field]))
        ]
        summary[field] = float(np.mean(values)) if values else 0.0
    return summary


def cal_per_view_metrics_RE(img1, img2, angRes_out):
    # 逐缺失视角计算 PSNR/SSIM 与尾部指标，返回记录列表。
    #
    #     与 cal_metrics_RE 相同的评测口径：跳过四角、每视角裁掉 22 像素边界。
    #     每条记录含 view_u/view_v/psnr/ssim 及 tail_error_metrics 的三个字段，
    #     后续可与高斯诊断按 (view_u,view_v) 合并。
    if len(img1.size()) == 2:
        H, W = img1.size()
        img1 = img1.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0, 2, 1, 3)
    if len(img2.size()) == 2:
        H, W = img2.size()
        img2 = img2.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0, 2, 1, 3)

    U, V, h, w = img1.size()
    bd = 22
    records = []
    for u in range(U):
        for v in range(V):
            if (u, v) in ((0, 0), (0, V - 1), (U - 1, 0), (U - 1, V - 1)):
                continue
            gt_view = img1[u, v, bd:-bd, bd:-bd]
            pred_view = img2[u, v, bd:-bd, bd:-bd]
            record = {
                'view_u': u,
                'view_v': v,
                'psnr': float(cal_psnr(gt_view, pred_view)),
                'ssim': float(cal_ssim(gt_view, pred_view)),
            }
            record.update(tail_error_metrics(pred_view, gt_view))
            records.append(record)
    return records


def merge_view_metrics_and_diagnostics(view_metrics, view_diagnostics):
    # 按 (view_u,view_v) 把图像指标行与高斯诊断行合并成一条记录。
    diagnostics_by_view = {(item['view_u'], item['view_v']): item for item in view_diagnostics}
    merged = []
    for metric in view_metrics:
        row = dict(metric)
        diagnostic = diagnostics_by_view.get((row['view_u'], row['view_v']), {})
        row.update(diagnostic)
        merged.append(row)
    return merged

def valid(test_loader, net, eval_angout=None, dataset_name='testset', epoch_idx=0):
    # 在一个固定测试集上做分块推理、指标统计与可选结果落盘。
    #
    #     输入 label 是预先切好的 angout x angout SAI；为了控制显存，先把大图按
    #     patchsize/stride 切成带重叠的块，逐块喂网络，再用 LFintegrate 拼回。
    #     返回 (平均 PSNR, 平均 SSIM, 样本数)。
    #
    #     副作用（由配置开关控制）：
    #       - 追加训练日志 txt；
    #       - 保存预测/GT 拼图 PNG；
    #       - 保存 per-sample 与 per-view CSV；
    #       - 汇总解码器的高斯偏移/尺度/opacity/置信度诊断。
    eval_angout = cfg.angout if eval_angout is None else eval_angout
    psnr_iter_test = []
    ssim_iter_test = []
    sample_metrics = []
    per_view_metrics = []
    dataset = getattr(test_loader, 'dataset', None)
    file_list = getattr(dataset, 'file_list', None)
    saved_count = 0
    for idx_iter, batch in (enumerate(test_loader)):
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError(
                "Expected fixed test loader to return (data, label), but got {}.".format(
                    type(batch).__name__
                )
            )
        # 测试 loader 固定返回 (data, label)：data 是 2x2 输入 SAI，
        # label 是 angout x angout 目标 SAI，均不含 batch 维。
        data, label = batch
        data = data.squeeze().to(cfg.device)  # numU, numV, h*angin, w*angin
        label = label.squeeze()

        uh, vw = data.shape
        h0, w0 = uh // cfg.angin, vw // cfg.angin
        # 按单视角空间尺寸切块，块内保留全部 2x2 输入视角。
        subLFin = LFdivide(data, cfg.angin, cfg.patchsize, cfg.stride)  # numU, numV, h*angin, w*angin
        numU, numV, H, W = subLFin.shape
        minibatch = 1
        num_inference = numU*numV//minibatch
        tmp_in = subLFin.contiguous().view(numU*numV, subLFin.shape[2], subLFin.shape[3])
        sample_gaussian_accumulator = {}
        previous_gaussian_stats_enabled = set_gaussian_diagnostics_enabled(net, cfg.save_test_diagnostics)
        inference_net = get_plain_net(net) if cfg.save_test_diagnostics else net
        with torch.no_grad():
            # 每个 patch 的输出仍是完整 angout x angout SAI，先按块缓存。
            subLFout = torch.zeros(
                numU, numV,
                eval_angout * cfg.patchsize,
                eval_angout * cfg.patchsize
            ).to(cfg.device)
            ptr = 0
            for idx_inference in range(num_inference):
                tmp = tmp_in[idx_inference * minibatch:(idx_inference + 1) * minibatch, :, :].unsqueeze(1)
                out = inference_net(tmp.to(cfg.device), eval_angout)  # one patch output
                if cfg.save_test_diagnostics:
                    decoder = get_gaussian_decoder(net)
                    update_gaussian_diagnostic_accumulator(sample_gaussian_accumulator, decoder.current_gaussian_stats)

                subLFout.view(-1,
                eval_angout * cfg.patchsize,
                eval_angout * cfg.patchsize
                              )[ptr:ptr + out.shape[0]] = out

                ptr += out.shape[0]

                del out, tmp
                torch.cuda.empty_cache()
        set_gaussian_diagnostics_enabled(net, previous_gaussian_stats_enabled)
        # 拼回完整 SAI；重叠区只取每块中心有效区，避免重复叠加。
        outLF = LFintegrate(subLFout, eval_angout, cfg.patchsize, cfg.stride, h0, w0)

        # 数据集级指标：只统计缺视角，并裁掉每视角 22 像素边界。
        psnr, ssim = cal_metrics_RE(label, outLF, cfg.angin, eval_angout)

        if isinstance(file_list, list) and idx_iter < len(file_list):
            sample_name = file_list[idx_iter]
        else:
            sample_name = 'sample_%04d' % idx_iter
        # 合并两类逐视角信息：图像质量（PSNR/SSIM/尾部指标）与高斯行为诊断。
        view_diagnostics = finalize_gaussian_diagnostic_accumulator(sample_gaussian_accumulator)
        view_metric_records = cal_per_view_metrics_RE(label, outLF, eval_angout)
        tail_summary = summarize_tail_metrics(view_metric_records)
        merged_view_records = merge_view_metrics_and_diagnostics(view_metric_records, view_diagnostics)
        for view_record in merged_view_records:
            view_record.update({
                'sample_idx': idx_iter,
                'sample_name': sample_name,
            })
        per_view_metrics.extend(merged_view_records)
        gaussian_summary = summarize_gaussian_diagnostics(view_diagnostics)
        sample_record = {
            'sample_idx': idx_iter,
            'sample_name': sample_name,
            'psnr': float(psnr),
            'ssim': float(ssim)
        }
        sample_record.update(gaussian_summary)
        sample_record.update(tail_summary)
        sample_metrics.append(sample_record)
        if cfg.print_test_sample_metrics:
            sample_line = (
                'Dataset----%10s,\t AngOut---%d,\t sample_idx---%d,\t sample---%s,\t PSNR---%.6f,\t SSIM---%.6f,\t err_gt60_ratio---%.6f,\t top0_1pct_mse_share---%.6f,\t psnr_trim0.1pct---%.4f'
                % (
                    dataset_name,
                    eval_angout,
                    idx_iter,
                    sample_name,
                    psnr,
                    ssim,
                    tail_summary['err_gt60_ratio'],
                    tail_summary['top0_1pct_mse_share'],
                    tail_summary['psnr_trim_0_1pct']
                )
            )
            print(sample_line)
            txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
            txtfile.write(sample_line + '\n')
            txtfile.close()

        if cfg.print_test_diagnostics and cfg.save_test_diagnostics:
            diagnostic_line = (
                'Diagnostics----%10s,\t AngOut---%d,\t sample_idx---%d,\t sample---%s,\t delta_mean_px---%.4f,\t delta_p95_px---%.4f,\t delta_max_px---%.4f,\t delta_s_max_px---%.4f,\t delta_t_max_px---%.4f,\t tanh_sat---%.6f,\t center_clamp---%.6f,\t sigma_mean---%.4f,\t opacity_mean---%.4f,\t offset_limit_mean---%.4f,\t offset_limit_max---%.4f,\t conf_max---%.4f,\t conf_entropy---%.4f'
                % (
                    dataset_name,
                    eval_angout,
                    idx_iter,
                    sample_name,
                    gaussian_summary['gaussian_delta_px_mean'],
                    gaussian_summary['gaussian_delta_px_p95'],
                    gaussian_summary['gaussian_delta_px_max'],
                    gaussian_summary['gaussian_delta_s_px_max'],
                    gaussian_summary['gaussian_delta_t_px_max'],
                    gaussian_summary['gaussian_tanh_sat_ratio'],
                    gaussian_summary['gaussian_center_clamp_ratio'],
                    gaussian_summary['gaussian_sigma_mean'],
                    gaussian_summary['gaussian_opacity_mean'],
                    gaussian_summary['gaussian_offset_limit_px_mean'],
                    gaussian_summary['gaussian_offset_limit_px_max'],
                    gaussian_summary['gaussian_confidence_max_mean'],
                    gaussian_summary['gaussian_confidence_entropy']
                )
            )
            print(diagnostic_line)
            txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
            txtfile.write(diagnostic_line + '\n')
            txtfile.close()

        limit_not_exceeded = cfg.save_test_image_limit < 0 or saved_count < cfg.save_test_image_limit
        if cfg.save_test_images and limit_not_exceeded:
            save_test_lf_pair_png(
                epoch_idx=epoch_idx,
                dataset_name=dataset_name,
                eval_angout=eval_angout,
                sample_name=sample_name,
                sample_idx=idx_iter,
                pred_lf_4d=outLF,
                gt_lf_2d=label
            )
            saved_count += 1

        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    num_epoch_test = len(psnr_iter_test)
    if cfg.save_test_metrics:
        csv_path = save_per_sample_metrics_csv(savepath, epoch_idx, dataset_name, eval_angout, sample_metrics)
        print('Per-sample metrics saved to %s' % csv_path)
        txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
        txtfile.write('Per-sample metrics saved to %s\n' % csv_path)
        txtfile.close()

    if cfg.save_test_diagnostics:
        per_view_csv_path = save_per_view_diagnostics_csv(savepath, epoch_idx, dataset_name, eval_angout, per_view_metrics)
        print('Per-view diagnostics saved to %s' % per_view_csv_path)
        txtfile = open(savepath + cfg.tag + cfg.model_name + '_training.txt', 'a')
        txtfile.write('Per-view diagnostics saved to %s\n' % per_view_csv_path)
        txtfile.close()

    return psnr_epoch_test, ssim_epoch_test, num_epoch_test


def save_per_sample_metrics_csv(save_dir, epoch_idx, dataset_name, eval_angout, sample_metrics):
    # 保存“每张测试图一行”的指标 CSV（含缺视角均值、尾部指标、高斯汇总）。
    metrics_dir = os.path.join(save_dir, 'metrics_per_sample')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(
        metrics_dir,
        'epoch_%04d_angout_%dx%d_%s.csv' % (
            int(epoch_idx),
            int(eval_angout),
            int(eval_angout),
            sanitize_name(dataset_name)
        )
    )
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        summary_fields = [
            'gaussian_delta_px_mean',
            'gaussian_delta_px_p95',
            'gaussian_delta_px_max',
            'gaussian_delta_s_px_max',
            'gaussian_delta_t_px_max',
            'gaussian_tanh_sat_ratio',
            'gaussian_center_clamp_ratio',
            'gaussian_sigma_mean',
            'gaussian_opacity_mean',
            'gaussian_offset_limit_px_mean',
            'gaussian_offset_limit_px_max',
            'gaussian_confidence_max_mean',
            'gaussian_confidence_entropy',
            'err_gt60_ratio',
            'top0_1pct_mse_share',
            'psnr_trim_0_1pct',
        ]
        csv_writer.writerow(['epoch', 'dataset', 'angout', 'sample_idx', 'sample_name', 'psnr', 'ssim'] + summary_fields)
        for item in sample_metrics:
            csv_writer.writerow([
                epoch_idx,
                dataset_name,
                eval_angout,
                item['sample_idx'],
                item['sample_name'],
                item['psnr'],
                item['ssim']
            ] + [item.get(field, '') for field in summary_fields])
    return csv_path


def save_per_view_diagnostics_csv(save_dir, epoch_idx, dataset_name, eval_angout, per_view_metrics):
    # 保存“每个缺失视角一行”的指标/诊断 CSV，用于定位具体坏视角。
    #
    #     列包含 PSNR/SSIM、尾部误差、高斯偏移（delta）、尺度（sigma）、opacity、
    #     四锚点置信度、偏移上限、tanh 饱和率与中心裁剪率等。
    metrics_dir = os.path.join(save_dir, 'metrics_per_view')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_path = os.path.join(
        metrics_dir,
        'epoch_%04d_angout_%dx%d_%s.csv' % (
            int(epoch_idx),
            int(eval_angout),
            int(eval_angout),
            sanitize_name(dataset_name)
        )
    )
    diagnostic_fields = [
        'count',
        'patches',
        'delta_px_mean',
        'delta_px_p95',
        'delta_px_max',
        'delta_s_px_mean',
        'delta_s_px_max',
        'delta_t_px_mean',
        'delta_t_px_max',
        'sigma_s_mean',
        'sigma_s_max',
        'sigma_t_mean',
        'sigma_t_max',
        'opacity_mean',
        'opacity_max',
        'offset_limit_px_mean',
        'offset_limit_px_max',
        'confidence_mean',
        'confidence_max_mean',
        'confidence_entropy',
        'confidence_anchor_00_mean',
        'confidence_anchor_01_mean',
        'confidence_anchor_10_mean',
        'confidence_anchor_11_mean',
        'tanh_sat_ratio',
        'tanh_sat_s_ratio',
        'tanh_sat_t_ratio',
        'center_clamp_ratio',
        'err_gt60_ratio',
        'top0_1pct_mse_share',
        'psnr_trim_0_1pct',
    ]
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch',
            'dataset',
            'angout',
            'sample_idx',
            'sample_name',
            'view_u',
            'view_v',
            'psnr',
            'ssim',
        ] + diagnostic_fields)
        for item in per_view_metrics:
            csv_writer.writerow([
                epoch_idx,
                dataset_name,
                eval_angout,
                item['sample_idx'],
                item['sample_name'],
                item['view_u'],
                item['view_v'],
                item['psnr'],
                item['ssim'],
            ] + [item.get(field, '') for field in diagnostic_fields])
    return csv_path

def sanitize_name(name):
    # 把数据集/样本名中的路径分隔符和 Windows 非法字符替换成下划线。
    safe_name = str(name).replace('\\', '_').replace('/', '_').replace(' ', '_')
    safe_name = safe_name.replace(':', '_').replace('*', '_').replace('?', '_')
    safe_name = safe_name.replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    return safe_name


def lf_4d_to_2d(lf_4d):
    # 把 [U,V,H,W] 视角网格拼回 SAI [U*H,V*W]；先转置再 view，保证块顺序正确。
    view_u, view_v, height, width = lf_4d.shape
    return lf_4d.permute(0, 2, 1, 3).contiguous().view(view_u * height, view_v * width)


def sai_2d_to_lf_4d(sai_2d, angout):
    # lf_4d_to_2d 的逆操作：SAI [angout*H,angout*W] -> [angout,angout,H,W]。
    if sai_2d.ndim != 2:
        raise ValueError('SAI tensor must have shape [U*H, V*W]')
    if angout <= 0:
        raise ValueError('angout must be positive')
    height, width = sai_2d.shape
    if height % angout != 0 or width % angout != 0:
        raise ValueError('SAI dimensions must be divisible by angout')
    return sai_2d.view(angout, height // angout, angout, width // angout).permute(0, 2, 1, 3).contiguous()


def crop_lf_view_borders(lf_4d, border=22):
    # 对四维光场逐视角裁掉四周 border 像素（默认 22，与指标口径一致）。
    if lf_4d.ndim != 4:
        raise ValueError('light field tensor must have shape [U, V, H, W]')
    if border < 0:
        raise ValueError('border must be non-negative')
    _, _, height, width = lf_4d.shape
    if height <= 2 * border or width <= 2 * border:
        raise ValueError('border leaves no pixels in at least one sub-aperture view')
    return lf_4d[:, :, border:height - border, border:width - border]


def crop_sai_border(sai_2d, border=22):
    # 对二维 SAI 整体裁掉四周 border 像素（注意：会同时跨过视角块边界）。
    if sai_2d.ndim != 2:
        raise ValueError('SAI tensor must have shape [U*H, V*W]')
    if border < 0:
        raise ValueError('border must be non-negative')
    height, width = sai_2d.shape
    if height <= 2 * border or width <= 2 * border:
        raise ValueError('border leaves no pixels in at least one SAI dimension')
    return sai_2d[border:height - border, border:width - border]


def save_test_lf_pair_png(epoch_idx, dataset_name, eval_angout, sample_name, sample_idx, pred_lf_4d, gt_lf_2d):
    # 保存一张测试图的预测/GT 拼接图。
    #
    #     目录结构：saved_test_images/epoch_XXXX/angout_NxN/<dataset>/；
    #     文件名为 <样本序号>_<样本名>_pred.png / _gt.png。
    #     保存前先把预测转成 [U,V,H,W]，把 GT SAI 还原成四维，二者都裁掉
    #     22 像素视角边界，再拼回大图，保证预测与 GT 空间对齐。
    save_dir = os.path.join(
        savepath,
        'saved_test_images',
        'epoch_%04d' % int(epoch_idx),
        'angout_%dx%d' % (int(eval_angout), int(eval_angout)),
        sanitize_name(dataset_name)
    )
    os.makedirs(save_dir, exist_ok=True)

    sample_stem = os.path.splitext(os.path.basename(str(sample_name)))[0]
    if sample_stem == '':
        sample_stem = 'sample_%04d' % int(sample_idx)
    safe_stem = '%04d_%s' % (int(sample_idx), sanitize_name(sample_stem))

    pred_lf_4d = pred_lf_4d.detach().cpu().to(torch.float32)
    gt_lf_4d = sai_2d_to_lf_4d(gt_lf_2d.detach().cpu().to(torch.float32), eval_angout)
    pred_2d = lf_4d_to_2d(crop_lf_view_borders(pred_lf_4d))
    gt_2d = lf_4d_to_2d(crop_lf_view_borders(gt_lf_4d))

    pred_png_path = os.path.join(save_dir, safe_stem + '_pred.png')
    gt_png_path = os.path.join(save_dir, safe_stem + '_gt.png')

    save_tensor_image_png(pred_2d, pred_png_path)
    save_tensor_image_png(gt_2d, gt_png_path)

    return pred_png_path, gt_png_path


def save_tensor_image_png(tensor_2d, save_path):
    # 把 [0,1] 的二维浮点张量量化成 uint8 灰度 PNG（四舍五入到最近整数）。
    image = tensor_2d.detach().cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image, mode='L').save(save_path)


def save_ckpt(state, save_path, filename='checkpoint.pth.tar'):
    # 把 build_training_checkpoint 生成的字典写到 save_path/filename。
    torch.save(state, os.path.join(save_path,filename))


def main(cfg):
    # 程序主入口：固定随机种子、建目录、建训练/测试 Loader，然后调用 train()。
    #
    #     注意 savepath 是模块级全局变量，在 __main__ 中创建；训练日志、图片、
    #     CSV、checkpoint 都写到该目录下。
    setup_seed(10)
    if not os.path.exists(savepath):
        try:
            os.mkdir(savepath)
        except:
            os.makedirs(savepath)


    os.system('cp -r ../code ' + savepath)
    # train_set = TrainSetLoader_RE_HCI(dataset_dir=cfg.trainset_dir, cfg = cfg)
    print("Training mode: {} ({})".format(cfg.train_mode, training_mode_name(cfg.train_mode)))
    if cfg.train_mode == ARBITRARY_SCALE_MODE:
        print("Training angular output: random integer in [{}, {}]".format(
            cfg.angout_min, cfg.angout_max))
    else:
        print("Training angular output: fixed {}x{}".format(cfg.angout, cfg.angout))
    print("Training augmentation: {}".format(
        "enabled" if training_augmentation_enabled(cfg.train_mode) else "disabled"))
    print("Training data mode: {} (fixed by policy)".format(TRAIN_DATA_MODE))
    print("Test data mode: {} (pre-sliced; online angular crop disabled)".format(TEST_DATA_MODE))

    train_set = TrainSetLoaderRawLF(dataset_dir=cfg.trainset_dir, raw_lf_key=cfg.raw_lf_key)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(cfg)
    # test_Names, test_Loaders, length_of_tests = SSR1_TestSetDataLoader(cfg, ['EPFL'])
    train(cfg, train_loader, test_Names, test_Loaders)

def setup_seed(seed):
    # 固定 Python/NumPy/PyTorch 的随机种子，便于复现实验。
    #
    #     这里没有开启 cudnn.deterministic，因此卷积仍可能选择非确定性算法；
    #     如需严格逐位复现，需要额外设置 deterministic 并关闭 benchmark。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
if __name__ == '__main__':
    cfg = parse_args()
    if cfg.cuda_visible_devices != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    if cfg.eval_angouts.strip() == "":
        cfg.eval_angouts = str(cfg.angout)
    cfg.eval_angouts_list = [int(item.strip()) for item in cfg.eval_angouts.split(',') if item.strip() != '']
    validate_training_policy(
        train_mode=cfg.train_mode,
        angout=cfg.angout,
        angout_min=cfg.angout_min,
        angout_max=cfg.angout_max,
        source_ang_res=cfg.source_ang_res,
    )
    if any(eval_angout != cfg.angout for eval_angout in cfg.eval_angouts_list):
        raise ValueError(
            "fixed test data only supports eval_angouts equal to --angout. "
            "Got eval_angouts={} and angout={}.".format(cfg.eval_angouts_list, cfg.angout)
        )
    global savepath, writer
    savepath = '../save/gs_checkpoint_' + cfg.tag + '/'
    print(savepath)
    writer = SummaryWriter(os.path.join(savepath, 'tensorboard'))

    main(cfg)
