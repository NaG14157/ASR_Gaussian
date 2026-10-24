# =============================================================================
# checkpoint / 命令行布尔参数工具
#
# 这个文件只处理“权重字典的兼容性”和“训练状态的打包/恢复”，不涉及网络结构。
# 这样 train.py 就不必关心 DataParallel 前缀、旧权重缺键、旧 checkpoint
# 没有 scheduler 等细节。
# =============================================================================


def str2bool(value):
    # 把命令行字符串转换成 bool，兼容 yes/no、true/false、t/f、1/0。
    #
    #     argparse 的 type=bool 会把字符串 "False" 也解释成 True，因此项目里所有
    #     布尔开关都显式使用本函数。传入真正的 bool 时原样返回。
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean value expected.')


def clean_state_dict_key(key):
    # 去掉 DataParallel 保存权重时自动加上的 "module." 前缀。
    #
    #     单卡直接保存的权重没有该前缀；多卡 torch.nn.DataParallel 包一层后会有。
    #     统一去掉后才能把旧权重加载进单卡或新的 DataParallel 模型。
    return key[7:] if key.startswith('module.') else key


def filter_pretrain_state_dict(state_dict, current_state):
    # 筛选出“当前模型存在且形状完全相同”的预训练参数。
    #
    #     参数：
    #       state_dict   -- checkpoint 里的旧权重（键可能带 module. 前缀）
    #       current_state-- 当前模型 net.state_dict()，用于比对键名和张量形状
    #     返回：
    #       filtered_state -- 可直接 load_state_dict(strict=False) 的兼容子集
    #       skipped_keys   -- 因缺键或形状变化而被跳过的原始键，供日志排查
    #
    #     这样模型新增/删改少量层时，仍能复用其余主干权重，而不是整份权重加载失败。
    filtered_state = {}
    skipped_keys = []
    for key, value in state_dict.items():
        clean_key = clean_state_dict_key(key)
        if clean_key not in current_state or current_state[clean_key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[clean_key] = value
    return filtered_state, skipped_keys

def build_training_checkpoint(epoch, optimizer, scheduler, net, loss_list):
    # 打包一次完整训练状态，供断点续训使用。
    #
    #     epoch 记录“已经完成的 epoch 数”（保存处传入 idx_epoch + 1）；
    #     optimizer/scheduler 的 state_dict 保证续训时动量、学习率进度可恢复；
    #     loss_list 仅用于记录训练历史，加载时不会被模型使用。
    return {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'state_dict': net.state_dict(),
        'loss': loss_list,
    }


def restore_scheduler_state(scheduler, checkpoint, epoch_state):
    # 恢复学习率调度器；兼容没有 scheduler 字段的旧 checkpoint。
    #
    #     返回 True 表示从 checkpoint 完整恢复了调度器状态；
    #     返回 False 表示旧格式缺少 scheduler，只能把 last_epoch 对齐到已训练
    #     epoch 数，让 StepLR 从正确的时间点继续衰减（但无法恢复历史精确状态）。
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        return True

    scheduler.last_epoch = epoch_state
    return False
