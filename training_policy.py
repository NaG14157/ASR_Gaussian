# =============================================================================
# 训练策略层（只做“怎么训练”的决策，不参与任何网络张量计算）
#
# 本文件把训练模式相关的分支从 train.py 中抽出来，集中回答四个问题：
#   1. 当前配置是否合法（角度范围、原始光场角度数）；
#   2. 本次迭代应该用多大的输出角分辨率 angout；
#   3. 要不要做数据增强；
#   4. 日志和 checkpoint 文件名里该写什么标签。
#
# 两种模式：
#   模式 1（任意尺度）：每个 batch 从 [angout_min, angout_max] 随机取 angout，
#                       让同一套权重见到多种输出角度规模，且不做几何增强；
#   模式 2（固定尺度微调）：始终使用 --angout，并启用同步几何/中心缩放增强。
# =============================================================================
import random


# 训练模式编号。1=任意尺度预训练；2=固定尺度微调。
ARBITRARY_SCALE_MODE = 1
FIXED_SCALE_MODE = 2
VALID_TRAIN_MODES = (ARBITRARY_SCALE_MODE, FIXED_SCALE_MODE)
# 数据协议标签：训练从原始 9x9 光场在线裁切；测试使用预先切好的固定配对。
TRAIN_DATA_MODE = "raw9x9"
TEST_DATA_MODE = "fixed"


def _validate_train_mode(train_mode):
    # 校验模式编号，避免调用方传入未定义的模式后静默走错分支。
    if train_mode not in VALID_TRAIN_MODES:
        raise ValueError(
            "train_mode must be 1 (arbitrary-scale) or 2 (fixed-scale), got {}".format(train_mode)
        )


def validate_training_policy(train_mode, angout, angout_min, angout_max, source_ang_res):
    # 在训练开始前统一检查角度相关配置，失败时尽早抛错。
    #
    #     约束来源：
    #       - 输出角度至少为 3，因为 2x2 只有四个已知角、没有“缺失视角”可监督；
    #       - angout 不能超过原始光场的 source_ang_res，否则在线裁切取不到那么多视角；
    #       - 任意尺度模式下还要求 min <= max。
    _validate_train_mode(train_mode)
    if source_ang_res < 3:
        raise ValueError("source_ang_res must be at least 3")

    if train_mode == ARBITRARY_SCALE_MODE:
        if angout_min < 3:
            raise ValueError("angout_min must be at least 3 in arbitrary-scale mode")
        if angout_min > angout_max:
            raise ValueError("angout_min must be less than or equal to angout_max")
        if angout_max > source_ang_res:
            raise ValueError("angout_max must not exceed source_ang_res")
        return

    if angout < 3:
        raise ValueError("angout must be at least 3 in fixed-scale mode")
    if angout > source_ang_res:
        raise ValueError("angout must not exceed source_ang_res")


def resolve_training_angout(
        train_mode,
        angout,
        angout_min,
        angout_max,
        source_ang_res,
        rng=None):
    # 返回本次前向使用的输出角分辨率。
    #
    #     固定尺度模式直接返回 angout；任意尺度模式用传入的 rng（默认全局 random）
    #     在闭区间 [angout_min, angout_max] 内均匀取整数。训练循环每个 batch 调一次，
    #     因此同一 epoch 内不同样本可以有不同的输出角度网格。
    validate_training_policy(train_mode, angout, angout_min, angout_max, source_ang_res)
    if train_mode == FIXED_SCALE_MODE:
        return angout
    rng = random if rng is None else rng
    return rng.randint(angout_min, angout_max)


def training_augmentation_enabled(train_mode):
    # 只有固定尺度模式启用增强。
    #
    #     任意尺度模式本身已经在“角度尺度”上做随机化，再叠加翻转/缩放会让
    #     每个 batch 的监督分布更不稳定，因此这里显式关闭。
    _validate_train_mode(train_mode)
    return train_mode == FIXED_SCALE_MODE


def apply_training_augmentation(train_mode, data, label, augmentation_fn):
    # 按模式决定是否调用增强函数；data 与 label 始终成对传入、成对返回。
    #
    #     增强函数由调用方注入（训练时是 augmentation_with_resize），这样本模块
    #     不依赖 torch/numpy，策略本身可以独立测试。
    if not training_augmentation_enabled(train_mode):
        return data, label
    return augmentation_fn(data, label)


def training_mode_name(train_mode):
    # 返回用于日志展示的模式名称。
    _validate_train_mode(train_mode)
    if train_mode == ARBITRARY_SCALE_MODE:
        return "arbitrary-scale"
    return "fixed-scale fine-tuning"


def training_scale_tag(train_mode, angout, angout_min, angout_max):
    # 生成 checkpoint 文件名中的尺度标签，例如 any7-9 / fixed7x7。
    #
    #     该字符串会直接拼进权重文件名，修改格式会影响已有文件的命名约定。
    _validate_train_mode(train_mode)
    if train_mode == ARBITRARY_SCALE_MODE:
        return "any{}-{}".format(angout_min, angout_max)
    return "fixed{}x{}".format(angout, angout)
