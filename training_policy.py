import random


ARBITRARY_SCALE_MODE = 1
FIXED_SCALE_MODE = 2
VALID_TRAIN_MODES = (ARBITRARY_SCALE_MODE, FIXED_SCALE_MODE)
TRAIN_DATA_MODE = "raw9x9"
TEST_DATA_MODE = "fixed"


def _validate_train_mode(train_mode):
    if train_mode not in VALID_TRAIN_MODES:
        raise ValueError(
            "train_mode must be 1 (arbitrary-scale) or 2 (fixed-scale), got {}".format(train_mode)
        )


def validate_training_policy(train_mode, angout, angout_min, angout_max, source_ang_res):
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
    validate_training_policy(train_mode, angout, angout_min, angout_max, source_ang_res)
    if train_mode == FIXED_SCALE_MODE:
        return angout
    rng = random if rng is None else rng
    return rng.randint(angout_min, angout_max)


def training_augmentation_enabled(train_mode):
    _validate_train_mode(train_mode)
    return train_mode == FIXED_SCALE_MODE


def apply_training_augmentation(train_mode, data, label, augmentation_fn):
    if not training_augmentation_enabled(train_mode):
        return data, label
    return augmentation_fn(data, label)


def training_mode_name(train_mode):
    _validate_train_mode(train_mode)
    if train_mode == ARBITRARY_SCALE_MODE:
        return "arbitrary-scale"
    return "fixed-scale fine-tuning"


def training_scale_tag(train_mode, angout, angout_min, angout_max):
    _validate_train_mode(train_mode)
    if train_mode == ARBITRARY_SCALE_MODE:
        return "any{}-{}".format(angout_min, angout_max)
    return "fixed{}x{}".format(angout, angout)
