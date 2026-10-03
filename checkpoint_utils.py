def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1'):
        return True
    if value in ('no', 'false', 'f', '0'):
        return False
    raise ValueError('Boolean value expected.')


def clean_state_dict_key(key):
    return key[7:] if key.startswith('module.') else key


def filter_pretrain_state_dict(state_dict, current_state):
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
    return {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'state_dict': net.state_dict(),
        'loss': loss_list,
    }


def restore_scheduler_state(scheduler, checkpoint, epoch_state):
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        return True

    scheduler.last_epoch = epoch_state
    return False
