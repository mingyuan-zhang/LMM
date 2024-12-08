import torch


def create_mask_sequence(mask_cfg, seq_len):
    type_name = mask_cfg['type']
    if type_name == 'raster order':
        num_tokens = mask_cfg['num_tokens']
        idx_list = []
        all_idx = torch.arange(seq_len)
        for i in range(0, seq_len, num_tokens):
            idx_list.append(all_idx[i: i + num_tokens])
        return idx_list
    elif type_name == 'random order':
        num_tokens = mask_cfg['num_tokens']
        idx_list = []
        all_idx = torch.randperm(seq_len)
        for i in range(0, seq_len, num_tokens):
            idx_list.append(all_idx[i: i + num_tokens])
        return idx_list
    elif type_name == 'single':
        idx_list = [torch.arange(seq_len)]
        return idx_list
    else:
        raise NotImplementedError()
