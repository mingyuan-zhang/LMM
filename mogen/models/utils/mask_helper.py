import torch

def get_tomato_slice(idx):
    if idx == 0:
        result = [0, 1, 2, 3, 463, 464, 465]
    else:
        result = [
            4 + (idx - 1) * 3,
            4 + (idx - 1) * 3 + 1,
            4 + (idx - 1) * 3 + 2,
            157 + (idx - 1) * 6,
            157 + (idx - 1) * 6 + 1,
            157 + (idx - 1) * 6 + 2,
            157 + (idx - 1) * 6 + 3,
            157 + (idx - 1) * 6 + 4,
            157 + (idx - 1) * 6 + 5,
            463 + idx * 3,
            463 + idx * 3 + 1,
            463 + idx * 3 + 2,
        ]
    return result


def get_part_slice(idx_list, func):
    result = []
    for idx in idx_list:
        result.extend(func(idx))
    return result


def expand_mask_to_all(mask, body_scale, hand_scale, face_scale):
    func = get_tomato_slice
    root_slice = get_part_slice([0], func)
    head_slice = get_part_slice([12, 15], func)
    stem_slice = get_part_slice([3, 6, 9], func)
    larm_slice = get_part_slice([14, 17, 19, 21], func)
    rarm_slice = get_part_slice([13, 16, 18, 20], func)
    lleg_slice = get_part_slice([2, 5, 8, 11], func)
    rleg_slice = get_part_slice([1, 4, 7, 10], func)
    lhnd_slice = get_part_slice(range(22, 37), func)
    rhnd_slice = get_part_slice(range(37, 52), func)
    face_slice = range(619, 669)
    B, T = mask.shape[0], mask.shape[1]
    mask = mask.view(B, T, -1)
    all_mask = torch.zeros(B, T, 669).type_as(mask)
    all_mask[:, :, root_slice] = mask[:, :, 0].unsqueeze(-1).repeat(1, 1, len(root_slice))
    all_mask[:, :, head_slice] = mask[:, :, 1].unsqueeze(-1).repeat(1, 1, len(head_slice))
    all_mask[:, :, stem_slice] = mask[:, :, 2].unsqueeze(-1).repeat(1, 1, len(stem_slice))
    all_mask[:, :, larm_slice] = mask[:, :, 3].unsqueeze(-1).repeat(1, 1, len(larm_slice))
    all_mask[:, :, rarm_slice] = mask[:, :, 4].unsqueeze(-1).repeat(1, 1, len(rarm_slice))
    all_mask[:, :, lleg_slice] = mask[:, :, 5].unsqueeze(-1).repeat(1, 1, len(lleg_slice))
    all_mask[:, :, rleg_slice] = mask[:, :, 6].unsqueeze(-1).repeat(1, 1, len(rleg_slice))
    all_mask[:, :, lhnd_slice] = mask[:, :, 7].unsqueeze(-1).repeat(1, 1, len(lhnd_slice))
    all_mask[:, :, rhnd_slice] = mask[:, :, 8].unsqueeze(-1).repeat(1, 1, len(rhnd_slice))
    all_mask[:, :, face_slice] = mask[:, :, 9].unsqueeze(-1).repeat(1, 1, len(face_slice))
    all_mask[:, :, root_slice] *= body_scale
    all_mask[:, :, head_slice] *= body_scale
    all_mask[:, :, stem_slice] *= body_scale
    all_mask[:, :, larm_slice] *= body_scale
    all_mask[:, :, rarm_slice] *= body_scale
    all_mask[:, :, lleg_slice] *= body_scale
    all_mask[:, :, rleg_slice] *= body_scale
    all_mask[:, :, lhnd_slice] *= hand_scale
    all_mask[:, :, rhnd_slice] *= hand_scale
    all_mask[:, :, face_slice] *= face_scale
    return all_mask
