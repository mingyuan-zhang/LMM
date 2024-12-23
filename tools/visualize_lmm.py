import argparse
import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mogen.models.utils.imagebind_wrapper import (
    extract_text_feature,
    extract_audio_feature
)
from mogen.models import build_architecture

from mogen.utils.plot_utils import (
    plot_3d_motion,
    add_audio,
    get_audio_length
)
from mogen.datasets.paramUtil import (
    t2m_body_hand_kinematic_chain,
    t2m_kinematic_chain
)
from mogen.datasets.utils import recover_from_ric

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def plot_tomato(data, kinematic_chain, result_path, npy_path, fps, sigma=None):
    joints = recover_from_ric(torch.from_numpy(data).float(), 52).numpy()
    if sigma is not None:
        joints = motion_temporal_filter(joints, sigma=sigma)       
    # joints = joints / 120
    plot_3d_motion(
        out_path=result_path,
        joints=joints,
        kinematic_chain=kinematic_chain,
        title=None,
        fps=fps)
    if npy_path is not None:
        np.save(npy_path, joints)

def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--motion_length',
                        type=int,
                        help='expected motion length',
                        nargs='+')
    parser.add_argument('--fps',
                        type=float,
                        help='fps',
                        default=20.0)
    parser.add_argument('--out', help='output animation file')
    parser.add_argument('--text',
                        type=str,
                        help='text description',
                        nargs='+',
                        default=[])
    parser.add_argument('--speech',
                        type=str,
                        help='path to speech wav files',
                        nargs='+',
                        default=[])
    parser.add_argument('--dataset_name',
                        type=str,
                        help='dataset name',
                        default='humanml3d_t2m')
    parser.add_argument('--sigma',
                        type=float,
                        help='hyperparamter for post smoothing',
                        default=2.5)
    parser.add_argument('--pose_npy',
                        help='output pose sequence file',
                        default=None)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    data_dir = "data/motionverse/statistics"
    mean_path = os.path.join(data_dir, "mean.npy")
    std_path = os.path.join(data_dir, "std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)

    device = args.device
    
    motion_length = args.motion_length
    if len(args.speech) > 0:
        assert len(args.speech) == len(motion_length)
        for i in range(len(motion_length)):
            motion_length[i] = min(motion_length[i], int(get_audio_length(args.speech[i]) * args.fps) + 1)
    num_intervals = len(motion_length)
    max_length = max(motion_length)
    
    input_dim = 669
    motion = torch.zeros(num_intervals, max_length, input_dim).to(device)
    motion_mask = torch.zeros(num_intervals, max_length).to(device)
    motion_metas = []
    rotation_type = "h3d_rot"
    for i in range(num_intervals):
        motion_mask[i, :motion_length[i]] = 1
        motion_metas.append(
            {
                'meta_data': dict(framerate=args.fps, dataset_name=args.dataset_name, rotation_type=rotation_type)
            }
        )
    motion_mask = motion_mask.unsqueeze(-1).repeat(1, 1, 10)
    motion_mask[:, :, 9] = 0
    kinematic_chain = t2m_body_hand_kinematic_chain
    
    motion_length = torch.Tensor(motion_length).long().to(device)
    model = model.to(device)
    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'motion_metas': motion_metas,
        'num_intervals': num_intervals
    }
    if len(args.text) > 0:
        text_word_feat, text_seq_feat = \
            extract_text_feature(args.text)
        assert text_word_feat.shape[0] == len(args.text)
        assert text_word_feat.shape[1] == 77
        assert text_word_feat.shape[2] == 1024
        assert text_seq_feat.shape[0] == len(args.text)
        assert text_seq_feat.shape[1] == 1024
        input['text_word_feat'] = text_word_feat
        input['text_seq_feat'] = text_seq_feat
        input['text_cond'] = torch.Tensor([1.0] * num_intervals).to(device)
    else:
        input['text_word_feat'] = torch.zeros(num_intervals, 77, 1024).to(device)
        input['text_seq_feat'] = torch.zeros(num_intervals, 1024)
        input['text_cond'] = torch.Tensor([0] * num_intervals).to(device)


    if len(args.speech) > 0:
        speech_word_feat, speech_seq_feat = \
            extract_audio_feature(args.speech)
        assert speech_word_feat.shape[0] == len(args.speech)
        assert speech_word_feat.shape[1] == 229
        assert speech_word_feat.shape[2] == 768
        assert speech_seq_feat.shape[0] == len(args.speech)
        assert speech_seq_feat.shape[1] == 1024
        input['speech_word_feat'] = speech_word_feat
        input['speech_seq_feat'] = speech_seq_feat
        input['speech_cond'] = torch.Tensor([1.0] * num_intervals).to(device)
    else:
        input['speech_word_feat'] = torch.zeros(num_intervals, 229, 768).to(device)
        input['speech_seq_feat'] = torch.zeros(num_intervals, 1024)
        input['speech_cond'] = torch.Tensor([0] * num_intervals).to(device)
        
    all_pred_motion = []
    with torch.no_grad():
        input['inference_kwargs'] = {}
        output = model(**input)
        for i in range(num_intervals):
            pred_motion = output[i]['pred_motion'][:int(motion_length[i])]
            pred_motion = pred_motion.cpu().detach().numpy()
            pred_motion = pred_motion * std + mean
            all_pred_motion.append(pred_motion)
        pred_motion = np.concatenate(all_pred_motion, axis=0)

    plot_tomato(data=pred_motion,
                kinematic_chain=kinematic_chain,
                result_path=args.out,
                npy_path=args.pose_npy,
                fps=args.fps,
                sigma=args.sigma)
    
    if len(args.speech) > 0:
        add_audio(args.out, args.speech)
        
if __name__ == '__main__':
    main()