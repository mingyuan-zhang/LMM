from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import os
import numpy as np
from tqdm import tqdm
import json
import pickle




class FeatureExtractor(imagebind_model.ImageBindModel):

    def forward(self, inputs):
        outputs = {}
        for modality_key, modality_value in inputs.items():
            reduce_list = (
                modality_value.ndim >= 5
            )  # Audio and Video inputs consist of multiple clips
            if reduce_list:
                B, S = modality_value.shape[:2]
                modality_value = modality_value.reshape(
                    B * S, *modality_value.shape[2:]
                )

            if modality_value is not None:
                modality_value = self.modality_preprocessors[modality_key](
                    **{modality_key: modality_value}
                )
                trunk_inputs = modality_value["trunk"]
                head_inputs = modality_value["head"]
                modality_value = self.modality_trunks[modality_key](**trunk_inputs)
                word_feat = modality_value
                seq_feat = self.modality_heads[modality_key](
                    word_feat, **head_inputs
                )
                seq_feat = self.modality_postprocessors[modality_key](
                    seq_feat
                )
        return word_feat, seq_feat
    

def imagebind_huge(pretrained=False):
    model = FeatureExtractor(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )

    if pretrained:
        file_path = os.path.abspath(os.path.dirname(__file__))
        ckpt_dir = os.path.join(file_path, '../../../data/motionverse/pretrained')
        ckpt_path = os.path.join(ckpt_dir, 'imagebind_huge.pth')
        if not os.path.exists(ckpt_path):
            print(
                "Downloading imagebind weights to motionverse/pretrained/imagebind_huge.pth ..."
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ckpt_path,
                progress=True,
            )

        model.load_state_dict(torch.load(ckpt_path))
    return model


def extract_text_feature(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    text_list = text
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    }
    with torch.no_grad():
        text_word_feat, text_seq_feat = model(inputs)
    return text_word_feat, text_seq_feat


def extract_audio_feature(audio_paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    inputs = {
        ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device)
    }
    inputs['audio'] = inputs['audio'][:, :1]
    with torch.no_grad():
        audio_word_feat, audio_seq_feat = model(inputs)
    return audio_word_feat, audio_seq_feat

