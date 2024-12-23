<div align="center">

<h1>Large Motion Model for Unified Multi-Modal Motion Generation</h1>

<div>
    <a href='https://mingyuan-zhang.github.io/' target='_blank'>Mingyuan Zhang</a><sup>1*</sup>&emsp;
    <a href='' target='_blank'>Daisheng Jin</a><sup>1*</sup>&emsp;
    <a href='https://www.linkedin.com/in/rheallyc/' target='_blank'>Chenyang Gu</a><sup>1*</sup>&emsp;
    <a href='https://hongfz16.github.io/' target='_blank'>Fangzhou Hong</a><sup>1</sup>&emsp;
    <a href='https://caizhongang.github.io/' target='_blank'>Zhongang Cai</a><sup>1,2</sup>&emsp;
    <a href='https://www.linkedin.com/in/jingfang-h-26746013a/' target='_blank'>Jingfang Huang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=MaAiOikAAAAJ&hl=en' target='_blank'>Chongzhi Zhang</a><sup>1</sup>&emsp;
    <a href='https://gxyes.github.io/' target='_blank'>Xinying Guo</a><sup>1</sup>&emsp;
    <a href='https://yanglei.me/' target='_blank'>Lei Yang</a><sup>2</sup>&emsp;
    <a href='https://personal.ntu.edu.sg/yhe/' target='_blank'>Ying He</a><sup>1</sup>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a><sup>1+</sup>
</div>
<div>
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp;
    <sup>2</sup>SenseTime Research&emsp;
</div>
<div>
    <sup>*</sup>co-first authors
    <sup>+</sup>corresponding author
</div>


---

<h4 align="center">
  <a href="https://mingyuan-zhang.github.io/projects/LMM.html" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/pdf/2404.01284.pdf" target='_blank'>[PDF]</a> •
  <a href="https://arxiv.org/abs/2404.01284" target='_blank'>[arXiv]</a> •
  <a href="https://www.youtube.com/watch?v=Aprm9h8lFj4" target='_blank'>[Video]</a> •
  <a href="https://lmm.readthedocs.io/en/latest/index.html" target='_blank'>[Documentation]</a> •
  <a href="https://huggingface.co/spaces/mingyuan/LMM" target='_blank'>[Hugging Face Demo]</a>
  <br> <br>
  <a href='https://lmm.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/lmm/badge/?version=latest' alt='Documentation Status' height="20"/>
  </a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=mingyuan-zhang/LMM" alt="visitor badge" height="20"/>
</h4>

</div>

<div align="center">
<tr>
    <img src="imgs/teaser.png" width="90%"/>
</tr>
</div>

>**Abstract:** Human motion generation, a cornerstone technique in animation and video production, has widespread applications in various tasks like text-to-motion and music-to-dance.
Previous works focus on developing specialist models tailored for each task without scalability.
In this work, we present <strong>Large Motion Model (LMM)</strong>, a motion-centric, multi-modal framework that unifies mainstream motion generation tasks into a generalist model.
A unified motion model is appealing since it can leverage a wide range of motion data to achieve broad generalization beyond a single task.
However, it is also challenging due to the heterogeneous nature of substantially different motion data and tasks.
LMM tackles these challenges from three principled aspects:
<strong>1)</strong> <i>Data:</i> We consolidate datasets with different modalities, formats and tasks into a comprehensive yet unified motion generation dataset,
    <strong>MotionVerse</strong>, comprising 10 tasks, 16 datasets, a total of 320k sequences, and 100 million frames.
<strong>2)</strong> <i>Architecture:</i> We design an articulated attention mechanism <strong>ArtAttention</strong> that incorporates body part-aware modeling into Diffusion Transformer backbone.
<strong>3)</strong> <i>Pre-Training:</i> We propose a novel pre-training strategy for LMM, which employs variable frame rates and masking forms, to better exploit knowledge from diverse training data.
Extensive experiments demonstrate that our generalist LMM achieves competitive performance across various standard motion generation tasks over state-of-the-art specialist models. Notably, LMM exhibits strong generalization capabilities and emerging properties across many unseen tasks.


## Updates

[12/2024] Release code for [LMM](https://mingyuan-zhang.github.io/projects/LMM.html), [FineMoGen](https://mingyuan-zhang.github.io/projects/FineMoGen.html), [MoMat-MoGen](https://digital-life-project.com/), [ReMoDiffuse](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html) and [MotionDiffuse](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html)

## Benchmark and Model Zoo

#### Supported methods

- [x] [MotionDiffuse](https://mingyuan-zhang.github.io/projects/ReMoDiffuse.html)
- [x] [MDM](https://guytevet.github.io/mdm-page/)
- [x] [ReMoDiffuse](https://mingyuan-zhang.github.io/projects/MotionDiffuse.html)
- [x] [MoMat-MoGen](https://digital-life-project.com/)
- [x] [FineMoGen](https://mingyuan-zhang.github.io/projects/FineMoGen.html)
- [x] [LMM](https://mingyuan-zhang.github.io/projects/LMM.html)


## Citation

If you find our work useful for your research, please consider citing the paper:

```
@inproceedings{zhang2025large,
  title={Large motion model for unified multi-modal motion generation},
  author={Zhang, Mingyuan and Jin, Daisheng and Gu, Chenyang and Hong, Fangzhou and Cai, Zhongang and Huang, Jingfang and Zhang, Chongzhi and Guo, Xinying and Yang, Lei and He, Ying and others},
  booktitle={European Conference on Computer Vision},
  pages={397--421},
  year={2025},
  organization={Springer}
}
@article{zhang2023finemogen,
  title={Finemogen: Fine-grained spatio-temporal motion generation and editing},
  author={Zhang, Mingyuan and Li, Huirong and Cai, Zhongang and Ren, Jiawei and Yang, Lei and Liu, Ziwei},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={13981--13992},
  year={2023}
}
@article{zhang2023remodiffuse,
  title={ReMoDiffuse: Retrieval-Augmented Motion Diffusion Model},
  author={Zhang, Mingyuan and Guo, Xinying and Pan, Liang and Cai, Zhongang and Hong, Fangzhou and Li, Huirong and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2304.01116},
  year={2023}
}
@article{zhang2022motiondiffuse,
  title={MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model},
  author={Zhang, Mingyuan and Cai, Zhongang and Pan, Liang and Hong, Fangzhou and Guo, Xinying and Yang, Lei and Liu, Ziwei},
  journal={arXiv preprint arXiv:2208.15001},
  year={2022}
}
```

## Installation

```shell
# Create Conda Environment
conda create -n mogen python=3.9 -y
conda activate mogen

# Install Pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

# Install MMCV
pip install "mmcv-full>=1.4.2,<=1.9.0" -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.1/index.html

# Install Pytorch3d
conda install -c bottler nvidiacub -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install pytorch3d -c pytorch3d -y

# Install tutel
python3 -m pip install --verbose --upgrade git+https://github.com/microsoft/tutel@main

# Install other requirements
pip install -r requirements/mogen.txt

# Install ImageBind
pip install --no-deps git+https://github.com/facebookresearch/ImageBind@main
```

## Data Preparation

Please kindly refer to the [documentation](https://lmm.readthedocs.io/en/latest/datasets/motionverse.html) for the detailed instruction.

## Model Inference

You may try our oneline demo on [Hugging Face](https://huggingface.co/spaces/mingyuan/LMM). Also you can download the pretrained weights form [google drive](https://drive.google.com/drive/folders/1FBu7AtKRKesUu6q4CNa4aq451BpEvy1O?usp=drive_link) and run the visualization script locally:
```shell
PYTHONPATH=".":$PYTHONPATH python tools/visualize_lmm.py ${CONFIG} ${CHECKPOINT} \
    --text ${TEXT} \
    --speech ${SPEECH_WAV_PATH} \
    --motion_length ${MOTION_LENGTH} \
    --out ${OUTPUT_ANIMATION_PATH} \
    --fps 20.0 \
    --device cpu
```