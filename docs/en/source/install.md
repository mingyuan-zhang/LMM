# Installation

<!-- TOC -->

<!-- - [Requirements](#requirements)
- [Prepare environment](#prepare-environment)
- [Install MMHuman3D](#install-mmhuman3d)
- [A from-scratch setup script](#a-from-scratch-setup-script) -->

<!-- TOC -->

## Requirements

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
pip install -r requirements.txt

# Install ImageBind
pip install --no-deps git+https://github.com/facebookresearch/ImageBind@main
```