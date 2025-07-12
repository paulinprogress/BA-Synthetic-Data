Bachelor's thesis for B.Sc. Media Technology at [HAW Hamburg](https://www.haw-hamburg.de/)
# Contrastive Learning with Stable Diffusion-based Data Augmentation: Improving Image Classification with Synthetic Data

A [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) classifier was trained using synthetic data generated with [DA-Fusion](https://arxiv.org/abs/2302.07944) - a Stable Diffusion-based data augmentation method that can generate semantically meaningful variations of images.

DA-Fusion was used to generate both in-distribution and (near) out-of-distribution (OOD) data by adjusting the augmentation strength. The OOD data only serve as negative examples for contrastive learning, with the goal of further improving the representations of the ID data.

The experiments showed that synthetic ID data improved classification, but OOD data did not.

## Installation

```bash
conda create -n synt-contrast python=3.7 pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6
conda activate synt-contrast
pip install diffusers["torch"] transformers pycocotools pandas matplotlib seaborn scipy
pip install -e da-fusion
pip install --upgrade huggingface_hub
huggingface-cli login
```

(Conda channels: `nvidia`, `pytorch`, `conda-forge`)

## Usage

Complete pipeline for the [MVIP dataset](https://fordatis.fraunhofer.de/handle/fordatis/358):

- `mvip_generate.augs.sh` for generating synthetic ID & OOD augmentations
- `mvip_run_experiments.sh` for executing the three different training runs with Supervised Contrastive Learning, examining the impact of the augmentations on classification performance

You can find the more detailed READMEs (modified for this project) here:

- [DA-Fusion](da_fusion/README.md)
- [Supervised Contrastive Learning](sup_contrast/README.md)