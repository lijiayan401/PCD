#  Progressive Class-level Distillation

## Environment

Python 3.8, torch 1.7.0

A selection of packages that may require additional installation: torchvision, tensorboardX, yacs, wandb, tqdm, scipy

## Pre-trained Teachers

Pre-trained teachers can be downloaded from [Decoupled Knowledge Distillation (CVPR 2022)](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints). Download the `cifar_teachers.tar` and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

## Training on CIFAR-100

```sh
# Train method X with the following code
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/X/vgg13_vgg8.yaml
# You can refer to the following code to additionally specify hyper-parameters
# Train DKD
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml DKD.ALPHA 1. DKD.BETA 8. DKD.T 4.
# Train PCD
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/pcd/vgg13_vgg8.yaml --same-t PCD.ALPHA 1. PCD.STEPS 3. PCD.T 4.
```
##Citation
@misc{li2025progressiveclassleveldistillation,
      title={Progressive Class-level Distillation}, 
      author={Jiayan Li and Jun Li and Zhourui Zhang and Jianhua Xu},
      year={2025},
      eprint={2505.24310},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.24310}, 
}
