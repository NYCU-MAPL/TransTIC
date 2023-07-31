# TransTIC: Transferring Transformer-based Image Compression from Human Visualization to Machine Perception
Accpeted to ICCV 2023

This repository contains the source code of our ICCV 2023 paper **TransTIC** [arXiv](https://arxiv.org/abs/2306.05085).

## Abstract
>This work aims for transferring a Transformer-based image compression codec from human vision to machine perception without fine-tuning the codec. We propose a transferable Transformer-based image compression framework, termed TransTIC. Inspired by visual prompt tuning, we propose an instance-specific prompt generator to inject instance-specific prompts to the encoder and task-specific prompts to the decoder. Extensive experiments show that our proposed method is capable of transferring the codec to various machine tasks and outshining the competing methods significantly. To our best knowledge, this work is the first attempt to utilize prompting on the low-level image compression task.

## Install

```bash
git clone https://github.com/NYCU-MAPL/TransTIC
cd TransTIC
pip install -U pip && pip install -e .
pip install timm tqdm click
```
Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for object detection and instance segementation.

## Dataset
The following datasets are used and needed to be downloaded.
- Flicker2W (download [here](https://github.com/liujiaheng/CompressionData), and use [this script](https://github.com/xyq7/InvCompress/tree/main/codes/scripts) for preprocessing)
- ImageNet1K
- COCO 2017 Train/Val
- Kodak

## Example Usage
Specify the data paths, target rate point, corresponding lambda, and checkpoint in the config file accordingly.

### Base Codec (for PSNR)
`python examples/train.py -c config/base_codec.yaml`

### Classification
`python examples/classification.py -c config/classification.yaml`<br>
Add argument `-T` for evaluation.

### Object Detection
`python examples/detection.py -c config/detection.yaml`<br>
Add argument `-T` for evaluation.

### Instance Segmentation

## Pre-trained Weights
|         Tasks         |       |       |       |       |
|:---------------------:|-------|-------|-------|-------|
|     Base codec (TIC)  | [1](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_1.pth.tar) | [2](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_2.pth.tar) | [3](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_3.pth.tar) | [4](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/base_codec_4.pth.tar) |
|     Classification    | [1](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/cls_1.pth.tar) | [2](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/cls_2.pth.tar) | [3](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/cls_3.pth.tar) | [4](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/cls_4.pth.tar) |
|    Object Detection   | [1](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/det_1.pth.tar) | [2](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/det_2.pth.tar) | [3](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/det_3.pth.tar) | [4](https://github.com/NYCU-MAPL/TransTIC/releases/download/v1.0/det_4.pth.tar) |
| Instance Segmentation | [1]() | [2]() | [3]() | [4]() |

## Citation
If you find our project useful, please cite the following paper.
```
@inproceedings{TransTIC,
  title={TransTIC: Transferring Transformer-based Image Compression from Human Visualization to Machine Perception},
  author={Chen, Yi-Hsin and Weng, Ying-Chieh and Kao, Chia-Hao and Chien, Cheng and Chiu, Wei-Chen and Peng, Wen-Hsiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={},
  year={2023}
}
```

## Ackownledgement
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI). The base codec is adopted from [TIC](https://github.com/lumingzzz/TIC)/[TinyLIC](https://github.com/lumingzzz/TinyLIC) and the prompting method is modified from [VPT](https://github.com/KMnP/vpt). We thank the authors for open-sourcing their code.
