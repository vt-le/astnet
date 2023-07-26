# ASTNet: Attention-based Residual Autoencoder for Video Anomaly Detection

This is the official implementation of **[Attention-based Residual Autoencoder for Video Anomaly Detection](https://doi.org/10.1007/s10489-022-03613-1)**.

> **Transfomer for Video Anomaly Detection**: See [HSTforU: Anomaly Detection in Aerial and Ground-based Videos with Hierarchical Spatio-Temporal Transformer for U-net
](https://vt-le.github.io/HSTforU/).

### [Project](https://vt-le.github.io/astnet/) | [Video](https://youtu.be/XOzXwKVKX-Y) | [Paper](http://dx.doi.org/10.1007/s10489-022-03613-1)
  
 <img src='static/img/shanghai/shanghai_curve_full.gif' align="center" width="70%">

## Updates
* [6/01/2023] Training script of [ASTNet](https://vt-le.github.io/astnet/) is released.
* [5/25/2022] [ASTNet](https://vt-le.github.io/astnet/) is available online.
* [4/21/2022] Code of [ASTNet](https://vt-le.github.io/astnet/) is released!

## Prerequisites
  * Linux or macOS
  * Python 3
  * PyTorch 1.7.0

## Setup
The code can be run with Python 3.6 and above.

Install the required packages:

    pip install -r requirements.txt
    
Clone this repo:

    git clone https://github.com/vt-le/astnet.git
    cd ASTNet/ASTNet

## Data preparation
We evaluate `ASTNet` on:
* <a href="http://www.svcl.ucsd.edu/projects/anomaly/dataset.html" target="_blank">UCSD Ped2</a>
* <a href="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html" target="_blank">CUHK Avenue</a>
* <a href="https://svip-lab.github.io/dataset/campus_dataset.html" target="_blank">ShanghaiTech Campus</a>

A dataset is a directory with the following structure:
  ```bash
  $ tree data
  ped2/avenue
  ├── training
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 00.jpg
  │       │   └── ...
  │       └── ...
  ├── testing
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 000.jpg
  │       │   └── ...
  │       └── ...
  └── ped2/avenue.mat
  
  shanghaitech
  ├── training
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 00.jpg
  │       │   └── ...
  │       └── ...
  ├── testing
  │   └── frames
  │       ├── ${video_1}$
  │       │   ├── 000.jpg
  │       │   ├── 001.jpg
  │       │   └── ...
  │       ├── ${video_2}$
  │       │   ├── 000.jpg
  │       │   └── ...
  │       └── ...
  └── test_frame_mask
      ├── 01_0014.npy
      ├── 01_0015.npy
      └── ...
  ```

## Evaluation
Please first download the pre-trained model

Dataset          | Pretrained Model |
-----------------|:--------:
UCSD Ped2   | [github][1] / [drive][4]
CUHK Avenue  | [github][2] / [drive][5]
ShanghaiTech        | [github][3] / [drive][6]

[1]: https://github.com/vt-le/storage/raw/wresnet/ped2.pth
[2]: https://github.com/vt-le/storage/raw/wresnet/avenue.pth
[3]: https://github.com/vt-le/storage/raw/wresnet/shanghai.pth
[4]: https://drive.google.com/file/d/1dmDVyAkI0FiEex3aEHDQMdgDFeBtH1Fo/view?usp=sharing
[5]: https://drive.google.com/file/d/1EtHqDiq3tDikgpXCm6LltWzgSnSvyH1U/view?usp=sharing
[6]: https://drive.google.com/file/d/1SwUGiwyhEUPIk8CTSS_5ZDjg3Idi7CJS/view?usp=sharing


To evaluate a pretrained `ASTNet` on a dataset, run:

```bash
 python test.py \
    --cfg <path/to/config/file> \
    --model-file </path/to/pre-trained/model>
```      
 
 For example, to evaluate `ASTNet` on Ped2:

```bash
python test.py \
    --cfg config/ped2_wresmet.yaml \
    --model-file pretrained.ped2.ptn
```

## Training from scratch
To train `ASTNet` on a dataset, run:
```bash
python train.py \
    --cfg <path/to/config/file>
```
For example, to train `ASTNet` on Ped2:

```bash
python train.py \
    --cfg config/ped2_wresmet.yaml
```

**Notes**:
- To change other options, see `<config/config_file.yaml>`.


## Citing
If you find our work useful for your research, please consider citing:
```BibTeX
@article{le2023attention,
  title={Attention-based residual autoencoder for video anomaly detection},
  author={Le, Viet-Tuan and Kim, Yong-Guk},
  journal={Applied Intelligence},
  volume={53},
  number={3},
  pages={3240--3254},
  year={2023},
  publisher={Springer}
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/astnet/issues) or contact:

    Viet-Tuan Le: tuanlv@sju.ac.kr



