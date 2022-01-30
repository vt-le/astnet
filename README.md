# ASTNet: Attention-based Residual Autoencoder for Video Anomaly Detection

This is the official implementation of **Attention-based Residual Autoencoder for Video Anomaly Detection**.

### [Project](https://vt-le.github.io/astnet/) | [Video](https://youtu.be/XOzXwKVKX-Y) | [Paper](#)
  
 <img src='img/shanghai/shanghai_curve_full.gif' align="center" width="70%">
 
## Prerequisites
  * Linux or macOS
  * Python 3
  * PyTorch 1.7.0

## Setup
The code can be run with Python 3.8 and above.

Install the required packages:

    pip install -r requirements.txt
    
Clone this repo:

    git clone https://github.com/vt-le/astnet.git
    cd astnet

## Testing
Please first download the pre-trained model

Dataset          | Pretrained Model |
-----------------|:--------:
UCSD Ped2   | [github][1] / drive
CUHK Avenue  | [github][2] / drive
ShanghaiTech Campus        | [github][3] / drive

[1]: https://github.com/vt-le/storage/raw/wresnet/ped2.pth
[2]: https://github.com/vt-le/storage/raw/wresnet/avenue.pth
[3]: https://github.com/vt-le/storage/raw/wresnet/shanghai.pth



After preparing a dataset, you can test the dataset by running:
    
    python test.py \
        --cfg /path/to/config/file \
        --model-file /path/to/pre-trained/model \
        GPUS [0]        
 
## Datasets
* <a href="http://www.svcl.ucsd.edu/projects/anomaly/dataset.html" target="_blank">UCSD Ped2</a>
* <a href="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html" target="_blank">CUHK Avenue</a>
* <a href="https://svip-lab.github.io/dataset/campus_dataset.html" target="_blank">ShanghaiTech Campus</a>

A dataset is a directory with the following structure:

    dataset
        ├── train
        │   └── ${video_id}
        |       └──${frame_id}.jpg
        ├── test
        │   └── ${video_id}
        |       └──${frame_id}.jpg
        └── $dataset$.mat

 
## Citing
If you find our work useful for your research, please consider citing:
```BibTeX
@article{le2022attention
  author    = {Viet-Tuan, Le 
               and Yong-Guk, Kim},
  title     = {Attention-based Residual Autoencoder for Video Anomaly Detection},
  journal   = {...},
  year      = {2022},
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/astnet/issues) or contact:

    Viet-Tuan Le: tuanlv@sju.ac.kr

