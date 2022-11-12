# ASTNet: Attention-based Residual Autoencoder for Video Anomaly Detection

This is the official implementation of **Attention-based Residual Autoencoder for Video Anomaly Detection**.

### [Project](https://vt-le.github.io/astnet/) | [Video](https://youtu.be/XOzXwKVKX-Y) | [Paper](http://dx.doi.org/10.1007/s10489-022-03613-1)
  
 <img src='static/img/shanghai/shanghai_curve_full.gif' align="center" width="70%">

## Updates
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
    cd astnet

## Testing
Please first download the pre-trained model

Dataset          | Pretrained Model |
-----------------|:--------:
UCSD Ped2   | [github][1] / [drive][4]
CUHK Avenue  | [github][2] / [drive][5]
ShanghaiTech Campus        | [github][3] / [drive][6]

[1]: https://github.com/vt-le/storage/raw/wresnet/ped2.pth
[2]: https://github.com/vt-le/storage/raw/wresnet/avenue.pth
[3]: https://github.com/vt-le/storage/raw/wresnet/shanghai.pth
[4]: https://drive.google.com/file/d/1pxX_lDeB6RNoEdtylKwN5eRD_E8_Bl54/view?usp=sharing
[5]: https://drive.google.com/file/d/1NW_x2g8UbtWJHPuniroEt2NEbR03W1XQ/view?usp=sharing
[6]: https://drive.google.com/file/d/1efayfuAsXZUEE4S0_Kq4yRFi_JirON9T/view?usp=sharing



After preparing a dataset, you can test the dataset by running:
    
    python astnet.py \
        --cfg /path/to/config/file \
        --model-file /path/to/pre-trained/model \
        GPUS [{GPU_index}]        
 
## Datasets
* <a href="http://www.svcl.ucsd.edu/projects/anomaly/dataset.html" target="_blank">UCSD Ped2</a>
* <a href="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html" target="_blank">CUHK Avenue</a>
* <a href="https://svip-lab.github.io/dataset/campus_dataset.html" target="_blank">ShanghaiTech Campus</a>

A dataset is a directory with the following structure:

    dataset
        ├── train
        │   └── ${video_id}$
        |       └──${frame_id}$.jpg
        ├── test
        │   └── ${video_id}$
        |       └──${frame_id}$.jpg
        └── $dataset$.mat

 
## Citing
If you find our work useful for your research, please consider citing:
```BibTeX
@article{le2022attention,
  title={Attention-based residual autoencoder for video anomaly detection},
  author={Le, Viet-Tuan and Kim, Yong-Guk},
  journal={Applied Intelligence},
  pages={1--15},
  year={2022},
  publisher={Springer}
}
```

## Contact
For any question, please file an [issue](https://github.com/vt-le/astnet/issues) or contact:

    Viet-Tuan Le: tuanlv@sju.ac.kr

## Other Links
> **Transfomer for Video Anomaly Detection**: See [PySTformer: Pyramidal Spatio-Temporal Transformer for Video Anomaly Detection](https://github.com/vt-le/pystformer).


