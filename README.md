# DesNet: Attention-based Residual Autoencoder for Video Anomaly Detection

This is the official code for DesNet: Attention-based Residual Autoencoder for Video Anomaly Detection.

 * [Project Page](https://vt-le.github.io/desnet/)
 * [Paper](#)
 * [Video](https://youtu.be/ghwkkJioeBc)
 
 
## Setup
The code can be run under any environment with Python 3.8 and above.

Install the required packages:

    pip install -r requirements.txt
    
Clone this repo:

    git clone https://github.com/vt-le/desnet.git
    cd desnet

## Testing
Please first download the pre-trained model

Dataset          | Pre-trained Mode |
-----------------|:--------:
UCSD Ped2   | [link][1]
CUHK Avenue  | [link][2]
ShanghaiTech Campus        | [link][3]

[1]: #
[2]: #
[3]: #


After preparing a dataset, you can test the dataset by running:
    
    python train.py \
        --cfg /path/to/config/file \
        --model-file /path/to/pre-trained/model \
        GPUS [0]        
 
## Datasets
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
@article{le2021attention
  author    = {Viet-Tuan, Le 
               and Yong-Guk, Kim},
  title     = {Attention-based Residual Autoencoder for Video Anomaly Detection},
  journal   = {...},
  year      = {2021},
}
```
