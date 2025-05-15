# MRA-Net

## Our Framework
<p align="center">
  <img src="img/model.png" width="80%"/>
</p>

## Using the code:
The code is stable while using Python 3.8, PyTorch 1.13.0, CUDA =11.7.

## Main environments:
```
conda create -n mranet python=3.8
conda activate mranet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install pytest chardet yacs termcolor
pip install timm==0.4.12
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## Data Presentation and Model Parameters
- We provide 200 sample images from the constructed HanBamboo dataset, including the original images, the annotated ground truth images, and the binarization results produced by our model
, at the [data](https://drive.google.com/drive/folders/1xR2LapoqvvTGaECGuFBfOmhBU7McmMcN?usp=drive_link).
- Furthermore, we have provided the optimal model [parameters](https://drive.google.com/drive/folders/1-vBfPsdzo6wrwC0DyGDNFQfdXN6Ypdn1?usp=drive_link).

## DataFormat
Make sure to put the files as the following structure:
```
dataset
├── train
|   ├── A
|   │   ├── ...
|   │
|   └── B
|       ├── ...
|
└── test
    ├── A
    |   ├── ...
    |
    └── B
        ├── ...
```

## Training and Validation
### 1) Train the model.
```
python train.py --dataroot path_dataset --lr 0.0002 --n_epochs 100 --size 256 --batchSize 4
```

### 2) Test.
```
python test.py --dataroot path_dataset --size 256 --batchSize 4
```

## Comparison with other methods
<p align="center">
  <img src="imgs/Comparison.png" width="60%"/>
</p>

## Visual comparison
<p align="center">
  <img src="imgs/Visual result.png"/>
</p>
