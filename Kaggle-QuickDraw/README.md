# Kaggle-QuickDraw
PyTorch implementation for [Kaggle Quick, Draw! Doodle Recognition Challenge](https://www.kaggle.com/c/quickdraw-doodle-recognition)


## Training
* Model
    - MobileNetV2 [1] (light-weight, pretrained on ImageNet) with Convolutional Block Attention Module (CBAM) [2]
* Augmentation
    - Random horizontally flip
    - Random affine transformation (rotation, translation, scale)
* Batch size: 512 * 2
* Optimizer: Adam
* Learning rate: cosine annealing from 1e-3 to 1e-5
* Weight decay: 1e-4
* Loss function: cross-entropy
* Using 300/class for validation, the remainings are training data (splitted into 10 folds), 5 epochs
* Final prediction: a single fold model with TTA (horizontal flip), 0.92860 on private LB / 0.92930 on public LB


## Requirements
* pytorch 0.4.1
* torchvision 0.2.1
* numpy
* opencv
* pandas
* tqdm

`pip install -r requirements.txt`


## Usage

### Data
* Download data from [Kaggle QuickDraw competition](https://www.kaggle.com/c/quickdraw-doodle-recognition/data)
* Extract train_simplified.zip
* Modify the path appropriately in `config.json`
* Run `PYTHONPATH=. python loaders/quickdraw_loader.py` first to generate training/validation pickle files

### To train/test the model
`python [train, test].py -h` for more details


## Reference
[1] [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

[2] [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

[3] Pytorch implementation for [MobileNetV2](https://github.com/tonylins/pytorch-mobilenet-v2)

[4] Official implementation for [CBAM](https://github.com/Jongchan/attention-module)
