# RFMNet: Referring Camouflaged Object Detection

Official code for our paper:

**Referring Camouflaged Object Detection With Multi-Context Overlapped Windows Cross-Attention**  
Paper link: https://arxiv.org/abs/2511.13249

This repository provides the training and testing code for `RFMNet` on the **R2C7K** dataset.

## Highlights

- Official public code release for the paper
- Support for both training and testing
- Support for CPU and CUDA environments
- Includes checkpoint conversion utility for the current model definition

## Repository Structure

```text
.
├── data/
├── snapshot/
│   ├── base/
│   └── saved_model/
├── utils/
├── requirements.txt
├── rfmnet.py
├── train.py
└── test.py
```

## Environment

The current code is aligned with:

- Python `3.8`
- PyTorch `2.0.0`
- torchvision `0.15.1`

### Install with CPU

```bash
conda create -n rfmnet-cpu python=3.8 -y
conda activate rfmnet-cpu

pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Install with CUDA

The following setup matches the current project environment most closely.

```bash
conda create -n rfmnet python=3.8 -y
conda activate rfmnet

conda install -y cudatoolkit=11.8 cudnn=8.9.2 -c defaults
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Dataset

This project uses the **R2C7K** dataset from the RefCOD project.

- RefCOD project: https://github.com/zhangxuying1004/RefCOD
- R2C7K download: https://pan.baidu.com/s/1LHdqpD3w24fcLb_dbR6DyA
- Access code: `2013`

Please organize the dataset as:

```text
R2C7K/
├── Camo/
│   ├── train/
│   └── test/
└── Ref/
    ├── Images/
    ├── RefFeat_ICON-R/
    └── Saliency_ICON-R/
```

By default, the code uses:

```text
./R2C7K/
```

If needed, change the `--data_root` argument in `train.py` or `test.py`.

## Weights

### Pretrained weights and trained weights

We provide the weights via Quark Netdisk:

- Link: https://pan.quark.cn/s/44001aad115b
- Access code: `KLJ9`

### RFMNet-Maps

For visualization comparison and further analysis, we also provide the predicted maps:

- Link: https://pan.quark.cn/s/d1502adbc943
- Access code: `zr5Q`

### Weight placement

Please place the files as follows:

```text
snapshot/
├── base/
│   ├── swins_cod_base_45_416.pth
│   └── swins_base_sod_45.pth
└── saved_model/
    └── rfmnet.pth
```

Meaning:

- Put **pretrained backbone weights** into `snapshot/base/`
- Put the **trained RFMNet checkpoint** `rfmnet.pth` into `snapshot/saved_model/`

## Training

Run training with:

```bash
python train.py
```

Example:

```bash
python train.py \
  --epoch 500 \
  --lr_0 1.5e-4 \
  --batchsize 1 \
  --trainsize 512 \
  --shot 1 \
  --dim 64 \
  --data_root ./R2C7K/ \
  --ckpt_path ./ckpt \
  --exp_name rfmnet
```

Training checkpoints will be saved to:

```text
./ckpt/<exp_name>/
```

For example:

```text
./ckpt/rfmnet/rfmnet_50.pth
./ckpt/rfmnet/rfmnet_100.pth
...
```

## Testing

Run testing with:

```bash
python test.py
```

Example:

```bash
python test.py \
  --dim 64 \
  --testsize 512 \
  --shot 1 \
  --data_root ./R2C7K/ \
  --save_root ./snapshot/
```

By default, `test.py` loads:

```text
./snapshot/saved_model/rfmnet.pth
```

So before testing, make sure `rfmnet.pth` is placed in:

```text
snapshot/saved_model/
```

## Notes

- `train.py` loads backbone initialization files from `snapshot/base/`
- `test.py` loads the final trained checkpoint from `snapshot/saved_model/rfmnet.pth`
- If you change file names or paths, please update them in the scripts accordingly

## Citation

If you find this repository useful, please cite our paper.

```bibtex
@article{wen2025referring,
  title={Referring Camouflaged Object Detection With Multi-Context Overlapped Windows Cross-Attention},
  author={Wen, Yu and Gao, Shuyong and Zhang, Shuping and Huang, Miao and Tao, Lili and Yang, Han and Xing, Haozhe and Zhang, Lihe and Hou, Boxue},
  journal={arXiv preprint arXiv:2511.13249},
  year={2025}
}
```

## Acknowledgement

- RefCOD: https://github.com/zhangxuying1004/RefCOD
- R2C7K dataset is built upon the RefCOD project resources
