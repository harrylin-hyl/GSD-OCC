# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n gsdocc python=3.8 -y
conda activate gsdocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu114  -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.12
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.5.2

```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.24.0
pip install mmsegmentation==0.24.0
```

**e. Install GSD-OCC.**
```shell
pip install -e .
# python setup.py install
```

