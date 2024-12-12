
## NuScenes
Download nuScenes V1.0 full dataset data.

For Occupancy Prediction Task, you need to download extra annotation from
https://github.com/Tsinghua-MARS-Lab/Occ3D



**Prepare nuScenes data**

*We genetate custom annotation files which are different from original BEVDet's*
```
python tools/create_data_bevdet.py
```


**ckpts**

*Pretrained model weights*


- get ``bevdet-r50-4d-depth-cbgs.pth`` from https://github.com/HuangJunJie2017/BEVDet

- put ``bevdet-r50-4d-depth-cbgs.pth`` into ``ckpts`` folder


**Folder structure**
```
GSD-OCC
├── mmdet3d/
├── tools/
├── configs/
├── ckpts/
│   ├── bevdet-r50-4d-depth-cbgs.pth
    ├── resnet50-0676ba61.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/  # ln -s occupancy gts to this location
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── bevdetv2-nuscenes_infos_val.pkl
|   |   ├── bevdetv2-nuscenes_infos_train.pkl
```