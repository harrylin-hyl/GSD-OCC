# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train GSD-OCC with 8 GPUs 
```
./tools/dist_train.sh ./occupancy_configs/gsd_occ/gsdocc_r50_256x704_16f_8x4_24.py 8

./tools/dist_train.sh ./occupancy_configs/gsd_occ/gsdocc_r50_512x1408_16f_8x4_24e.py 8
```

Eval GSD-OCC with 8 GPUs
```
./tools/dist_test.sh./occupancy_configs/gsd_occ/gsdocc_r50_256x704_16f_8x4_24.py ./path/to/ckpts.pth 8

./tools/dist_test.sh./occupancy_configs/gsd_occ/gsdocc_r50_512x1408_16f_8x4_24e.py ./path/to/ckpts.pth 8
```

# Visualization
```
python tools/dist_test.sh $config $checkpoint --out $savepath
python tools/analysis_tools/vis_frame.py $savepath $config --save-path $scenedir --scene-idx $sceneidx --vis-gt
python tools/analysis_tools/generate_gifs.py --scene-dir $scenedir
```