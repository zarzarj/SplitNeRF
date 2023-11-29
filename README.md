# SplitNeRF 
We provide the code for SplitNeRF

## Run
### Training on NeRF-Synthetic
Download the NeRF-Synthetic data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put it under `load/`. The file structure should be like `load/nerf_synthetic/lego`.

Run the launch script with `--train`, specifying the config file, the GPU(s) to be used (GPU 0 will be used by default), and the scene name:
```bash
# train NeRF
python launch.py --config configs/neus-split-blender_paper.yaml --gpu 0 --train dataset.scene=lego tag=example



## Citation
This code is built on top ofInstant NSR:
```
@misc{instant-nsr-pl,
    Author = {Yuan-Chen Guo},
    Year = {2022},
    Note = {https://github.com/bennyguo/instant-nsr-pl},
    Title = {Instant Neural Surface Reconstruction}
}
```
We would like to thank the authors for their codebase.