# SplitNeRF 
We provide the code for SplitNeRF. [Project Page](https://zarzarj.github.io/splitnerf.github.io/) [arXiv](https://arxiv.org/abs/2311.16671)

## Setup
Download and extract [OptiX 7.6.0 SDK](https://developer.nvidia.com/optix/downloads/7.6.0/linux64-x86_64).
Setup environment variables (required for installing python-optix)
```
export OPTIX_PATH=/path/to/optix
export CUDA_PATH=/path/to/cuda_toolkit
export OPTIX_EMBED_HEADERS=1 # embed the optix headers into the package
```

Install pytorch and other requirements:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  
pip install -r requirements.txt
```


## Run
### Training on NeRF-Synthetic
Download the NeRF-Synthetic data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put it under `load/`. The file structure should be like `load/nerf_synthetic/lego`.

Run the launch script with `--train`, specifying the config file, the GPU(s) to be used (GPU 0 will be used by default), and the scene name:
```bash
# train SplitNeRF
python launch.py --config configs/splitnerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=example
```


## Citation
This code is built on top of Instant NSR. We would like to thank the authors for their [codebase](https://github.com/bennyguo/instant-nsr-pl).

```
@misc{zarzar2023splitnerf,
      title={SplitNeRF: Split Sum Approximation Neural Field for Joint Geometry, Illumination, and Material Estimation}, 
      author={Jesus Zarzar and Bernard Ghanem},
      year={2023},
      eprint={2311.16671},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
