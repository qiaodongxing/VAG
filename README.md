# [VAG: Voxel Attenuation Grid For Sparse-View CBCT Reconstruction](https://ieeexplore.ieee.org/document/10647311)

Sparse view CBCT reconstruction has become one of the important research fields to reduce the radiation impact of CT scanning. However, the reconstruction of high-quality 3D CT volumes from sparse and noisy CBCT data still faces challenges such as slow convergence, long computation time, and increased noise. In light of these issues, we propose a voxel attenuation grid representation to explicitly model the attenuation field of the 3D CT volume. Since this representation does not involve the implementation of neural networks, our method for reconstruction is extremely fast. Furthermore, trim regularization and total variation regularization terms are introduced on top of the mean square error loss to optimize the voxel attenuation grid and significantly reduce the noise in the reconstructed 3D CT volume.
![The Pipeline of the proposed VAG method](https://github.com/user-attachments/assets/af98d92c-4778-46bd-b79f-910454b2021b)

This is a sparse CBCT reconstruction work, and the code is being updated


## Setup
We recommend setup with a conda environment, using the packages provided in requirements.txt.

## Reconstruction
```python plenoptimize.py```

## Data
You can process your own CT data following the instructions in [NAF project](https://github.com/Ruyi-Zha/naf_cbct/tree/main?tab=readme-ov-file)

## Acknowledgement
JAX implementation and code structure are adapted from[here](https://github.com/sarafridov/plenoxels).
Many thanks to the amazing TIGRE [toolbox](https://github.com/CERN/TIGRE).
The data processing borrows from the [NAF project](https://github.com/Ruyi-Zha/naf_cbct/tree/main?tab=readme-ov-file)
