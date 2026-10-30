# MSSF inference environment

Target: Ubuntu x86_64, RTX 4090 D, Python 3.9, CUDA runtime 11.8. No training is required.

## Activate

```bash
conda activate mssf-bench
cd '/home/d514/桌面/zcy/MSSF_zcy'
```

## Rebuild

1. Create the Conda environment using `environment.yml`. CUDA toolkit 11.8 supplies the NVVM/libdevice runtime used by the Numba evaluation kernel; it is separate from the installed NVIDIA driver.
2. Install these matching prebuilt CUDA wheels (Linux x86_64, CPython 3.9):
   - torch 2.0.1+cu118 and torchvision 0.15.2+cu118: https://download.pytorch.org/whl/cu118/
   - mmcv 2.0.1: https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
   - torch-scatter 2.1.2+pt20cu118: https://data.pyg.org/whl/torch-2.0.0+cu118.html
3. Run `python -m pip install -r repro/mssf-bench/requirements.txt`.
4. From the repository root run `python -m pip install -e . --no-build-isolation --no-deps`.
5. Run `python repro/mssf-bench/verify_environment.py`.

The exact installed versions are exported to `pip-freeze.txt` and `conda-explicit.txt` after validation. CUDA-wheel source URLs and checksums are recorded in `cuda-wheels.json`.

## Compatibility adjustment

`CalFLOPsHook` imports an author-customized THOP `profile_acts` function that standard THOP does not provide. Its import is deferred until this optional hook is actually invoked, so normal model registration and evaluation can work with standard THOP. Network layers, checkpoint tensors and evaluation rules are unchanged. The optional custom FLOPs hook itself still requires the author's THOP extension.

## Validation scope

The verification script checks GPU tensors, MMCV voxelization and deformable attention, spconv, torch-scatter, MSSF module registration, the Numba rotated-IoU kernel, and strict loading of the official MSSF-PP/VoD checkpoint. These checks do not measure accuracy, FPS or dataset completeness. Full validation-set evaluation follows dataset preparation and the separate evaluation-entry fixes.

## Numba and recent NVIDIA drivers

The server uses NVIDIA driver 595.99.02. Older Numba ctypes bindings can select the newer two-argument cuCtxGetDevice_v2 with a one-argument prototype. This environment keeps Numba 0.57.1 and enables NVIDIA CUDA Python 11.8.3 bindings through the Conda environment variable NUMBA_CUDA_USE_NVIDIA_BINDING=1. It must be set before importing Numba. Conda activation applies it automatically.

References:
- https://numba.readthedocs.io/en/0.57.1/cuda/bindings.html
- https://github.com/numba/numba/issues/10188
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CTX.html

The missing requirements/ files were restored from MMDetection3D v1.2.0, matching mmdet3d/version.py. Open3D 0.16.0 requires nbformat 5.5.0; nbconvert is pinned to 7.2.10 to keep package metadata consistent.

The Conda environment also sets PYTHONDONTWRITEBYTECODE=1 because the upstream repository tracks CPython bytecode caches. This keeps runtime cache updates out of source diffs.

The pip snapshot includes bootstrap tools and normalizes local CUDA-wheel references to their official HTTPS URLs with SHA-256. Run pip-freeze.txt from the repository root. The conda-explicit.txt CUDA archive URL is likewise normalized to conda-forge.
