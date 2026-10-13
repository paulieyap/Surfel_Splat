# SurfelSplat environment.
# The README pins torch 2.1.2 + CUDA 11.8, which cannot run on Blackwell GPUs
# (RTX 50xx, sm_120). This image uses CUDA 12.8 + torch 2.7.1 instead.
#
# Build and run (code bind-mounted, checkpoints/datasets/outputs in a named volume):
#   docker build -t surfelsplat:cu128 .
#   docker volume create surfelsplat-data
#   docker run -d --gpus all --name surfelsplat --restart unless-stopped --shm-size 16g \
#     -v "$PWD":/workspace/Surfel_Splat -v surfelsplat-data:/workspace/data \
#     surfelsplat:cu128 sleep infinity
#   docker exec -it surfelsplat bash
# Inside the container, link checkpoints/, datasets/, outputs/ and point_clouds/
# to the matching folders under /workspace/data so large files stay in the volume.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Compile the CUDA extensions for Ampere/Ada/Hopper/Blackwell.
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0" \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python-is-python3 \
        git wget ffmpeg ninja-build build-essential \
        libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libxrender1 libsm6 libegl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip "setuptools<82" wheel==0.43.0

RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128

# Pure-Python deps from requirements_w_versions.txt, plus ones the code imports
# but the file omits (scipy, scikit-learn, trimesh).
RUN pip install \
        numpy==1.26.4 \
        tqdm==4.66.4 pytorch_lightning==2.2.4 hydra-core==1.3.2 \
        jaxtyping==0.2.28 beartype==0.18.5 wandb==0.17.0 einops==0.8.0 \
        colorama==0.4.6 scikit-image==0.23.2 colorspacious==1.1.2 \
        matplotlib==3.8.4 moviepy==1.0.3 imageio==2.34.1 timm==0.9.16 \
        dacite==1.8.1 lpips==0.1.4 e3nn==0.5.1 plyfile==1.0.3 tabulate==0.9.0 \
        svg.py==1.4.3 opencv-python==4.6.0.66 sk-video==1.1.10 open3d==0.18.0 \
        scipy scikit-learn trimesh black==24.4.2 ruff==0.4.4

# CUDA rasterizers (built against the torch above, for the archs listed).
RUN pip install --no-build-isolation \
        git+https://github.com/dcharatan/diff-gaussian-rasterization-modified \
        git+https://github.com/hbb1/diff-surfel-rasterization

# Sanity check that the extension imports.
RUN python -c "import torch, diff_surfel_rasterization, open3d, cv2; print(torch.__version__, torch.version.cuda)"

# torch >= 2.6 defaults torch.load to weights_only=True, which rejects the
# released SurfelSplat checkpoint. Restore the old behaviour.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    WANDB_MODE=offline \
    # Keep downloaded backbones (DINOv2, ~1.1 GB) in the data volume.
    TORCH_HOME=/workspace/data/torch_cache

# The repo is bind-mounted here at runtime so code edits don't need a rebuild.
WORKDIR /workspace/Surfel_Splat
CMD ["bash"]
