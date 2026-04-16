FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Diego Kleiman - Shukla Group (UIUC)"
LABEL org.opencontainers.image.source="https://github.com/ShuklaGroup/esmdynamic"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04"

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# System packages
RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    git \
    libxml2 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean --all -f -y

# Create environment with the exact Python version you are using
RUN conda create -n esmdynamic python=3.11.13 -y && conda clean --all -f -y

# Run subsequent commands inside the env
SHELL ["conda", "run", "-n", "esmdynamic", "/bin/bash", "-c"]

# Match your working CUDA toolkit / nvcc installs
RUN conda install -c nvidia \
    cuda-nvcc=12.9.86 \
    cuda-toolkit=12.9.1 \
    -y && \
    conda clean --all -f -y

# Match your working PyTorch install
RUN python -m pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# Match your working Python package installs
RUN python -m pip install --no-cache-dir \
    mdtraj \
    scipy \
    omegaconf \
    pytorch_lightning \
    biopython \
    ml_collections \
    einops \
    py3Dmol \
    modelcif \
    matplotlib \
    'plotly[express]' \
    dm-tree \
    tensorboard

RUN python -m pip install --no-cache-dir \
    git+https://github.com/NVIDIA/dllogger.git

RUN python -m pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/sokrypton/openfold.git'

RUN python -m pip install --no-cache-dir \
    git+https://github.com/ShuklaGroup/esmdynamic.git

# OpenFold resource file
RUN mkdir -p /opt/openfold/resources && \
    wget -q -O /opt/openfold/resources/stereo_chemical_props.txt \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Torch cache + pretrained weights
RUN mkdir -p /root/.cache/torch/hub/checkpoints/

RUN wget -q -O /root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt \
    https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esmdynamic.pt \
    https://databank.illinois.edu/datafiles/7odsk/download

WORKDIR /workspace

# Back to normal shell for interactive use
SHELL ["/bin/bash", "-c"]

CMD ["bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate esmdynamic && exec bash"]