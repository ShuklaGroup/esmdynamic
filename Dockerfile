FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

# LABEL org.opencontainers.image.version="1.0.0"
# LABEL org.opencontainers.image.authors="Shukla Group (UIUC)"
# LABEL org.opencontainers.image.source=""
# LABEL org.opencontainers.image.licenses=""
# LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04"

RUN apt-key del 7fa2af80 || true
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y \
    wget \
    libxml2 \
    git \
    cuda-minimal-build-11-3 \
    libcusparse-dev-11-3 \
    libcublas-dev-11-3 \
    libcusolver-dev-11-3 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH

# Create new conda environment with python 3.7
RUN conda create -n esmdynamic python=3.7 -y && conda clean --all

# Activate environment and install packages inside it
SHELL ["conda", "run", "-n", "esmdynamic", "/bin/bash", "-c"]

# Ensure the torch cache directory exists
RUN mkdir -p /root/.cache/torch/hub/checkpoints/

# Download required pretrained models into the torch cache
RUN wget -q -O /root/.cache/torch/hub/checkpoints/esmfold_3B_v1.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt && \
    wget -q -O /root/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D-contact-regression.pt \
    https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt

# Download esmdynamic weights
RUN wget -q -O /root/.cache/torch/hub/checkpoints/esmdynamic.pt \
   https://databank.illinois.edu/datafiles/jx4ui/download

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install "fair-esm[esmfold]"
RUN pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
RUN pip install 'fair-esm @ git+https://github.com/diegoeduardok/esmdynamic.git'
RUN pip install pandas
RUN pip install biopython
RUN pip install matplotlib
RUN pip install plotly[express]

# Download stereo_chemical_props.txt
RUN mkdir -p /opt/openfold/resources && \
    wget -q -P /opt/openfold/resources \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Make predict script executable
RUN echo '#!/bin/bash' > /usr/local/bin/predict && \
    echo 'python /opt/conda/envs/esmdynamic/lib/python3.7/site-packages/esm/esmdynamic/predict.py "$@"' >> /usr/local/bin/predict && \
    chmod +x /usr/local/bin/predict


WORKDIR /workspace

# Default shell back to bash (optional)
SHELL ["/bin/bash", "-c"]

# Entry point: activate conda env, then open bash shell
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate esmdynamic && exec bash"]
