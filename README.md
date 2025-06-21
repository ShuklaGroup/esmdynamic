# ESMDynamic

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diegoeduardok/esmdynamic/blob/main/examples/esmdynamic/esmdynamic.ipynb)
[![Download Data](https://img.shields.io/badge/I-Data_Bank-black?labelColor=FF5F05)](https://doi.org/10.13012/B2IDB-3773897_V1)


This is the code repository for publication [ref](DOI) TODO: ADD PUBLICATION REF. It contains a model to predict dynamic contact maps from single protein sequences.

This repository is based on [Evolutionary Scale Modeling](https://github.com/facebookresearch/esm), which has been archived.

<details close><summary><b>Table of contents</b></summary>

- [Usage](#usage)
    - [Quick Start](#quickstart)
    - [Installation](#install)
    	- [Docker](#install-docker)
    	- [Conda](#install-conda)
  - [Bulk Prediction](#bulkprediction)
  - [Visualization](#visualization)
- [Available Models and Datasets](#available)
  - [Pre-trained Model](#available-model)
  - [Datasets](#available-datatsets)
	  - [RCSB Clustering](#available-datatsets-rcsb)
	  - [mdCATH](#available-datatsets-mdcath)
- [Citations](#citations)
- [License](#license)
</details> 

## Usage <a name="usage"></a>

### Quick Start <a name="quickstart"></a>

If you wish to use the model to predict a small number of sequences, we recommend you simply use our [Google Colab Notebook](https://colab.research.google.com/github/diegoeduardok/esmdynamic/blob/main/examples/esmdynamic/esmdynamic.ipynb) with manual sequence entry.

Otherwise, building a Docker image with the `Dockerfile` is the simplest option to get started. Within the container, `run_esmdynamic` can be used to predict sequences in batches from a [FASTA](https://github.com/diegoeduardok/esmdynamic/blob/main/examples/esmdynamic/example.fasta) or [CSV](https://github.com/diegoeduardok/esmdynamic/blob/main/examples/esmdynamic/example.csv) file using flags `--fasta` or `--csv`. 

### Installation <a name="install"></a>

We recommend using the Dockerfile method to create a container with all required packages. Due to package deprecations, it may be difficult to install all requirements in a Conda environment. Additionally, the Docker setup process conviniently downloads the model weights. The only downside is that the Docker image takes relatively more space (~20 GB).

#### Docker <a name="install"></a>

First, make sure you have installed [Docker](https://docs.docker.com/engine/install/) in your system. 

Since a GPU is recommended to run the model, you should have installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) as well.

Next, follow the commands:

```bash
git clone https://github.com/diegoeduardok/esmdynamic.git # Clone repo
cd esmdynamic
docker build -t esmdynamic .
docker run --rm -it --gpus all -v "$PWD":/workspace esmdynamic # Run container in current dir w/GPU access
./run_esmdynamic -h # Print help for prediction script 
```

