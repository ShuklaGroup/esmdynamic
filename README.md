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

Otherwise, building a Docker image with the `Dockerfile` is the simplest option to get started. Within the container, [`run_esmdynamic`](https://github.com/diegoeduardok/esmdynamic/blob/main/esm/esmdynamic/predict.py) can be used to predict sequences in batches from a [FASTA](https://github.com/diegoeduardok/esmdynamic/blob/main/examples/esmdynamic/example.fasta) or [CSV](https://github.com/diegoeduardok/esmdynamic/blob/main/examples/esmdynamic/example.csv) file using flags `--fasta` or `--csv`. 

### Installation <a name="install"></a>

We recommend using the Dockerfile method to create an image with all required packages. Due to package deprecations, it may be difficult to install all requirements in a Python (e.g., Conda) environment. Additionally, the Docker setup process conviniently downloads the model weights. The only downside is that the Docker image takes relatively more space (~20 GB).

#### Docker <a name="install-docker"></a>

First, make sure you have installed [Docker](https://docs.docker.com/engine/install/). 

Since a GPU is recommended to run the model, you should have installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) as well.

Next, follow the commands:

```bash
git clone https://github.com/diegoeduardok/esmdynamic.git # Clone repo
cd esmdynamic
docker build -t esmdynamic .
docker run --rm -it --gpus all -v "$PWD":/workspace esmdynamic # Run container in current dir w/GPU access
./run_esmdynamic -h # Print help for prediction script 
```

#### Conda <a name="install-conda"></a>

Create a new environment with Python 3.7, cudatoolkit 11.3 (incompatible with NVIDIA Ada Lovelace `sm_89` or newer architectures), and the appropriate Pytorch version. Then, install required packages. Make sure `nvcc` is available (required by OpenFold).

```bash
conda create -n esmdynamic python=3.7
conda activate esmdynamic
conda install -c conda-forge cudatoolkit=11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
pip install 'fair-esm @ git+https://github.com/diegoeduardok/esmdynamic.git'
pip install pandas
pip install biopython # Handle FASTA input
pip install matplotlib # Visualization
pip install plotly[express] # Visualization
```

You can run the [`predict.py`](https://github.com/diegoeduardok/esmdynamic/blob/main/esm/esmdynamic/predict.py) script in this repo for inference (more instructions below).

### Bulk Prediction <a name="bulkprediction"></a>

Docs for the [`predict.py`](https://github.com/diegoeduardok/esmdynamic/blob/main/esm/esmdynamic/predict.py) script:

```
usage: predict.py [-h] (--sequence SEQUENCE | --fasta FASTA | --csv CSV) [--batch_size BATCH_SIZE] [--chunk_size CHUNK_SIZE] [--device {cpu,cuda}] [--output_dir OUTPUT_DIR]
               [--chain_ids CHAIN_IDS]

Predict dynamic contacts using ESMDynamic.

options:
  -h, --help            show this help message and exit
  --sequence SEQUENCE   Single sequence string.
  --fasta FASTA         Path to FASTA file with sequences.
  --csv CSV             CSV file with sequences (first column ID, second column sequence).
  --batch_size BATCH_SIZE
                        Batch size (default 1).
  --chunk_size CHUNK_SIZE
                        Model chunk size (default 256).
  --device {cpu,cuda}   Device (default: cuda).
  --output_dir OUTPUT_DIR
                        Output directory.
  --chain_ids CHAIN_IDS
                        Chain IDs to use (e.g., 'ABCDEF'). Default: A-Z.
```

With FASTA file input, the headers will be used as IDs. With CSV input, the first row are headers, the first column contains IDs, and the second column contains the sequences.

If you installed the Docker image, the inference script is exposed via the executable `run_esmdynamic`. For example, to recreate the dynamic contact maps in our publication, use either of the files in [examples](https://github.com/diegoeduardok/esmdynamic/tree/main/examples/esmdynamic):

```bash
./run_esmdynamic --csv example.csv --output_dir example
```

The output directory will contain the numerical output for each sequence in a plain text file that can be easily read by `numpy.loadtxt`. A PNG image and a HTML-based visualization file are also provided.

Depending on your system's memory, you may change the default values for `batch_size` or `chunk_size` to tradeoff speed and VRAM.

# Avilable Models and Datasets <a name="available"></a>
