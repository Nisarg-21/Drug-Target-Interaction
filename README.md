# CMA-DTI: A Cross-Modal Attention Framework for Explainable Drug-Target Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![DGL 0.7+](https://img.shields.io/badge/dgl-0.7+-orange.svg)](https://www.dgl.ai/)
[![RDKit](https://img.shields.io/badge/rdkit-2021.03+-brightgreen.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for **CMA-DTI**, a novel deep learning framework for Drug-Target Interaction (DTI) prediction with a focus on multi-modal fusion and interpretability.

## 🌟 Introduction

Accurate prediction of Drug-Target Interactions (DTI) is a cornerstone of modern drug discovery. While deep learning has shown great promise, effectively integrating multi-modal drug representations and providing clear, interpretable insights into the interaction mechanism remains a significant challenge.

**CMA-DTI** addresses these challenges by:
1.  **Leveraging Multi-Modal Drug Representations:** It effectively combines drug features from both graph structures (via **GCN**) and chemical sequences (via **ChemBERTa**).
2.  **Fine-Grained Internal Fusion:** A unique **intra-drug cross-attention** module is introduced to explicitly model the interactions between drug substructures and chemical subsequences, generating a comprehensive drug representation.
3.  **Powerful Protein Representation:** Utilizes the state-of-the-art pre-trained protein language model **ESM-2** to capture rich, contextual information from protein sequences.
4.  **Explainable Interaction Modeling:** A **drug-protein multi-head attention** module models the pairwise interactions between drug nodes and protein residues, providing a deep dive into the binding mechanism.
5.  **Dual-Level Interpretability:** The framework offers two layers of visualization—one for intra-drug feature associations and another for drug-protein interaction hotspots—enhancing the transparency and trustworthiness of predictions.

## 🔧 Model Architecture

The overall architecture of CMA-DTI is illustrated below. It consists of three main parts: Feature Encoders, a Drug Internal Cross-Modal Fusion Module, and a Drug-Protein Interaction Module.

![CMA-DTI Framework]([path/to/your/framework_diagram.png])
*Figure 1: The overall architecture of the CMA-DTI framework.*

## 🚀 Getting Started

### Prerequisites

This project is built with Python 3.8. We recommend using `conda` to manage the environment.

- PyTorch (>=1.7.1)
- DGL (>=0.7.1)
- DGL-LifeSci
- RDKit
- Hugging Face Transformers
- pandas, numpy, scikit-learn

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[your_username]/CMA-DTI.git
    cd CMA-DTI
    ```

2.  **Create and activate the conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate cmadti
    ```
    *(Note: You will need to create an `environment.yml` file listing all dependencies. Alternatively, provide manual installation steps.)*

    **Manual Installation Example:**
    ```bash
    conda create -n cmadti python=3.8
    conda activate cmadti
    # Install PyTorch with CUDA support (adjust for your CUDA version)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    # Install DGL
    conda install dgl-cuda11.3 -c dglteam
    # Install other dependencies
    conda install -c conda-forge rdkit
    pip install dgllife transformers pandas scikit-learn
    ```

3.  **Download Pre-trained Models:**
    Download the pre-trained ESM-2 and ChemBERTa models and place them in your desired directories. You'll need to update the paths in the configuration files.
    - **ESM-2 (esm2_t33_650M_UR50D):** [Link to Hugging Face model page]
    - **ChemBERTa:** [Link to Hugging Face model page]

### Data Preparation

1.  Download the BindingDB and BioSNAP datasets.
2.  Preprocess the data according to the instructions in `[path/to/your/data_preprocessing_script.py]`. This should include generating positive/negative samples and splitting the data for 10-fold cross-validation.
3.  Place the processed data in the `./datasets` directory.

## ⚙️ Usage

### Training

To train the CMA-DTI model, use the `main.py` script. You need to specify a configuration file.

```bash
python main.py --cfg configs/bindingdb_config.yaml --data BindingDB --split [your_fold_prefix]
