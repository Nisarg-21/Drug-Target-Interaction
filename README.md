# CMA-DTI: A Cross-Modal Fusion and Attentive Interaction Network for Explainable Drug-Target Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.7+](https://img.shields.io/badge/pytorch-1.7+-ee4c2c.svg)](https://pytorch.org/)
[![DGL 0.7+](https://img.shields.io/badge/dgl-0.7+-orange.svg)](https://www.dgl.ai/)
[![RDKit](https://img.shields.io/badge/rdkit-2021.03+-brightgreen.svg)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for **CMA-DTI**, a novel deep learning framework for Drug-Target Interaction (DTI) prediction with a focus on multi-modal fusion and interpretability.

## 1. Introduction

Accurate prediction of Drug-Target Interactions (DTI) is a cornerstone of modern drug discovery. While deep learning has shown great promise, effectively integrating multi-modal drug representations and providing clear, interpretable insights into the interaction mechanism remains a significant challenge.

**CMA-DTI** addresses these challenges by:
- **Leveraging Multi-Modal Drug Representations:** It effectively combines drug features from both graph structures (via **GCN**) and chemical sequences (via **ChemBERTa**).
- **Fine-Grained Internal Fusion:** A unique **intra-drug cross-attention** module is introduced to explicitly model the interactions between drug substructures and chemical subsequences, generating a comprehensive drug representation.
- **Powerful Protein Representation:** Utilizes the state-of-the-art pre-trained protein language model **ESM-2** to capture rich, contextual information from protein sequences.
- **Explainable Interaction Modeling:** A **drug-protein multi-head attention** module models the pairwise interactions between drug nodes and protein residues, providing a deep dive into the binding mechanism.
- **Dual-Level Interpretability:** The framework offers two layers of visualization—one for intra-drug feature associations and another for drug-protein interaction hotspots—enhancing the transparency and trustworthiness of predictions.

## 2. Framework

The overall architecture of CMA-DTI is illustrated below. It consists of four main parts: Feature Encoders, a Drug Internal Cross-Modal Fusion Module, a Drug-Protein Interaction Module, and a Prediction Module.
[Figure1.tiff](https://github.com/user-attachments/files/23011675/Figure1.tiff)

<img width="6172" height="4252" alt="whiteboard_exported_image-1" src="https://github.com/user-attachments/files/23011675/Figure1.tiff" />

The workflow is as follows:
1.  **Feature Encoding:** GCN, ChemBERTa, and ESM-2 are used to encode the drug's graph structure, SMILES sequence, and the protein's amino acid sequence, respectively.
2.  **Drug Internal Fusion:** The GCN-derived node features and ChemBERTa-derived token features are fed into a cross-attention module to generate a fused drug node representation.
3.  **Drug-Protein Interaction:** The fused drug node sequence and the protein residue sequence from ESM-2 are processed by a multi-head attention network to model their pairwise interactions.
4.  **Prediction:** The output from the interaction module is pooled and passed through an MLP to predict the final DTI probability score.

## 3. System Requirements

- **Operating System:** Linux (tested on Ubuntu 20.04)
- **GPU:** NVIDIA GPU with CUDA 11.3+ support is highly recommended for training.
- **Python:** 3.8+
- **Key Libraries:**
    - PyTorch (>=1.7.1)
    - DGL (>=0.7.1) with CUDA support
    - DGL-LifeSci
    - RDKit (>=2021.03)
    - Hugging Face Transformers
    - pandas, numpy, scikit-learn, yacs

## 4. Installation Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/qinchi1/CMA-DTI.git
    cd CMA-DTI
    ```

2.  **Create and activate the conda environment:**
    We recommend using `conda` for environment management.
    ```bash
    # Create a new conda environment
    conda create -n cmadti python=3.8 -y
    conda activate cmadti

    # Install PyTorch with CUDA support (adjust cudatoolkit version for your system)
    conda install pytorch torchvision toraudio cudatoolkit=11.3 -c pytorch

    # Install DGL with CUDA support
    conda install dgl-cuda11.3 -c dglteam

    # Install other essential libraries
    conda install -c conda-forge rdkit
    pip install dgllife transformers pandas scikit-learn yacs
    ```
    *Note: For a detailed list of package versions, please refer to the `environment.yml` file (if provided).*

3.  **Download Pre-trained Models:**
    Download the pre-trained ESM-2 and ChemBERTa models and place them in your desired directories. You will need to update the model paths in your configuration files.
    - **ESM-2 (esm2_t33_650M_UR50D):** [https://huggingface.co/facebook/esm2_t36_3B_UR50D]
    - **ChemBERTa:** [https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1/tree/main]

4.  **Prepare Datasets:**
    Download the BindingDB and BioSNAP datasets. Place them in the `./datasets` directory. Run the provided preprocessing scripts to generate the data splits for 10-fold cross-validation.
    ```bash
    # Example command (replace with your actual script)
    python scripts/preprocess_data.py --dataset BindingDB
    python scripts/preprocess_data.py --dataset BioSNAP
    ```

## 5. Run CMA-DTI

### Training the Model

To train the CMA-DTI model on a specific dataset and fold, use the `main.py` script.

**Example Command:**
```bash
python main.py --cfg configs/CMA.yaml --data BindingDB --split random
