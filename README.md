# UniPert

![Main Image](https://github.com/user-attachments/assets/0949007e-af84-4646-9141-232fef965f8d)


## 🛠️ Installation 

#### Step 1: Create a Conda Environment  
Create a new conda environment with Python 3.9:  
```bash
conda create -n unipert python=3.9 -y
conda activate unipert
```

#### Step 2: Install PyTorch and PyG  
As a prerequisite, you must have **PyTorch** and **PyTorch Geometric (PyG)** installed to use this repository.  

1. Install **PyTorch** based on your system and CUDA version. Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for detailed instructions.  
   We recommend using the development version `torch==1.12.1` for compatibility:  
   ```bash
   # CUDA-support (e.g., CUDA 11.3):
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

   # CPU-only
   pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
   ```

2. Install **PyG**. Refer to the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for system-specific instructions.  
   For most users, the following command should suffice:  
   ```bash
   pip install torch-geometric
   ```

#### Step 3: Clone the UniPert Repository  
Clone the UniPert repository to your local machine:
   ```bash
   git clone https://github.com/TencentAILabHealthcare/UniPert.git
   cd UniPert
   ```

#### Step 4: Install Dependencies
Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

##  📖 Demo

| Name | Description |
|-----------------|-------------|
| [🧬 Encode Genetic Perturbagens](demo/tutorial_encode_genetic_perturbagens.ipynb) | Tutorial on how to encode genetic perturbagens from FASTA files (such as those downloaded from UniProt) or gene name lists using UniPert. |
| [💊 Encode Chemical Perturbagens](demo/tutorial_encode_chemical_perturbagens.ipynb) | Tutorial on how to encode chemical perturbagens from compound-SMILES files (e.g., .csv, .txt, and .xlsx downloaded from PubChem and ChEMBL) or compound name list using UniPert. |
| [🔗 Encode Perturbagens For Perturbation AnnData](demo/tutorial_generate_UniPert_representation_for_pert_adata.ipynb) | Tutorial on how to generate UniPert embeddings for perturbation AnnData files (.h5ad) with genetic or chemical perturbagen metadata. |
<!--
## 🤝 Citation

If you find the models useful in your research, please cite:

```bibtex
@article {Li2025.02.02.635055,
  author = {Li, Yiming and Zeng, Min and Zhu, Jun and Liu, Linjing and Wang, Fang and Huang, Longkai and Yang, Fan and Li, Min and Yao, Jianhua},
  title = {Genetic-to-Chemical Perturbation Transfer Learning Through Unified Multimodal Molecular Representations},
  elocation-id = {2025.02.02.635055},
  year = {2025},
  doi = {10.1101/2025.02.02.635055},
  publisher = {Cold Spring Harbor Laboratory}
}
```
-->
## 📧 Contact

If you have any suggestions/ideas for UniPert or issues while using UniPert, please feel free to reach out to us. You can submit an issue on GitHub or contact us directly via email at:
	
- Yiming Li: liyiming5@qq.com or lym1998@csu.edu.cn
- Fan Yang: fionafyang@tencent.com

## 🚨 License 

This source code is licensed under the GPL-3.0 license found in the `LICENSE` file
in the root directory of this source tree.
