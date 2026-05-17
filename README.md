# UniPert

![Main Image](https://github.com/user-attachments/assets/0949007e-af84-4646-9141-232fef965f8d)

## 🔔 Update
- [2026-05] Improved UniPert's compatibility with modern Python ecosystems. Added external MMseqs2 CLI integration to improve stability and portability of sequence similarity computation.
- [2025-01] Released the first version of pre-trained UniPert model (see [Li et al. 2025](https://www.biorxiv.org/content/10.1101/2025.02.02.635055)). Added tutorials for encoding genetic perturbagens, chemical perturbagens, and perturbation AnnData objects.

## 🛠️ Installation 

#### Step 1: Create a Conda Environment  
Create a new conda environment with Python 3.10 or later:  
```bash
conda create -n unipert python=3.10 -y
conda activate unipert
```

#### Step 2: Install PyTorch
Install **PyTorch** based on your system and CUDA version. Please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).  
Example:  
```bash
# CUDA-support (e.g., CUDA 13.0):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install PyTorch Geometric (PyG)  
Install **PyG** following the official [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).  
For most users:  
```bash
pip install torch-geometric
```

#### Step 4: Install MMseqs2  
UniPert relies on the external **MMseqs2** CLI for sequence alignment. Please refer to the official [MMseqs2 user guide](https://github.com/soedinglab/mmseqs2). 
Install MMseqs2 using conda:
```bash
conda install -c conda-forge -c bioconda mmseqs2
```
Verify installation:
```bash
mmseqs --help
```

#### Step 5: Install UniPert
#### Option 1: Install directly from GitHub
```bash
pip install git+https://github.com/lynn-1998/UniPert.git
```
#### Option 2: Install from source
```bash
git clone https://github.com/lynn-1998/UniPert.git
cd UniPert
pip install -e .
```

##  📖 Demo

| Name | Description |
|-----------------|-------------|
| [🧬 Encode Genetic Perturbagens](demo/tutorial_encode_genetic_perturbagens.ipynb) | Tutorial on how to encode genetic perturbagens from FASTA files (such as those downloaded from UniProt) or gene name lists using UniPert. |
| [💊 Encode Chemical Perturbagens](demo/tutorial_encode_chemical_perturbagens.ipynb) | Tutorial on how to encode chemical perturbagens from compound-SMILES files (e.g., .csv and .txt downloaded from PubChem and ChEMBL) or compound name list using UniPert. |
| [🌐 Encode Perturbagens For Perturbation AnnData](demo/tutorial_generate_UniPert_representation_for_pert_adata.ipynb) | Tutorial on how to generate UniPert embeddings for perturbation AnnData files (.h5ad) with genetic or chemical perturbagen metadata. |

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

## 📧 Contact

If you have any suggestions/ideas for UniPert or issues while using UniPert, please feel free to reach out to us. You can submit an issue on GitHub or contact us directly via email at:
	
- Yiming Li: liyiming5@qq.com or lym1998@csu.edu.cn
- Fan Yang: fionafyang@tencent.com

## 🚨 License 

This source code is licensed under the GPL-3.0 license found in the `LICENSE` file
in the root directory of this source tree.
