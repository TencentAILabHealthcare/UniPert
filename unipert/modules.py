import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm 
from lamin_utils import logger
import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential

from rdkit.Chem import AllChem
from getSequence import getseq

from .utils import *
from . import DATA_DIR


class BGRL(nn.Module):
    """
    BGRL architecture for Graph representation learning.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor
        # target network
        self.target_encoder = copy.deepcopy(encoder)
        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        """Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        """
        Performs a momentum update of the target network's weights.
        
        Args:
            mm (float): Momentum value between 0.0 and 1.0.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        """
        Forward pass through the online and target networks.
        
        Args:
            online_x (Tensor): Input features for the online network.
            target_x (Tensor): Input features for the target network.
        
        Returns:
            Tuple: Predictions from the online network and target network outputs.
        """
        # forward online network
        online_y = self.online_encoder(online_x)
        # prediction
        online_q = self.predictor(online_y)
        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y
    

class GNN_Encoder(nn.Module):
    """
    GNN encoder for graph node (target) representation learning.
    """
    def __init__(self, layer_sizes, gnn_type='GCN', batchnorm=True, batchnorm_mm=0.99, layernorm=False, weight_standardization=False):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2

        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        # Construct gnn layers
        if gnn_type == 'GCN': gnn = GCNConv
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((gnn(in_dim, out_dim), 'x, edge_index, edge_weight -> x'),)
            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))
            layers.append(nn.PReLU())

        self.model = Sequential('x, edge_index, edge_weight', layers)   ### add edge_weight

    def forward(self, data):
        """
        Forward pass through the GNN encoder.
        
        Args:
            data (Data): Input data containing node features and edge information.
        
        Returns:
            Tensor: Output features after passing through the GNN layers.
        """
        if self.weight_standardization:
            self.standardize_weights()
        return self.model(data.x, data.edge_index, data.edge_attr)    ### add edge_weight

    def reset_parameters(self):
        """Reset the parameters of the model."""
        self.model.reset_parameters()

    def standardize_weights(self):
        """Standardize the weights of the GNN layers."""
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight
                

class MLP_Predictor(nn.Module):
    """ 
    MLP predictor for generating predictions from input features.
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


### ================= Compound Encoder ================= ###


class CompoundEncoder(nn.Module):
    """
    Compound encoder for generating embeddings from SMILES representations.
    """
    def __init__(self, embedder, output_dim=256):
        super().__init__()
        self.embedder = embedder
        self.emb_dict = embedder.saved_embs
        emb_dim = self.embedder.emb_dim
        self.linear_layer = nn.Linear(emb_dim, output_dim)

    def trainable_parameters(self):
        return list(self.linear_layer.parameters())

    def forward(self, smiles_list, device):
        """
        Forward pass to generate embeddings for a list of SMILES.
        
        Args:
            smiles_list (Union[str, List[str]]): A SMILES string or a list of SMILES strings.
            device (torch.device): The device to which the tensors should be moved.
        
        Returns:
            Tensor: Transformed embeddings after passing through the linear layer.
        """       
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        embs = []
        for smiles in smiles_list:
            assert smiles in list(self.emb_dict.keys()), \
                f'{smiles} embedding not found! Please get the embedding with CompoundEmbedder first!'
            embs.append(self.emb_dict[smiles].reshape(-1))
        embs = torch.tensor(np.array(embs), dtype=torch.float32).to(device)
        embs = self.linear_layer(embs)    # [num_sm, output_dim]
        return embs 


class CompoundEmbedderECFP4:
    """
    Compound SMILES representation using ECFP4 (Extended Connectivity Fingerprints).
    """
    def __init__(self, data_dir=DATA_DIR, device=torch.device('cuda:0')):
        super().__init__()
        self.device = device
        self.saved_embs = {}
        self.emb_dim = 2048
        self.ref_emb_file = os.path.join(data_dir, 'ref_ecfp4_embs.pkl')
        self.init_embs_from_saved_pkl()
        logger.success('ECFP4 embedder loaded.')

    def init_embs_from_saved_pkl(self):
        if os.path.exists(self.ref_emb_file):
            reps = pd.read_pickle(self.ref_emb_file)
            self.saved_embs.update(reps)
        else:
            print(f'Reference ECFP4 embedding file [{self.ref_emb_file}] not found.')
            print(f'A new reference ECFP4 embedding file will be created.')
            # Create a new pickle file
            with open(self.ref_emb_file, 'wb') as f:
                pickle.dump(self.saved_embs, f)

    def update_saved_pkl(self, new_embs):
        """
        Update the saved embeddings in the pickle file.
        
        Args:
            new_embs (dict): New embeddings to be added.
        """
        self.saved_embs.update(new_embs)
        with open(self.ref_emb_file, 'wb') as f:
            pickle.dump(self.saved_embs, f)
            logger.success(f'Reference ECFP4 embedding updated!')

    def get_emb_from_smiles(self, smiles):
        """
        Get the embedding for a given SMILES string.
        
        Args:
            smiles (str): A SMILES representation of a compound.
        
        Returns:
            dict: A dictionary containing the SMILES and its corresponding embedding.
        """
        if check_smiles(smiles):
            c_smiles = Chem.CanonSmiles(smiles)
            if smiles in list(self.saved_embs.keys()):
                return {smiles: self.saved_embs[smiles]}
            elif c_smiles in list(self.saved_embs.keys()):
                self.saved_embs.update({smiles: self.saved_embs[c_smiles]})
                return {smiles: self.saved_embs[c_smiles]}
            else:
                mol = Chem.MolFromSmiles(smiles)
                ecfp4_emb = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                ecfp4_emb = np.array(ecfp4_emb, dtype=float)  
                self.saved_embs.update({c_smiles: ecfp4_emb, smiles: ecfp4_emb})
                return {smiles: ecfp4_emb}
        else:
            return None
            
    def get_emb_from_smiles_list(self, smiles_list, save=False):
        """
        Get embeddings for a list of SMILES strings.
        
        Args:
            smiles_list (List[str]): A list of SMILES strings.
            save (bool): Whether to save the updated embeddings.
        
        Returns:
            dict: A dictionary of SMILES and their corresponding embeddings.
        """
        out_embs = {}
        for smiles in tqdm(smiles_list):
            emb = self.get_emb_from_smiles(smiles)
            if emb is not None:
                out_embs.update(emb)
        if save: self.update_saved_pkl(out_embs)
        return out_embs
    
    def get_bulk_emb_from_smiles_list(self, smiles_list, save=False):
        """
        Get embeddings for a list of SMILES strings.
        
        Args:
            smiles_list (List[str]): A list of SMILES strings.
            save (bool): Whether to save the updated embeddings.
        
        Returns:
            dict: A dictionary of SMILES and their corresponding embeddings.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        out_embs = {}
        total_tasks = len(smiles_list)
        cpu_cores = os.cpu_count() or 1
        if total_tasks>0:
            logger.info(f'Using {cpu_cores} cores to generate ECFP4 embeddings.')
            with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
                with tqdm(total=total_tasks, desc="Processing") as pbar:
                    futures = {executor.submit(compound_worker, self, smiles): smiles for smiles in smiles_list}
                    for future in as_completed(futures):
                        smiles = futures[future]
                        try:
                            partial_out_embs, key = future.result()
                            if key is not None:
                                out_embs.update(partial_out_embs)
                            pbar.update(1)  
                        except Exception as exc:
                            logger.error(f'{smiles} generated an exception: {exc}')
                            pbar.update(1) 
            logger.success(f'ECFP4 embeddings with {len(out_embs)} queries generated.')
            if save:self.update_saved_pkl(out_embs)
        return out_embs


def compound_worker(cls_instance, smiles):
    """
    Worker function to get embeddings for a single SMILES string.
    
    Args:
        cls_instance (CompoundEmbedderECFP4): Instance of the CompoundEmbedderECFP4 class.
        smiles (str): A SMILES representation of a compound.
    
    Returns:
        tuple: A tuple containing the embedding dictionary and the SMILES string.
    """
    emb = cls_instance.get_emb_from_smiles(smiles)
    if emb is not None:
        return {smiles: emb}, smiles
    else:
        return {}, None 
    

### ================= Target Encoder ================= ###

class TargetEncoder(nn.Module):
    """
    Target encoder for generating embeddings from protein sequences.
    """
    def __init__(self, embedder, output_dim=256):
        super().__init__()
        self.embedder = embedder
        self.emb_dict = embedder.saved_embs
        emb_dim = embedder.emb_dim
        self.linear_layer = nn.Linear(emb_dim, output_dim)

    def trainable_parameters(self):
        """Returns the parameters of the linear layer that can be trained."""
        return list(self.linear_layer.parameters())

    def forward(self, tgt_list, device):
        """
        Forward pass to generate embeddings for a list of target identifiers.
        
        Args:
            tgt_list (Union[str, List[str]]): A target identifier string or a list of target identifiers.
            device (torch.device): The device to which the tensors should be moved.
        
        Returns:
            Tensor: Transformed embeddings after passing through the linear layer.
        """
        if isinstance(tgt_list, str):
            tgt_list = [tgt_list]
        embs = []
        for tgt in tgt_list:
            assert tgt in list(self.emb_dict.keys()), \
                f'{tgt} emb not found! Please get the emb with TargetEmbedder first!'
            embs.append(self.emb_dict[tgt].reshape(-1))
        embs = torch.tensor(np.array(embs), dtype=torch.float32).to(device)
        embs = self.linear_layer(embs)    
        return embs   


class TargetEmbedderESM2:
    """
    Protein sequence representation using ESM2 (Evolutionary Scale Modeling).
    """
    def __init__(self, data_dir=DATA_DIR, device=torch.device('cuda:0')):
        super().__init__()
        self.data_dir = data_dir
        self.device = device
        self.model, self.alphabet = None, None
        self.saved_embs = {}
        self.emb_dim = 1280
        self.ref_emb_file = os.path.join(data_dir, 'ref_esm2_embs.pkl')
        self.ref_fasta_file = os.path.join(data_dir, 'ref_target_seq.fasta')
        self.model = None
        self.load_model()
        self.init_embs_from_saved_pkl()
        logger.success('ESM2 embedder created.')

    def load_model(self):
        """Load the ESM2 model and its alphabet."""
        from esm import pretrained
        self.model, self.alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
        self.model = self.model.to(self.device)
        if self.device.type == 'cuda': self.model.half()
        logger.success('ESM2 model loaded.')

    def update_saved_pkl(self, new_embs):
        """
        Update the saved embeddings in the pickle file.
        
        Args:
            new_embs (dict): New embeddings to be added.
        """
        self.saved_embs.update(new_embs)
        with open(self.ref_emb_file, 'wb') as f:
            pickle.dump(self.saved_embs, f)
            logger.success(f'ref_esm2_embs.pkl updated.')
            return

    def init_embs_from_saved_pkl(self):
        """Initialize embeddings from a saved pickle file or create a new one from the reference FASTA."""
        if os.path.exists(self.ref_emb_file):
            ref_reps = pd.read_pickle(self.ref_emb_file)
            self.saved_embs.update(ref_reps)
            logger.download(f'Reference ESM2 embedding file loaded.')
        else:
            logger.warning(f'Reference ESM2 embedding file [{self.ref_emb_file}] not found.') 
            logger.info(f'A new reference ESM2 embedding file will be created.')
            ref_reps = self.get_embs_from_fasta(self.ref_fasta_file)
            self.update_saved_pkl(ref_reps)

    def update_genome_wide_embs(self, save=False):
        """
        Update genome-wide embeddings based on target information.
        
        Args:
            save (bool): Whether to save the updated embeddings.
        
        Returns:
            dict: A dictionary of gene symbols and their corresponding embeddings.
        """
        out_embs = {}
        tgt_ptbg_info_file = os.path.join(self.data_dir, 'target_info_HGNC_id_19321_with_labels.csv')
        df = pd.read_csv(tgt_ptbg_info_file).dropna(subset=['Approved symbol', 'UniProt accession']).loc[:, ['Approved symbol', 'UniProt accession']].drop_duplicates()
        mapping_dict = dict(zip(df['Approved symbol'], df['UniProt accession']))
        for gene_symbol, uniprot_id in mapping_dict.items():
                if uniprot_id in list(self.saved_embs.keys()):
                    out_embs.update({gene_symbol: self.saved_embs[uniprot_id]})
        if save: self.update_saved_pkl(out_embs)
        return out_embs

    def get_emb_from_seq(self, seq):
        """
        Get the embedding for a given protein sequence.
        
        Args:
            seq (str): A protein sequence.
        
        Returns:
            np.ndarray: The embedding representation of the sequence.
        """
        seq = seq[:5000]   # Max length of ESM input sequence set to 5000
        with torch.no_grad():   
            toks = torch.tensor([self.alphabet.get_idx(i) for i in seq]).reshape(1, -1)
            toks = toks.to(self.device, non_blocking=True)
            out = self.model(toks, repr_layers=[33], return_contacts=False)
            token_reps = out["representations"][33].mean(1).reshape(-1).detach().cpu().numpy()
        return token_reps
 
    def get_emb_from_gene_symbol(self, gene):
        """
        Get the embedding for a given gene symbol.
        
        Args:
            gene (str): A gene symbol.
        
        Returns:
            dict: A dictionary containing the gene symbol and its corresponding embedding.
        """
        if gene in list(self.saved_embs.keys()):
            return {gene: self.saved_embs[gene]}    
        elif gene.upper() in list(self.saved_embs.keys()):
            return {gene: self.saved_embs[gene.upper()]}     
        else:
            logger.print(f'Retrieving sequence for {gene} ...')
            try:
                pro_seq = getseq(gene, just_sequence=True)
            except:
                logger.warning(f'No sequence found for {gene}.')
                return None
            out_rep = self.get_emb_from_seq(pro_seq)
            # self.saved_embs.update({gene: out_rep})
            return {gene: out_rep}

    def get_emb_from_gene_symbol_list(self, gene_list, save=False):
        """
        Get embeddings for a list of gene symbols.
        
        Args:
            gene_list (List[str]): A list of gene symbols.
            save (bool): Whether to save the updated embeddings.
        
        Returns:
            dict: A dictionary of gene symbols and their corresponding embeddings.
        """
        gene_list = list(set(gene_list))
        out_embs = {}
        for gene in gene_list:
            emb = self.get_emb_from_gene_symbol(gene)
            if emb is not None:
                out_embs.update(emb)
        # logger.success(f'ESM2 embeddings with {len(out_embs)} querys generated.')
        if save: self.update_saved_pkl(out_embs)
        return out_embs
    
    def get_emb_from_uniprot_id(self, uniprot_id):
        """
        Get the embedding for a given UniProt Accession Number.
        
        Args:
            uniprot_id (str): A UniProt Accession Number.
        
        Returns:
            dict: A dictionary containing the UniProt AC and its corresponding embedding.
        """
        uniprot_id = uniprot_id.upper()
        if uniprot_id not in list(self.saved_embs.keys()):
            logger.print(f'Retrieving sequence for {uniprot_id} ...')
            try:
                pro_seq = getseq(uniprot_id, uniprot_id=True, just_sequence=True)
            except:
                logger.warning(f'No sequence found for {uniprot_id}.')
                return None
            out_rep = self.get_emb_from_seq(pro_seq)
            # self.saved_embs.update({uniprot_id: out_rep})
            return {uniprot_id: out_rep}
        else:
            return {uniprot_id: self.saved_embs[uniprot_id]}
    
    def get_emb_from_uniprot_id_list(self, uid_list, save=False):
        """
        Get embeddings for a list of UniProt Accession Numbers.
        
        Args:
            uid_list (List[str]): A list of UniProt Accession Numbers.
            save (bool): Whether to save the updated embeddings.
        
        Returns:
            dict: A dictionary of UniProt Accession Numbers and their corresponding embeddings.
        """
        uid_list = list(set(uid_list))
        out_embs = {}
        for uid in uid_list:
            emb = self.get_emb_from_uniprot_id(uid)
            if emb is not None:
                out_embs.update(emb)
        # logger.success(f'ESM2 embeddings with {len(out_embs)} querys generated.')
        if save: self.update_saved_pkl(out_embs)
        return out_embs

    def get_embs_from_fasta(self, fasta_file, save=False):
        """
        Get embeddings from a FASTA file.
        
        Args:
            fasta_file (str): Path to the FASTA file.
            save (bool): Whether to save the updated embeddings.
        
        Returns:
            dict: A dictionary of headers and their corresponding embeddings.
        """
        dataset = FastaBatchedDataset.from_file(fasta_file)
        out_embs = {}
        for header, seq in tqdm(dataset):  
            emb = self.get_emb_from_seq(seq)
            if 'PN=' in header:      # pert name as key
                pn = header.split('PN=')[1]
                out_embs.update({pn: emb})  
            if '|' in header:      
                uniprot_id = header.split('|')[1].split(' ')[0]  
                if uniprot_id in list(self.saved_embs.keys()): 
                    out_embs.update({uniprot_id: emb}) 
                else:
                    emb = self.get_emb_from_seq(seq) 
                    out_embs.update({uniprot_id: emb})
            if ('PN=' not in header) and ('|' in header):
                out_embs.update({header: emb}) 
        logger.success(f'ESM2 embeddings with {len(out_embs)} querys generated.')
        if save: self.update_saved_pkl(out_embs)
        return out_embs