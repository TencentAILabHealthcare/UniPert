from typing import List, Tuple, Optional
import pickle
import anndata
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from lamin_utils import logger, colors

import mmseqs
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import torch
import torch.nn as nn
from torch_geometric.data import Data 
from torch_geometric.utils import add_self_loops

from .modules import *
from .utils import *
from . import DATA_DIR, MODEL_DIR, MMSEQS_CACHE_DIR


class UniPert:
    def __init__(self, data_dir: str = DATA_DIR, model_dir: str = MODEL_DIR):
        """
        Initialize the UniPert model with specified data and model directories.

        Args:
            data_dir (str, optional): The directory where data files are located. Defaults to DATA_DIR.
            model_dir (str, optional): The directory where the model files are stored. Defaults to MODEL_DIR.
        """
        # Data
        self.data_dir = data_dir

        # Device
        if torch.cuda.is_available() and torch.version.cuda:
            self.device = torch.device('cuda:0')
            logger.info("CUDA is available. Using CUDA.")
        else:
            self.device = torch.device('cpu')
            logger.info("CUDA is not available or not properly installed. Using CPU instead.")

        # MMseqs tool
        self.client = None

        # Compound search service
        self.cp_server = None

        # Reference target graph information
        self.ref_target_seq_file = os.path.join(data_dir, 'ref_target_seq.fasta')
        self.ref_target_sim_file = os.path.join(data_dir, 'ref_target_seq_similarity.csv')
        self.ref_target = None
        self.ref_target_sim = None
        self.ref_target_graph = None

        # Custom target graph information
        self.custom_target_seq_file = os.path.join(data_dir, 'custom_target_seq.fasta')
        self.custom_target_sim_file = os.path.join(data_dir, 'custom_target_seq_similarity.csv')
        self.custom_target = None
        self.custom_target_sim = None
        self.custom_target_graph = None
        self.custom_target_ui2pn = defaultdict(list)     # {UniProt_id: [perturbagen_names]}

        # Reference-custom target graph node ID mapping 
        self.node_name2id = None

        # Custom compound information
        self.custom_compound_smiles_file = os.path.join(data_dir, 'custom_compound_smiles.txt')

        # Check if model is already trained at the given path, load model
        self.loaded = False
        try:
            self.load_models(save_dir=model_dir)
            self.loaded = True
        except FileNotFoundError:
            logger.error(f"Model have not found at [{model_dir}]. Please download the trained model file [unipert_model.pt].")

        # Prepare reference graph  
        self.prepare_ref_target_graph() 
        
        # Saved UniPert representations
        self.unipert_reps = {}    

        # Loaded
        if self.loaded: logger.success(colors.green(f"Model loaded and initialized."))


    def set_model_hparams(self, params: Optional[dict] = None):
        """
        Set the hyperparameters for the model.

        Args:
            params (dict, optional): A dictionary of hyperparameters to update the model's settings.
        """
        # Model hyperparameters
        self.save_dir = MODEL_DIR
        self.model_hparams = {
            'out_dim': 256,
            'gnn_type': 'GCN',
            'layer_num': 2,
            'target_embedder': 'ESM2',
            'compound_embedder': 'ECFP4',
        }
        if params:
            params.pop('device', None)
            self.model_hparams.update(params)
        for key, value in self.model_hparams.items():
            setattr(self, key, value)
        self.model_path = os.path.join(self.save_dir, 'unipert_model.pt')
        

    def construct_model(self) -> nn.ModuleList:
        """
        Construct the UniPert model architecture.

        Returns:
            nn.ModuleList: A list of constructed UniPert models including compound and target models.
        """
        logger.info(colors.yellow(f"Constructing UniPert model..."))

        # Prepare embedders
        self.tgt_embedder = TargetEmbedderESM2(data_dir=self.data_dir, device=self.device)
        self.cp_embedder = CompoundEmbedderECFP4(data_dir=self.data_dir, device=self.device)
        self.tgt_embs = None
        self.cp_embs = None

        # Prepare UniPert model
        self.cp_model = CompoundEncoder(
            embedder=self.cp_embedder, 
            output_dim=self.out_dim
            )
        self.tgt_encoder = GNN_Encoder(
            layer_sizes=[self.tgt_embedder.emb_dim]+[self.out_dim]*self.layer_num, 
            gnn_type=self.gnn_type, 
            batchnorm=True
            )
        self.tgt_predictor = MLP_Predictor(self.out_dim, self.out_dim, hidden_size=self.out_dim*4)
        self.tgt_model = BGRL(self.tgt_encoder, self.tgt_predictor)
        
        unipert_models = nn.ModuleList([])
        unipert_models.append(self.cp_model)
        unipert_models.append(self.tgt_model)

        logger.success(f"UniPert model constructed.")
        return unipert_models


    def load_models(
            self, 
            save_dir: Optional[str] = None, 
    ):
        """
        Load the pre-trained models from the specified directory.

        Args:
            save_dir (str, optional): The directory from which to load the model.
        """
        # Define model saved path
        if not save_dir:
            model_path = os.path.join(self.save_dir, 'unipert_model.pt')
        else:
            self.save_dir = save_dir
            model_path = os.path.join(save_dir, 'unipert_model.pt')

        # Construct and load model
        # Model weights
        model_weights = torch.load(model_path, map_location=self.device)
        model_weights['hparams']['device'] = self.device
        model_weights['hparams']['save_dir'] = self.save_dir
        # Model hyperparameters
        self.set_model_hparams(model_weights.pop('hparams'))
        # Model structure
        unipert_models = self.construct_model()
        unipert_models = unipert_models.to(self.device)
        logger.download(f"Pretrained model file loaded.")
        self.cp_model.load_state_dict(model_weights['cp_encoder'])
        self.tgt_encoder.load_state_dict(model_weights['tgt_encoder'])
            

    def prepare_mmseqs(self):
        """
        Prepare the MMseqs client and create a reference database.
        """
        logger.info('Preparing MMseqs and creating reference database...')
        import shutil
        if os.path.exists(MMSEQS_CACHE_DIR):
            shutil.rmtree(MMSEQS_CACHE_DIR)
        os.makedirs(MMSEQS_CACHE_DIR)  
        self.client = mmseqs.MMSeqs(storage_directory=MMSEQS_CACHE_DIR)
        self.client.databases.create('ref', 'ref genome-wide protein database', self.ref_target_seq_file)
        logger.success('MMseqs reference database created.')


    def prepare_cp_server(self):
        """
        Prepare the compound search service by connecting to either PubChem or ChemSpider.
        """
        if check_chemspipy():
            from chemspipy import ChemSpider
            self.cp_server = ChemSpider(os.environ['CHEMSPIDER_APIKEY'])
            self.cp_server_name = 'chemspider'
            logger.success(f"chemspider server connected successfully.")
        elif check_pubchempy():
            import pubchempy as pcp
            self.cp_server = pcp
            self.cp_server_name = 'pubchem'  
            logger.success(f"pubchempy server connected successfully.")
        else:
            logger.warning(f"pubchempy or chemspipy service cannot be use! Please check your network connection.")
            self.cp_server = None
            self.cp_server_name = None


    def cal_similarity_from_fasta(
            self, 
            custom_seq_fasta: Optional[str] = None, 
            save_sim_file: bool = True
    ):
        """
        Calculate the similarity between custom sequences and a reference FASTA file.

        Args:
            custom_seq_fasta (str, optional): The path to the custom FASTA file. If provided, it will be used for similarity calculation.
            save_sim_file (bool, optional): Whether to save the similarity results to a CSV file. Defaults to True.
        """
        # Prepare the mmseqs client
        if self.client is None: 
            self.prepare_mmseqs()

        # Define the custom target sequence file
        if custom_seq_fasta is not None:
            self.custom_target_seq_file = custom_seq_fasta
            # self.custom_target_sim_file = self.custom_target_seq_file.replace('.fasta', '_similarity.csv')

        # Extract unique IDs of custom target sequences from the fasta file
        self.custom_target = list(set([seq_record.id.split('|')[1] for seq_record in SeqIO.parse(self.custom_target_seq_file, "fasta")]))

        # Use the mmseqs client to calculate similarity
        logger.info(f'Calculating similarity between {self.custom_target_seq_file} and reference fasta file...')
        results = self.client.databases[0].search_file(
                self.custom_target_seq_file, 
                search_type="protein",
                headers=["query_sequence_id", "target_sequence_id", "sequence_identity"]
                )
        self.custom_target_sim = results.dataframe

        # If required, save the similarity results to a CSV file
        if save_sim_file:
            self.custom_target_sim.to_csv(self.custom_target_sim_file, index=False, header=False)
            logger.save(f"MMseqs2 results saved to {self.custom_target_sim_file}")


    def prepare_ref_target_graph(self):
        """
        Prepare the reference target graph by loading or constructing it from the specified sequence file.

        This method checks if the reference target graph is already prepared. If not, it will load the necessary
        sequence and similarity files, generate node embeddings, and construct the graph data structure.

        Raises:
            ValueError: If the reference target sequence file does not exist.
        """
        # Check if the reference target graph is already prepared
        if self.ref_target_graph is not None:
            logger.download(f"Reference target graph prepared.")
            return
        
        # Check if the reference target sequence file exists
        if not os.path.exists(self.ref_target_seq_file):
            logger.error(f"{self.ref_target_seq_file} does not found. Please download it first.")
            raise ValueError(f'{self.ref_target_seq_file} not found.')

        # Prepare graph edges if similarity data is not available
        if self.ref_target_sim is None:
            if not os.path.exists(self.ref_target_sim_file):
                self.cal_similarity_from_fasta(self.ref_target_seq_file)

        # Load the similarity data
        self.ref_target_sim = pd.read_csv(self.ref_target_sim_file, header=None).drop_duplicates()
        edge_info_df = self.ref_target_sim.copy()

        # Get graph node names (UniProt IDs) from the reference sequence file
        self.ref_target = list(set([seq_record.id.split('|')[1] for seq_record in SeqIO.parse(self.ref_target_seq_file, "fasta")]))

        # Generate graph node embeddings if not already available
        if self.tgt_embedder.saved_embs == {}:
            self.tgt_embedder.get_embs_from_fasta(self.ref_target_seq_file)
        self.tgt_embs = self.tgt_embedder.saved_embs
        node_feature_df = pd.DataFrame(self.tgt_embs).T

        # Generate a mapping dictionary from node names to IDs
        node_names = self.ref_target
        self.node_name2id = {name: i for i, name in enumerate(node_names)}
        edge_info_df.iloc[:, 0] = edge_info_df.iloc[:, 0].map(self.node_name2id)
        edge_info_df.iloc[:, 1] = edge_info_df.iloc[:, 1].map(self.node_name2id)

        # Generate graph data
        x = torch.tensor(node_feature_df.loc[node_names, :].values, dtype=torch.float)
        edge_index = torch.tensor(edge_info_df.iloc[:, [1, 0]].astype(int).values.T, dtype=torch.long)
        edge_weight = torch.tensor(edge_info_df.iloc[:, 2].values, dtype=torch.float)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=len(node_names))

        # Create the reference target graph
        self.ref_target_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).to(self.device)
        self.ref_target_graph.node_names = node_names    # UniProt accessions

        logger.success(f"Reference target graph prepared.")


    def construct_custom_target_graph(
            self, 
            custom_seq_fasta: Optional[str] = None
    ):
        """
        Construct a custom target graph based on the provided custom sequence FASTA file.

        Args:
            custom_seq_fasta (str, optional): The path to the custom FASTA file used to construct the target graph.
        """
        logger.info(f'Constructing reference-custom target graph from {custom_seq_fasta}...')

        # Prepare graph edges by calculating similarity from the fasta file
        self.cal_similarity_from_fasta(custom_seq_fasta)
        edge_info_df = self.custom_target_sim.copy()

        # Prepare graph node embeddings from the custom target sequence file
        custom_embs = self.tgt_embedder.get_embs_from_fasta(self.custom_target_seq_file)
        self.tgt_embs.update(custom_embs)
        node_feature_df = pd.DataFrame(self.tgt_embs).T

        # Generate node name and ID mapping dictionary for new nodes
        new_nodes = list(set(self.custom_target) - set(self.ref_target))
        self.node_name2id.update({name: i+len(self.ref_target) for i, name in enumerate(new_nodes)})

        # Map the edge information to the new node IDs
        edge_info_df.iloc[:, 0] = edge_info_df.iloc[:, 0].map(self.node_name2id)
        edge_info_df.iloc[:, 1] = edge_info_df.iloc[:, 1].map(self.node_name2id)

        # Generate graph data
        x = torch.tensor(node_feature_df.loc[self.ref_target+new_nodes, :].values, dtype=torch.float)
        edge_index = torch.tensor(edge_info_df.iloc[:, [1, 0]].astype(int).values.T, dtype=torch.long).to(self.device)  # target, query for massage passing
        edge_index = torch.concat((self.ref_target_graph.edge_index, edge_index), dim=1)
        edge_weight = torch.tensor(edge_info_df.iloc[:, 2].values, dtype=torch.float).to(self.device)
        edge_weight = torch.concat((self.ref_target_graph.edge_attr, edge_weight), dim=0)

        # Add self-loops to the graph
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=len(self.ref_target+new_nodes))

        # Create the custom target graph
        self.custom_target_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).to(self.device)
        self.custom_target_graph.node_names = self.ref_target+new_nodes    # uniprot accession

        # logger.success(f"Reference-custom target graph created with {len(self.custom_target_graph.node_names)} nodes and {edge_weight.shape[0]} edges.")

    
    def cp_pertId2smiles_mapping(
            self, 
            csv_file: str,
            key_column_name: str = 'pert_id',
             value_column_name: str = 'canonical_smiles'
    ) -> dict:
        """
        Map perturbagen IDs to their corresponding SMILES representations from a CSV file.

        Args:
            csv_file (str): The path to the CSV file containing perturbation information.
            key_column_name (str, optional): The column name for perturbation IDs. Defaults to 'pert_id'.
            value_column_name (str, optional): The column name for SMILES representations. Defaults to 'canonical_smiles'.

        Returns:
            dict: A dictionary mapping perturbation IDs to their corresponding SMILES.
        """
        # Read the CSV file and filter out rows without PubChem CID
        # key_column_name, value_column_name = 'pert_id', 'canonical_smiles'
        df = pd.read_csv(csv_file).dropna(subset=['pubchem_cid']).loc[:, [key_column_name, value_column_name]].drop_duplicates()
        df = df[df[value_column_name]!='restricted']
        return dict(zip(df[key_column_name], df[value_column_name]))


    def tgt_symbol2uniprot_mapping(
            self, 
            csv_file: str, 
            key_column_name: str = 'Approved symbol',
            value_column_name: str = 'UniProt accession'
    ) -> dict:
        """
        Map target symbols to their corresponding UniProt accessions from a CSV file.

        Args:
            csv_file (str): The path to the CSV file containing target information.
            key_column_name (str, optional): The column name for target symbols. Defaults to 'Approved symbol'.
            value_column_name (str, optional): The column name for UniProt accessions. Defaults to 'UniProt accession'.

        Returns:
            dict: A dictionary mapping target symbols to their corresponding UniProt accessions.
        """
        df = pd.read_csv(csv_file).dropna(subset=[key_column_name, value_column_name]).loc[:, [key_column_name, value_column_name]].drop_duplicates()
        return dict(zip(df[key_column_name], df[value_column_name]))


    def load_unipert_reps(self):
        """
        Load UniPert representations from a pickle file.

        This method checks if the representations file exists and updates the current representations.
        """
        reps_path = os.path.join(self.save_dir, 'unipert_reps.pkl')
        if os.path.exists(reps_path):
            unipert_reps = pd.read_pickle(reps_path)
            self.unipert_reps.update(unipert_reps)    
            logger.success(f"Referece representations loaded.")
        # else:
        #     logger.warning(f"{reps_path} not found. Encoding reference targets...")
        
    ### ================= Inference ================= ###

    def encode_ref_perturbagens(
            self, 
            save: bool = False
    ):
        """
        Encode reference perturbagens and update the UniPert representations.

        Args:
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.
        """
        tgt_ptbg_info_file = os.path.join(self.data_dir, 'ref_targets.csv')
        self.cp_model.eval()
        self.tgt_encoder.eval()

        # Map perturbation IDs
        tgt_symbol2uniprot_dict = self.tgt_symbol2uniprot_mapping(tgt_ptbg_info_file)  # 'cmap_name' -> 'uniprot accession'

        # Encode genetic perturbagens
        tgt_unipert_reps = {}
        tgt_names = self.ref_target_graph.node_names   # uniprot accession
        tgt_reps = self.tgt_encoder(self.ref_target_graph.to(self.device)).detach().cpu().numpy() 
        assert len(tgt_names) == tgt_reps.shape[0]
        for gene_symbol, uniprot_id in tgt_symbol2uniprot_dict.items():
            self.tgt_embedder.saved_embs[gene_symbol] = self.tgt_embedder.saved_embs[uniprot_id]
            tgt_unipert_reps[gene_symbol] = tgt_reps[tgt_names.index(uniprot_id)]
        for uniprot_id in tgt_names:
            tgt_unipert_reps[uniprot_id] = tgt_reps[tgt_names.index(uniprot_id)]
        
        self.unipert_reps.update(tgt_unipert_reps)     
        logger.success(f"{len(tgt_names)} reference targets encoded.")  
            
        # Save generated UniPert representation
        if save:
            reps_path = os.path.join(self.save_dir, 'unipert_reps.pkl')
            with open(reps_path, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            logger.save(f'UniPert representations saved to {reps_path}.')


    def enc_gene_ptbgs_from_fasta(
            self, 
            custom_seq_fasta: Optional[str] = None,
            save: bool = False
    ) -> dict:
        """
        Encode genetic perturbagens from a provided FASTA file.

        Args:
            fasta_file (str, optional): The path to the custom FASTA file containing genetic perturbagens.
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            dict: A dictionary of encoded genetic perturbagens.
        """
        assert custom_seq_fasta is not None, "Please provide a fasta file."
        self.tgt_encoder.eval()

        # Prepare data by constructing the custom target graph
        self.construct_custom_target_graph(custom_seq_fasta)
        self.custom_target_graph = self.custom_target_graph.to(self.device)

        # Encode genetic perturbagens
        custom_target_reps = {}
        tgt_names = self.custom_target_graph.node_names    # uniprot accession
        tgt_reps = self.tgt_encoder(self.custom_target_graph).detach().cpu().numpy() 

        assert len(tgt_names) == tgt_reps.shape[0]          # Ensure names match representations
        for tgt_uid in self.custom_target:
            if tgt_uid in self.custom_target_ui2pn.keys():
                pns = self.custom_target_ui2pn[tgt_uid]
                for pn in pns:
                    custom_target_reps[pn] = tgt_reps[tgt_names.index(tgt_uid)]
            else:
                custom_target_reps[tgt_uid] = tgt_reps[tgt_names.index(tgt_uid)]
        self.unipert_reps.update(custom_target_reps)

        # Save representations if required
        if save:
            save_dir = os.path.join(self.save_dir, f'unipert_reps.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            logger.save(f'Custom perturbagen representations saved at {save_dir}.')
        return custom_target_reps


    def enc_gene_ptbgs_from_gene_names(
            self, 
            gene_names: List[str], 
            save: bool = False
    ) -> Tuple[dict, list]:
        """
        Encode genetic perturbagens based on provided gene names.

        Args:
            gene_names (list): A list of gene names to encode.
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            tuple: A tuple containing a dictionary of encoded perturbagens and a list of invalid inputs.
        """
        if not gene_names:
            logger.warning("gene_names is empty, skipping operation.")
            return

        self.load_unipert_reps()
        self.encode_ref_perturbagens()
        gene_names = list(set(gene_names))
        out_reps = {}
        invalid_inputs = []
        records = []
        logger.info(f"Encoding {len(gene_names)} genetic perturbagens with UniPert...")
        for gene in tqdm(gene_names):
            if gene in list(self.unipert_reps.keys()):
                out_reps[gene] = self.unipert_reps[gene]
                continue
            if gene.upper() in list(self.unipert_reps.keys()):
                out_reps[gene] = self.unipert_reps[gene.upper()]
                continue
            proid_seq = get_tgt_seq_from_gene_name(gene)   # [uniprot_id, head, seq]
            if proid_seq:
                self.custom_target_ui2pn[proid_seq[0]].append(gene)
                rec = SeqRecord(Seq(proid_seq[2]), 
                                id=proid_seq[1], 
                                description=f'PN={gene}'    # pert name
                                )
                records.append(rec) 
            else:
                invalid_inputs.append(gene)

        # Write to FASTA file if records exist
        if records != []:
            SeqIO.write(records, self.custom_target_seq_file, "fasta")
            reps = self.enc_gene_ptbgs_from_fasta(
                self.custom_target_seq_file,
                save=False
                )
            out_reps.update(reps)

        # Save representations if required
        if save:
            reps_path = os.path.join(self.save_dir, f'unipert_reps.pkl')
            self.unipert_reps.update(out_reps)
            with open(reps_path, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            logger.save(f'Custom perturbagen representations saved at {reps_path}.')

        logger.success(f"{len(out_reps)} encoded succesfully, {len(invalid_inputs)} failed.")
        return out_reps, invalid_inputs
    

    def enc_gene_ptbgs_from_uniprot_accessions(
            self, 
            uniprot_accessions: List[str],  
            save: bool = False
    ) -> Tuple[dict, list]:  
        """
        Encode genetic perturbagens based on provided UniProt accessions.

        Args:
            uniprot_accessions (list): A list of UniProt accessions to encode.
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            tuple: A tuple containing a dictionary of encoded perturbations and a list of invalid inputs.
        """
        if not uniprot_accessions:
            logger.warning("uniprot_accessions is empty, skipping operation.")
            return

        self.load_unipert_reps()
        self.encode_ref_perturbagens()
        uniprot_accessions = list(set(uniprot_accessions))
        out_reps = {}
        invalid_inputs = []
        records = []
        logger.info(f"Encoding {len(uniprot_accessions)} genetic perturbagens with UniPert...")
        for query_ua in tqdm(uniprot_accessions):
            if query_ua in list(self.unipert_reps.keys()):
                out_reps[query_ua] = self.unipert_reps[query_ua]
                continue
            if query_ua.upper() in list(self.unipert_reps.keys()):
                out_reps[query_ua] = self.unipert_reps[query_ua.upper()]
                continue
            proid_seq = get_tgt_seq_from_uniprot_accession(query_ua)   # [uniprot_id, head, seq]
            if proid_seq:
                self.custom_target_ui2pn[proid_seq[0]].append(query_ua)
                rec = SeqRecord(Seq(proid_seq[2]), 
                                id=proid_seq[1], 
                                description=f'PN={query_ua}'    # pert name
                                )
                records.append(rec) 
            else:
                invalid_inputs.append(query_ua)

       # Write to FASTA file if records exist
        if records != []:
            SeqIO.write(records, self.custom_target_seq_file, "fasta")
            reps = self.encode_genetic_perturbagens_from_fasta(
                self.custom_target_seq_file,
                save=False
                )
            out_reps.update(reps)

        # Save representations if required
        if save:
            reps_path = os.path.join(self.save_dir, f'unipert_reps.pkl')
            self.unipert_reps.update(out_reps)
            with open(reps_path, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            logger.save(f'Custom perturbagen representations saved at {reps_path}.')
        return out_reps, invalid_inputs
        

    def enc_chem_ptbgs_from_smiles(
            self, 
            smiles_list: List[str],
            save: bool = False
    ) -> dict:
        """
        Encode chemical perturbations from a list of SMILES strings.

        Args:
            smiles_list (list): A list of SMILES strings representing chemical compounds.
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            dict: A dictionary of encoded chemical perturbations.
        """
        # Get SMILES embedder
        self.cp_model.eval()
        unembedded_sms = [s for s in smiles_list if s not in self.cp_model.embedder.saved_embs.keys()]
        new_embs = self.cp_model.embedder.get_emb_from_smiles_list(unembedded_sms)
        self.cp_model.embedder.saved_embs.update(new_embs)
        self.cp_model.emb_dict = self.cp_model.embedder.saved_embs

        # Encode chemical perturbagens
        custom_compound_reps = {}
        if not smiles_list: 
            return custom_compound_reps
        
        cp_reps = self.cp_model(smiles_list, self.device).detach().cpu().numpy()   
        assert len(smiles_list) == cp_reps.shape[0]
        for i, (sm, reps) in enumerate(zip(smiles_list, cp_reps)):
            custom_compound_reps[sm] = cp_reps[i]
        self.unipert_reps.update(custom_compound_reps)

        # Save representations if required
        if save:
            save_dir = os.path.join(self.save_dir, f'unipert_reps.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            logger.save(f'Custom perturbagen representations saved at {save_dir}.')

        logger.success(f"{len(custom_compound_reps)} encoded succesfully.")
        return custom_compound_reps


    def enc_chem_ptbgs_from_sms_csv(
            self, 
            custom_cp_sms_csv: str, 
            cp_col_name: str = 'Compound', 
            sms_col_name: str = 'SMILES', 
            save: bool = False
    ) -> dict:
        """
        Encode chemical perturbations from a CSV file containing compound and SMILES information.

        Args:
            custom_cp_sms_csv (str): The path to the CSV file containing compound and SMILES data.
            cp_col_name (str, optional): The column name for compound names. Defaults to 'Compound'.
            sms_col_name (str, optional): The column name for SMILES representations. Defaults to 'SMILES'.
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            dict: A dictionary of encoded chemical perturbations.
        """
        self.cp_model.eval()

        # Get compound and SMILES data
        cp_sms_df = pd.read_csv(custom_cp_sms_csv).loc[:, [cp_col_name, sms_col_name]]
        cp_sms_df = cp_sms_df.dropna(subset=[sms_col_name])
        cp_sms_df = cp_sms_df.drop_duplicates()
        assert len(cp_sms_df) > 0, "No custom compounds found in the provided csv file."

        cp_sms_df['is_valid'] = cp_sms_df[sms_col_name].apply(lambda sms: check_smiles(sms))
        cp_sms_df = cp_sms_df[cp_sms_df['is_valid']==True]
        compound_lists = cp_sms_df[cp_col_name].tolist()
        smiles_list = cp_sms_df[sms_col_name].tolist()

        # Get SMILES embedder
        _ = self.cp_model.embedder.get_emb_from_smiles_list(smiles_list)
        self.cp_model.emb_dict = self.cp_model.embedder.saved_embs

        # Encode perturbagens
        custom_compound_reps = {}
        if not smiles_list: 
            return custom_compound_reps, compound_lists

        cp_reps = self.cp_model(smiles_list, self.device).detach().cpu().numpy()   
        assert len(smiles_list) == cp_reps.shape[0] == len(compound_lists)
        for i, (cp, _) in enumerate(zip(compound_lists, smiles_list)):
            custom_compound_reps[cp] = cp_reps[i]
        
        self.unipert_reps.update(custom_compound_reps)

        # Save representations if required
        if save:
            save_dir = os.path.join(self.save_dir, f'perturbagen_reps.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            print(f'Custom perturbagen representations saved at {save_dir}.')
        
        return custom_compound_reps, []
    

    def enc_chem_ptbgs_from_dict(
            self, 
            cp_sms_dict: str, 
            save: bool = False
    ) -> dict:
        """
        Encode chemical perturbations from a dict with keys of compound names and values of SMILES information.

        Args:
            cp_sms_dict (dict): A dictionary containing compound names as keys and SMILES as values.
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            dict: A dictionary of encoded chemical perturbations and invalid SMILES.
        """

        self.cp_model.eval()

        # Validate input
        if not isinstance(cp_sms_dict, dict) or not cp_sms_dict:
            raise ValueError("cp_sms_dict must be a non-empty dictionary.")
        
        invalid_cps = []
        for cp in list(cp_sms_dict.keys()): 
            if not check_smiles(cp_sms_dict[cp]):
                invalid_cps.append(cp) 
                cp_sms_dict.pop(cp)  

        smiles_list = list(cp_sms_dict.values())
        compound_lists = list(cp_sms_dict.keys())

        # Get SMILES embedder
        _ = self.cp_model.embedder.get_emb_from_smiles_list(smiles_list)
        self.cp_model.emb_dict = self.cp_model.embedder.saved_embs

        # Encode perturbagens
        custom_compound_reps = {}
        if not smiles_list: return custom_compound_reps, invalid_cps
    
        cp_reps = self.cp_model(smiles_list, self.device).detach().cpu().numpy()   
        assert len(smiles_list) == cp_reps.shape[0] == len(compound_lists)
        for i, (cp, _) in enumerate(zip(compound_lists, smiles_list)):
            custom_compound_reps[cp] = cp_reps[i]
        
        self.unipert_reps.update(custom_compound_reps)

        # Save representations if required
        if save:
            save_dir = os.path.join(self.save_dir, f'perturbagen_reps.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            print(f'Custom perturbagen representations saved at {save_dir}.')

        return custom_compound_reps, invalid_cps
    

    def enc_chem_ptbgs_from_compound_names(
            self, 
            compound_names: List[str], 
            save: bool = False
    ) -> dict:
        """
        Encode chemical perturbations from a list of compounds names.

        Args:
            compound_names (list): A list of compound names to encode. 
            save (bool, optional): Whether to save the updated representations to a file. Defaults to False.

        Returns:
            dict: A dictionary of encoded chemical perturbations and invalid SMILES.
        """

        self.cp_model.eval()

        # Validate input
        if not isinstance(compound_names, list) or not compound_names:
            raise ValueError("compound_names must be a non-empty list.")
        
        # Check API 
        if not self.cp_server: self.prepare_cp_server()
        if not self.cp_server: return {}, compound_names

        # Retrieve SMILES
        invalid_cps = []
        smiles_list = []
        compound_lists = []
        for cp in compound_names: 
            sms = get_cp_sms_from_compound_name(cp, server=self.cp_server, server_name=self.cp_server_name)  
            if sms and check_smiles(sms):
                smiles_list.append(sms)
                compound_lists.append(cp)
            else:
                invalid_cps.append(cp)

        # Get SMILES embedder
        _ = self.cp_model.embedder.get_emb_from_smiles_list(smiles_list)
        self.cp_model.emb_dict = self.cp_model.embedder.saved_embs

        # Encode perturbagens
        custom_compound_reps = {}
        if not smiles_list: 
            return custom_compound_reps, invalid_cps

        cp_reps = self.cp_model(smiles_list, self.device).detach().cpu().numpy()   
        assert len(smiles_list) == cp_reps.shape[0] == len(compound_lists)
        for i, (cp, _) in enumerate(zip(compound_lists, smiles_list)):
            custom_compound_reps[cp] = cp_reps[i]
        
        self.unipert_reps.update(custom_compound_reps)

        # Save representations if required
        if save:
            save_dir = os.path.join(self.save_dir, f'perturbagen_reps.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(self.unipert_reps, f)
            print(f'Custom perturbagen representations saved at {save_dir}.')

        return custom_compound_reps, invalid_cps


    def enc_ptbgs_for_pert_adata(
            self,
            adata: anndata.AnnData,  
            ptbg_cols: List[str] = ['perturbation'],  
            ptbg_types: Optional[List[str]] = None, 
            control_key: str = 'control',
            return_results: bool = False
    )-> dict:
        """
        Generate embeddings for perturbagens based on the given AnnData object.

        Args:
            adata (anndata.AnnData): A required AnnData object containing perturbation information.
            ptbg_cols (list of str): Required list of adata.obs.column names indicating perturbagen info.
            ptbg_types (list of str, optional): An optional list of strings indicating the type of each perturbation key. Valid types include 'genetic' and 'chemical'.
            control_key (str, optional): The key used to identify control perturbations. Defaults to 'control'.
            return_results (bool, optional): Whether to return the results as a dictionary. Defaults to False.

        Returns:
            dict, optional: A dictionary containing UniPert representations and invalid perturbations if return_results is True.
        """
        # adata = adata.copy()

        valid_types = ['genetic', 'chemical']
        if not isinstance(ptbg_cols, list) or not isinstance(ptbg_types, list):
            raise ValueError("ptbg_keys and ptbg_types must be list")
        elif len(ptbg_cols) != len(ptbg_types):
            raise ValueError("ptbg_keys and ptbg_types must have the same length")
        elif any(t not in valid_types for t in ptbg_types):
            raise ValueError("ptbg_types must be 'genetic' or 'chemical'")
        
        genetic_ptbgs = []
        chemical_ptbgs = []
        invalid_ptbgs = []

        for key, typ in zip(ptbg_cols, ptbg_types):
            if key not in adata.obs.columns:
                raise ValueError(f"{key} not found in adata.obs.columns")
            else:
                ptbgs = list(adata.obs.dropna(subset=[key])[key].unique())
                if control_key in ptbgs: ptbgs.remove(control_key)
                if typ == 'genetic':
                    genetic_ptbgs.extend(ptbgs)
                elif typ == 'chemical':
                    chemical_ptbgs.extend(ptbgs)    

        unipert_embs = {}
        # Encode genetic perturbagens
        if genetic_ptbgs != []:
            genetic_ptbgs = list(set(genetic_ptbgs))
            records = []
            logger.info(f"Retrieving sequence for genetic perturbagens...")
            for gene in tqdm(genetic_ptbgs):
                if gene in self.unipert_reps.keys():
                    unipert_embs[gene] = self.unipert_reps[gene]
                    continue
                proid_seq = get_tgt_seq_from_gene_name(gene)   # [uniprot_id, head, seq]
                if proid_seq:
                    self.custom_target_ui2pn[proid_seq[0]].append(gene)
                    rec = SeqRecord(Seq(proid_seq[2]), 
                                    id=proid_seq[1], 
                                    description=f'PN={gene}'    # pert name
                                    )
                    records.append(rec) 
                else:
                    invalid_ptbgs.append(gene)
                    
            if records:
                # Write to FASTA file
                # records = list(set(records))
                SeqIO.write(records, self.custom_target_seq_file, "fasta")
                genetic_ptbg_embs = self.enc_gene_ptbgs_from_fasta(
                    self.custom_target_seq_file,
                    save=False
                    )
                unipert_embs.update(genetic_ptbg_embs)

        # Encode chemical perturbagens
        chemical_ptbg_embs = None
        if chemical_ptbgs != []:
            if not self.cp_server: 
                self.prepare_cp_server()
            chemical_ptbgs = list(set(chemical_ptbgs))
            cp_sms = {}
            logger.info("Retrievaling SMILES for chemical perturbagens...")
            for compound in tqdm(chemical_ptbgs):
                if compound in self.unipert_reps.keys():
                    unipert_embs[compound] = self.unipert_reps[compound]
                    continue
                sms = get_cp_sms_from_compound_name(compound, server=self.cp_server, server_name=self.cp_server_name)
                if sms:
                    cp_sms[compound] = sms
                else:
                    invalid_ptbgs.append(compound)
            
            if cp_sms:
                # Transfer chemical SMILES (dict) to DataFrame and write to CSV file
                chemical_ptbg_df = pd.DataFrame(list(cp_sms.items()), columns=['Compound', 'SMILES'])
                chemical_ptbg_df.to_csv(self.custom_compound_smiles_file, index=False)
                chemical_ptbg_embs, _ = self.enc_chem_ptbgs_from_sms_csv(
                    self.custom_compound_smiles_file, 
                    cp_col_name='Compound', 
                    sms_col_name='SMILES',
                    save=False
                    )
                unipert_embs.update(chemical_ptbg_embs)

        # Save UniPert embeddings to adata.uns['UniPert_reps']
        adata.uns['UniPert_reps'] = unipert_embs
        adata.uns['invalid_ptbgs'] = invalid_ptbgs
        logger.success(colors.green(f'UniPert representations generated!'))
        logger.info(f"{len(unipert_embs)} perturbagens' UniPert representations saved to adata.uns['UniPert_reps']")
        
        if invalid_ptbgs:
            logger.warning(f"{len(invalid_ptbgs)} perturbagens can not be repersentated and saved to adata.uns['invalid_ptbgs']: \n{invalid_ptbgs}")

        if return_results:
            return {'UniPert_reps': unipert_embs, 'invalid_ptbgs': invalid_ptbgs}