from typing import List, Tuple, Optional
import pickle
import anndata
import pandas as pd
import subprocess
from tqdm import tqdm
from collections import defaultdict
from lamin_utils import logger, colors

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
    def __init__(self, data_dir: str = DATA_DIR, model_dir: str = MODEL_DIR, model_name='unipert_model.pt'):
        """
        Initialize the UniPert model with specified data and model directories.

        Args:
            data_dir (str, optional): The directory where data files are located. Defaults to DATA_DIR.
            model_dir (str, optional): The directory where the model files are stored. Defaults to MODEL_DIR.
        """
        # Paths
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.mmseqs_cache_dir = MMSEQS_CACHE_DIR

        # Device
        if torch.cuda.is_available() and torch.version.cuda:
            self.device = torch.device('cuda:0')
            logger.info("CUDA is available. Using CUDA.")
        else:
            self.device = torch.device('cpu')
            logger.info("CUDA is not available. Using CPU instead.")

        # Compound search service
        self.compound_search_service = None
        self.compound_search_service_name = None

        # Reference target graph information
        self.ref_target_seq_file = os.path.join(self.data_dir, 'ref_target_seq.fasta')
        self.ref_target_sim_file = os.path.join(self.data_dir, 'ref_target_seq_similarity.csv')
        self.ref_target = None
        self.ref_target_sim = None
        self.ref_target_graph = None

        # Custom target graph information
        self.custom_target_seq_file = os.path.join(self.data_dir, 'custom_target_seq.fasta')
        self.custom_target_sim_file = os.path.join(self.data_dir, 'custom_target_seq_similarity.csv')
        self.custom_target = None
        self.custom_target_sim = None
        self.custom_target_graph = None
        self.custom_target_ui2pn = defaultdict(list)     # {UniProt_id: [perturbagen_names]}

        # Graph node ID mapping 
        self.node_name2id = None

        # Custom compound cache
        self.custom_compound_smiles_file = os.path.join(self.data_dir, 'custom_compound_smiles.txt')

        # UniPert representation cache
        self.unipert_reps = {}    
        self.ref_target_encoded = False

        # Load pretrained model
        self.model_loaded = False
        try:
            self.load_model(model_dir=model_dir, model_name=model_name)
            self.model_loaded = True
        except FileNotFoundError:
            logger.error(f"Model not found at [{model_dir}]. Please download [{model_name}].")

        # Prepare reference graph  
        self._prepare_ref_target_graph() 
        
        # Initialization complete
        if self.model_loaded: logger.success(colors.green(f"Model loaded and initialized."))


    def set_model_hparams(self, params: Optional[dict] = None):
        """
        Initialize UniPert model configuration.
        """
        self.model_hparams = {
            'out_dim': 256,
            'gnn_type': 'GCN',
            'layer_num': 2,
            'target_embedder': 'ESM2',
            'compound_embedder': 'ECFP4',
        }
        for key, value in self.model_hparams.items():
            setattr(self, key, value)
        

    def _build_model(self) -> nn.ModuleList:
        """
        Build the UniPert model architecture.

        Returns:
            nn.ModuleList: A list of Builded UniPert models including compound and target models.
        """
        logger.info(colors.yellow(f"Building UniPert model..."))

        # Prepare embedders
        self.tgt_embedder = TargetEmbedderESM2(data_dir=self.data_dir, device=self.device)
        self.cp_embedder = CompoundEmbedderECFP4(data_dir=self.data_dir, device=self.device)
        self.tgt_embs = None
        self.cp_embs = None

        # Prepare UniPert model
        self.cp_model = CompoundEncoder(embedder=self.cp_embedder, output_dim=self.out_dim)
        self.tgt_encoder = GNN_Encoder(layer_sizes=[self.tgt_embedder.emb_dim]+[self.out_dim]*self.layer_num, gnn_type=self.gnn_type, batchnorm=True)
        self.tgt_predictor = MLP_Predictor(self.out_dim, self.out_dim, hidden_size=self.out_dim*4)
        self.tgt_model = BGRL(self.tgt_encoder, self.tgt_predictor)
        
        unipert_models = nn.ModuleList([])
        unipert_models.append(self.cp_model)
        unipert_models.append(self.tgt_model)

        logger.success(f"UniPert model architecture initialized.")
        return unipert_models


    def load_model(
            self,
            model_dir: Optional[str] = None,
            model_name: str = 'unipert_model.pt',
    ):
        """
        Load pretrained UniPert model weights.

        Args:
            model_dir:Directory containing pretrained model weights.
            model_name:Name of the model checkpoint file.
        """
        # Resolve model checkpoint path
        model_dir = model_dir if model_dir is not None else self.model_dir
        model_path = os.path.join(model_dir, model_name)

        # Load model weights
        model_weights = torch.load(model_path,map_location=self.device,weights_only=True)

        # Initialize model configuration
        self.set_model_hparams()

        # Build model architecture
        unipert_models = self._build_model()
        unipert_models = unipert_models.to(self.device)

        # Load pretrained weights
        self.cp_model.load_state_dict(model_weights['cp_encoder'])
        self.tgt_encoder.load_state_dict(model_weights['tgt_encoder'])

        logger.download("Pretrained UniPert model loaded.")
            

    def _prepare_mmseqs_database(self):
        """
        Initialize the MMseqs2 reference database.
        """
        logger.info('Initialize the MMseqs2 reference database...')
        import shutil

        # Reset MMseqs cache directory
        if os.path.exists(self.mmseqs_cache_dir):
            shutil.rmtree(self.mmseqs_cache_dir)
        os.makedirs(self.mmseqs_cache_dir, exist_ok=True)  
        
        # Create MMseqs reference database
        ref_db_path = os.path.join(self.mmseqs_cache_dir, 'ref')
        subprocess.run(["mmseqs", "createdb", self.ref_target_seq_file, ref_db_path], check=True)

        logger.success('MMseqs reference database created.')


    def _initialize_compound_search_service(self):
        """
        Initialize compound search services using PubChem or ChemSpider.
        """
        # Try PubChem
        pubchem_service = check_pubchempy_service()
        if pubchem_service:
            # import pubchempy as pcp
            self.compound_search_service = pubchem_service
            self.compound_search_service_name = 'pubchem'  
            logger.success(f"PubChem service initialized successfully.")
            return

        # Try ChemSpider
        chemspider_service = check_chemspipy_service()
        if chemspider_service:
            # from chemspipy import ChemSpider
            self.compound_search_service = chemspider_service
            self.compound_search_service_name = 'chemspider'
            logger.success(f"chemspider server initialized successfully.")
            return

        logger.warning(f"No compound search service is available! Please check your network connection.")
        self.compound_search_service = None
        self.compound_search_service_name = None


    def _compute_sequence_similarity(
            self, 
            custom_seq_fasta: Optional[str] = None, 
            save_sim_file: bool = True
    ):
        """
        Compute the similarity between custom sequences and a reference FASTA file.

        Args:
            custom_seq_fasta (str, optional): The path to the custom FASTA file. If provided, it will be used for similarity calculation.
            save_sim_file (bool, optional): Whether to save the similarity results to a CSV file. Defaults to True.
        """
        # Prepare MMseqs2 reference database
        ref_db_path = os.path.join(self.mmseqs_cache_dir, 'ref')
        if not os.path.exists(ref_db_path): 
            self._prepare_mmseqs_database()

        # Set custom target sequence FASTA file
        if custom_seq_fasta is not None:
            self.custom_target_seq_file = custom_seq_fasta
            # self.custom_target_sim_file = self.custom_target_seq_file.replace('.fasta', '_similarity.csv')

        # Extract unique IDs of custom target sequences
        self.custom_target = list(set([seq_record.id.split('|')[1] if '|' in seq_record.id else seq_record.id for seq_record in SeqIO.parse(self.custom_target_seq_file, "fasta")]))

        logger.info(f'Computing similarities between {self.custom_target_seq_file} and reference sequences...')

        # Run MMseqs2 with search commond
        # Define MMseqs2 working paths
        query_db_path = os.path.join(self.mmseqs_cache_dir, 'query')
        result_db_path = os.path.join(self.mmseqs_cache_dir, 'result')
        tmp_dir = os.path.join(self.mmseqs_cache_dir, 'tmp')
        aln_result = os.path.join(self.mmseqs_cache_dir, 'alnResult.m8')
        
        # Clean previous MMseqs2 outputs to avoid FileExistsError
        import glob
        import shutil
        for f in glob.glob(f"{query_db_path}*") + glob.glob(f"{result_db_path}*"):
            try:
                os.remove(f)
            except OSError:
                pass
        if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
        
        # Run MMseqs2
        subprocess.run(["mmseqs", "createdb", self.custom_target_seq_file, query_db_path], check=True)
        subprocess.run(["mmseqs", "search", query_db_path, ref_db_path, result_db_path, tmp_dir, "-a"], check=True)
        subprocess.run(["mmseqs", "convertalis", query_db_path, ref_db_path, result_db_path, aln_result, "--format-output", "query,target,nident,alnlen,fident", "--format-mode", "4"], check=True)
        
        result_df = pd.read_csv(aln_result, sep='\t')
        result_df["fident_exact"] = (result_df["nident"]/result_df["alnlen"])
        sim_df = result_df.loc[:, ["query", "target", "fident_exact"]]
        sim_df.to_csv(self.custom_target_sim_file, header=None, index=False)
        self.custom_target_sim = sim_df

        # If required, save the similarity results to a CSV file
        if save_sim_file:
            self.custom_target_sim.to_csv(self.custom_target_sim_file, index=False, header=False)
            logger.save(f"MMseqs2 results saved to {self.custom_target_sim_file}")


    def _prepare_ref_target_graph(self):
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
        
        # Check if the reference target sequence FASTA file exists
        if not os.path.exists(self.ref_target_seq_file):
            raise FileNotFoundError(
                f"Reference sequence file not found: "
                f"{self.ref_target_seq_file}"
            )

        # Prepare graph edges if similarity data is not available
        if self.ref_target_sim is None:
            if not os.path.exists(self.ref_target_sim_file):
                self._compute_sequence_similarity(self.ref_target_seq_file)

        # Load similarity table
        self.ref_target_sim = pd.read_csv(self.ref_target_sim_file, header=None).drop_duplicates()
        edge_info_df = self.ref_target_sim.copy()

        # Extract graph node names (UniProt IDs) from the reference sequence file
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

        # Build the reference target graph object
        self.ref_target_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).to(self.device)
        self.ref_target_graph.node_names = node_names    # UniProt accessions

        logger.success(f"Reference target graph prepared.")


    def _build_custom_target_graph(
            self, 
            custom_seq_fasta: Optional[str] = None
    ):
        """
        Build a custom target graph from input custom sequence FASTA file.

        Args:
            custom_seq_fasta (str, optional): Path to the custom FASTA file.
        """
        logger.info(f'Building reference-custom target graph from {custom_seq_fasta} ...')

        # Compute sequence similarity
        self._compute_sequence_similarity(custom_seq_fasta)
        edge_info_df = self.custom_target_sim.copy()

        # Prepare graph node embeddings
        custom_embs = self.tgt_embedder.get_embs_from_fasta(self.custom_target_seq_file)
        self.tgt_embs.update(custom_embs)
        node_feature_df = pd.DataFrame(self.tgt_embs).T

        # Add new nodes to mapping
        new_nodes = list(set(self.custom_target) - set(self.ref_target))
        self.node_name2id.update({name: i+len(self.ref_target) for i, name in enumerate(new_nodes)})

        # Map node names to node ids
        edge_info_df.iloc[:, 0] = edge_info_df.iloc[:, 0].map(self.node_name2id)
        edge_info_df.iloc[:, 1] = edge_info_df.iloc[:, 1].map(self.node_name2id)

        # Generate graph data
        x = torch.tensor(node_feature_df.loc[self.ref_target+new_nodes, :].values, dtype=torch.float)
        edge_index = torch.tensor(edge_info_df.iloc[:, [1, 0]].astype(int).values.T, dtype=torch.long).to(self.device)  # target, query for massage passing
        edge_index = torch.concat((self.ref_target_graph.edge_index, edge_index), dim=1)
        edge_weight = torch.tensor(edge_info_df.iloc[:, 2].values, dtype=torch.float).to(self.device)
        edge_weight = torch.concat((self.ref_target_graph.edge_attr, edge_weight), dim=0)

        # Add self-loops
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=len(self.ref_target+new_nodes))

        # Build custom target graph object
        self.custom_target_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).to(self.device)
        self.custom_target_graph.node_names = self.ref_target+new_nodes    # uniprot accession

        # logger.success(f"Reference-custom target graph created with {len(self.custom_target_graph.node_names)} nodes and {edge_weight.shape[0]} edges.")


    def _load_gene_to_uniprot_mapping(
            self, 
            csv_file: str, 
            gene_column: str = 'Approved symbol',
            uniprot_column: str = 'UniProt accession'
    ) -> dict:
        """
        Map target symbols to their corresponding UniProt accessions from a CSV file.

        Args:
            csv_file (str): The path to the CSV file containing target information.
            gene_column (str, optional): The column name for target symbols. Defaults to 'Approved symbol'.
            uniprot_column (str, optional): The column name for UniProt accessions. Defaults to 'UniProt accession'.

        Returns:
            dict: A dictionary mapping target symbols to their corresponding UniProt accessions.
        """
        df = pd.read_csv(csv_file).dropna(subset=[gene_column, uniprot_column]).loc[:, [gene_column, uniprot_column]].drop_duplicates()
        return dict(zip(df[gene_column], df[uniprot_column]))


    def load_unipert_reps(self):
        """
        Load cached UniPert representations from disk.
        """
        reps_path = os.path.join(self.model_dir,'unipert_reps.pkl')
        if os.path.exists(reps_path):
            with open(reps_path, "rb") as f:
                cached_unipert_reps = pickle.load(f)
            self.unipert_reps.update(cached_unipert_reps)
            logger.success(f"Loaded cached referece UniPert representations.")

    def _save_reps(
            self,
            reps: dict,
            save_path: str,
    ):
        with open(save_path, "wb") as f:
            pickle.dump(reps, f)
        logger.save(f"UniPert representations saved to {save_path}.")

    ### ================= Application ================= ###

    def _encode_ref_targets(
            self, 
            save: bool = False
    ):
        """
        Encode reference perturbagens and update the UniPert representations.
        """
        if self.ref_target_encoded: return

        tgt_ptbg_info_file = os.path.join(self.data_dir, 'ref_targets.csv')
        self.tgt_encoder.eval()

        # Map perturbation IDs
        tgt_symbol2uniprot_dict = self._load_gene_to_uniprot_mapping(tgt_ptbg_info_file)  # 'cmap_name' -> 'uniprot accession'

        # Encode genetic perturbagens
        tgt_names = self.ref_target_graph.node_names   # uniprot accession
        tgt_index = {name:idx for idx, name in enumerate(tgt_names)}
        tgt_reps = self.tgt_encoder(self.ref_target_graph.to(self.device)).detach().cpu().numpy() 
        assert len(tgt_names) == tgt_reps.shape[0]
        tgt_unipert_reps = {}
        for gene_symbol, uniprot_id in tgt_symbol2uniprot_dict.items():
            self.tgt_embedder.saved_embs[gene_symbol] = self.tgt_embedder.saved_embs[uniprot_id]
            tgt_unipert_reps[gene_symbol] = tgt_reps[tgt_index[uniprot_id]]
        for uniprot_id in tgt_names:
            tgt_unipert_reps[uniprot_id] = tgt_reps[tgt_index[uniprot_id]]
        
        self.unipert_reps.update(tgt_unipert_reps)    
        self.ref_target_encoded = True
        logger.info(f"{len(tgt_names)} reference targets encoded.")  


    def _encode_genes_from_fasta(
            self, 
            fasta_file: str,
    ) -> dict:
        """
        Encode genetic perturbagens from FASTA file.

        Args:
            fasta_file (str, optional): The path to the custom FASTA file containing genetic perturbagens.

        Returns:
            dict: A dictionary of encoded genetic perturbagens.
        """

        self.tgt_encoder.eval()

        # Prepare data by constructing the custom target graph
        self._build_custom_target_graph(fasta_file)
        self.custom_target_graph = self.custom_target_graph.to(self.device)

        # Encode genetic perturbagens
        tgt_names = self.custom_target_graph.node_names    # uniprot accession
        tgt_index = {name:idx for idx, name in enumerate(tgt_names)}
        tgt_reps = self.tgt_encoder(self.custom_target_graph).detach().cpu().numpy() 
        assert len(tgt_names) == tgt_reps.shape[0]          

        custom_target_reps = {}
        for tgt_uid in self.custom_target:
            rep = tgt_reps[tgt_index[tgt_uid]]
            if tgt_uid in self.custom_target_ui2pn.keys():
                pns = self.custom_target_ui2pn[tgt_uid]
                for pn in pns:
                    custom_target_reps[pn] = rep
            else:
                custom_target_reps[tgt_uid] = rep
        self.unipert_reps.update(custom_target_reps)

        return custom_target_reps, []


    def _encode_genes_from_gene_names(
            self, 
            gene_names: List[str], 
    ) -> Tuple[dict, list]:
        """
        Encode genetic perturbagens based on provided gene names.

        Args:
            gene_names (list): A list of gene names to encode.

        Returns:
            tuple: A tuple containing a dictionary of encoded perturbagens and a list of invalid inputs.
        """
        if not gene_names:
            logger.warning("gene_names is empty.")
            return {}, []

        # self.load_unipert_reps()
        self._encode_ref_targets()
        gene_names = list(set(gene_names))
        out_reps = {}
        invalid_inputs = []
        records = []
        logger.info(f"Encoding {len(gene_names)} genetic perturbagens with UniPert...")
        for gene in tqdm(gene_names):
            if gene in self.unipert_reps:
                out_reps[gene] = self.unipert_reps[gene]
                continue
            if gene.upper() in self.unipert_reps:
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
        if records:
            SeqIO.write(records, self.custom_target_seq_file, "fasta")
            new_reps, _ = self._encode_genes_from_fasta(self.custom_target_seq_file)
            out_reps.update(new_reps)

        return out_reps, invalid_inputs
    

    def _encode_genes_from_uniprot_ids(
            self, 
            uniprot_accessions: List[str],  
    ) -> Tuple[dict, list]:  
        """
        Encode genetic perturbagens based on provided UniProt accessions.

        Args:
            uniprot_accessions (list): A list of UniProt accessions to encode.

        Returns:
            tuple: A tuple containing a dictionary of encoded perturbations and a list of invalid inputs.
        """
        if not uniprot_accessions:
            logger.warning("uniprot_accessions is empty.")
            return {}, []

        # self.load_unipert_reps()
        self._encode_ref_targets()
        uniprot_accessions = list(set(uniprot_accessions))
        out_reps = {}
        invalid_inputs = []
        records = []
        logger.info(f"Encoding {len(uniprot_accessions)} genetic perturbagens with UniPert...")
        for query_ua in tqdm(uniprot_accessions):
            if query_ua in self.unipert_reps:
                out_reps[query_ua] = self.unipert_reps[query_ua]
                continue
            if query_ua.upper() in self.unipert_reps:
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
            new_reps = self.encode_genetic_perturbagens_from_fasta(
                self.custom_target_seq_file,
                save=False
                )
            out_reps.update(new_reps)

        return out_reps, invalid_inputs
        

    def _encode_compounds_from_smiles(
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
        logger.info(f"Encoding {len(smiles_list)} chemcial perturbagens with UniPert...")

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

        return custom_compound_reps


    def _encode_compounds_from_csv(
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

        logger.info(f"Encoding {len(smiles_list)} chemcial perturbagens with UniPert...")

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

        return custom_compound_reps, []
    

    def _encode_compounds_from_dict(
            self, 
            cp_sms_dict: str, 
    ) -> dict:
        """
        Encode chemical perturbations from a dict with keys of compound names and values of SMILES information.

        Args:
            cp_sms_dict (dict): A dictionary containing compound names as keys and SMILES as values.

        Returns:
            dict: A dictionary of encoded chemical perturbations and invalid SMILES.
        """

        logger.info(f"Encoding {len(cp_sms_dict)} chemcial perturbagens with UniPert...")

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

        return custom_compound_reps, invalid_cps
    

    def _encode_compounds_from_names(
            self, 
            compound_names: List[str], 
            save: bool = False
    ) -> dict:
        """
        Encode chemical perturbations from a list of compounds names.

        Args:
            compound_names (list): A list of compound names to encode. 

        Returns:
            dict: A dictionary of encoded chemical perturbations and invalid SMILES.
        """

        self.cp_model.eval()

        # Validate input
        if not isinstance(compound_names, list) or not compound_names:
            raise ValueError("compound_names must be a non-empty list.")

        logger.info(f"Encoding {len(compound_names)} chemcial perturbagens with UniPert...")
        
        # Check API 
        if not self.compound_search_service: self._initialize_compound_search_service()
        if not self.compound_search_service: return {}, compound_names

        # Retrieve SMILES
        invalid_cps = []
        smiles_list = []
        compound_lists = []
        logger.info(f"Retrievaling SMILES for chemical perturbagens...")
        for cp in tqdm(compound_names): 
            sms = get_cp_sms_from_compound_name(cp, server=self.compound_search_service, server_name=self.compound_search_service_name)  
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

        return custom_compound_reps, invalid_cps


### ================= public API facade ================= ###

    def encode_genes(
            self,
            gene_names: Optional[List[str]] = None,
            uniprot_ids: Optional[List[str]] = None,
            fasta_file: Optional[str] = None,
            save_path: Optional[str] = None
    ) -> Tuple[dict, list]:
        """
        Encode genetic perturbations into UniPert representations.

        Args:
            gene_names: List of gene symbols.
            uniprot_ids: List of UniProt accessions.
            fasta_file: FASTA file containing protein sequences.
            save: Whether to save generated representations.
            save_path: Output path for saved representations.

        Returns:
            Tuple of:
                - encoded representations
                - invalid inputs
        """

        input_count = sum([
            gene_names is not None,
            uniprot_ids is not None,
            fasta_file is not None,
        ])

        if input_count != 1:
            raise ValueError(
                "Exactly one of "
                "`gene_names`, "
                "`uniprot_ids`, or "
                "`fasta_file` "
                "must be provided."
            )

        if gene_names is not None:
            encoded_reps, invalid_inputs = self._encode_genes_from_gene_names(gene_names=gene_names)
        elif uniprot_ids is not None:
            encoded_reps, invalid_inputs = self._encode_genes_from_uniprot_ids(uniprot_accessions=uniprot_ids)
        else:
            encoded_reps, invalid_inputs = self._encode_genes_from_fasta(fasta_file=fasta_file)

        if save_path is not None:
            self._save_reps(reps=encoded_reps,save_path=save_path)

        return encoded_reps, invalid_inputs


    def encode_compounds(
            self,
            compound_names: Optional[List[str]] = None,
            compound_dict: Optional[dict] = None,
            csv_file: Optional[str] = None,
            smiles_list: Optional[List[str]] = None,
            save_path: Optional[str] = None
    ):
        """
        Encode chemical perturbations into UniPert representations.

        Args:
            compound_names: List of compound names.
            compound_dict: Dictionary mapping compound names to SMILES.
            csv_file: CSV file containing compound/SMILES mappings.
            smiles_list: List of SMILES strings.
            save_path: Output path for saved representations.

        Returns:
            Encoded representations and invalid inputs.
        """

        input_count = sum([
            compound_names is not None,
            compound_dict is not None,
            csv_file is not None,
            smiles_list is not None,
        ])

        if input_count != 1:
            raise ValueError(
                "Exactly one compound input source "
                "must be provided."
            )

        if compound_names is not None:
            encoded_reps, invalid_inputs =  self._encode_compounds_from_names(compound_names=compound_names)
        elif compound_dict is not None:
            encoded_reps, invalid_inputs = self._encode_compounds_from_dict(cp_sms_dict=compound_dict)
        elif csv_file is not None:
            return self._encode_compounds_from_csv(csv_file=csv_file)
        else:
            encoded_reps, invalid_inputs = self._encode_compounds_from_smiles(smiles_list=smiles_list)

        if save_path is not None:
            self._save_reps(reps=encoded_reps,save_path=save_path)

        return encoded_reps, invalid_inputs


    def encode_anndata_perturbations(
            self,
            adata: anndata.AnnData,  
            perturbation_columns: List[str] = ['perturbation'],  
            perturbation_types: List[str] = ['genetic'], 
            control_key: str = 'control',
            return_results: bool = False
    )-> dict:
        """
        Generate embeddings for perturbation reagents (perturbagens) based on the given AnnData object.

        Args:
            adata (anndata.AnnData): A required AnnData object containing perturbation information.
            perturbation_columns (list of str): Required list of adata.obs.column names indicating perturbagen info.
            perturbation_types (list of str): An list of strings indicating the type of each perturbation key. Valid types include 'genetic' and 'chemical'.
            control_key (str, optional): The key used to identify control perturbations. Defaults to 'control'.
            return_results (bool, optional): Whether to return the results as a dictionary. Defaults to False.

        Returns:
            dict, optional: A dictionary containing UniPert representations and invalid perturbations if return_results is True.
        """
        # adata = adata.copy()

        valid_types = ['genetic', 'chemical']
        if len(perturbation_columns) != len(perturbation_types):
            raise ValueError("perturbation_columns and perturbation_types must have the same length")
        elif any(t not in valid_types for t in perturbation_types):
            raise ValueError("Supported perturbation types: 'genetic' or 'chemical'")
        
        genetic_ptbgs = []
        chemical_ptbgs = []
        invalid_ptbgs = []

        for col, typ in zip(perturbation_columns, perturbation_types):
            if col not in adata.obs.columns:
                raise ValueError(f"{col} not found in adata.obs")
            ptbgs = adata.obs[col].dropna().unique().tolist()
            ptbgs = [ptbg for ptbg in ptbgs if ptbg != control_key]
            if typ == 'genetic':
                genetic_ptbgs.extend(ptbgs)
            elif typ == 'chemical':
                chemical_ptbgs.extend(ptbgs)    

        unipert_reps = {}
        invalid_ptbgs = []

        # Encode genetic perturbagens
        if genetic_ptbgs:
            genetic_ptbgs = list(set(genetic_ptbgs))
            # logger.info(f"Encoding {len(genetic_ptbgs)} genetic perturbagens...")
            gene_ptbg_reps, invalid_genes = self.encode_genes(gene_names=genetic_ptbgs)
            unipert_reps.update(gene_ptbg_reps)
            invalid_ptbgs.extend(invalid_genes)

        # Encode chemical perturbagens
        if chemical_ptbgs:
            chemical_ptbgs = list(set(chemical_ptbgs))
            # logger.info(f"Encoding {len(chemical_ptbgs)} chemical perturbagens...")
            chem_ptbg_reps, invalid_compounds = self.encode_compounds(compound_names=chemical_ptbgs)
            unipert_reps.update(chem_ptbg_reps)
            invalid_ptbgs.extend(invalid_compounds)

        # Save UniPert embeddings to adata.uns['UniPert_reps']
        adata.uns['UniPert_reps'] = unipert_reps
        adata.uns['invalid_ptbgs'] = invalid_ptbgs

        logger.success(colors.green(f'UniPert representations generated!'))
        logger.info(f"{len(unipert_reps)} perturbagens' UniPert representations saved to adata.uns['UniPert_reps']")
        
        if invalid_ptbgs:
            logger.warning(f"{len(invalid_ptbgs)} perturbagens can not be repersentated and saved to adata.uns['invalid_ptbgs']: \n{invalid_ptbgs}")

        if return_results:
            return {'UniPert_reps': unipert_reps, 'invalid_ptbgs': invalid_ptbgs}
