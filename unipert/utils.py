import os
import warnings
from lamin_utils import logger
from getSequence import getseq
from rdkit import Chem


class FastaBatchedDataset(object):
    """
    ref: https://github.com/facebookresearch/esm/blob/main/esm/data.py
    """
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        # assert len(set(sequence_labels)) == len(
        #     sequence_labels
        # ), "Found duplicate sequence labels"

        # Remove duplicated sequence labels while preserving order
        unique_sequences = {}
        for label, seq in zip(sequence_labels, sequence_strs):
            if label not in unique_sequences:
                unique_sequences[label] = seq
            if label in unique_sequences:
                if unique_sequences[label] != seq:
                    logger.warning(f"Duplicate entry with different sequences: {label}")

        sequence_labels = list(unique_sequences.keys())
        sequence_strs = list(unique_sequences.values())

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]
    

def check_smiles(smiles):
    """
    Check if a SMILES string is valid.

    Parameters:
    smiles (str): The SMILES string to check.
    
    Returns:
    bool: Whether the SMILES string is valid.
    """
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
    except:
        logger.print(f'Invalid SMILES: {smiles}')
        # print(f'Invalid SMILES: {smiles}')
        return False

    if m is None:
        logger.print(f'Invalid SMILES: {smiles}')
        # print(f'Invalid SMILES: {smiles}')
        return False
    else:
        try:
            Chem.SanitizeMol(m)
        except:
            logger.print(f'Invalid chemistry: {smiles}')
            # print(f'Invalid chemistry: {smiles}')
            return False
    return True


def get_tgt_seq_from_gene_name(gene_name, organism_id='9606'):
    """
    Retrieve the best match UniProt ID for a given gene name using the UniProt API.
    
    Parameters:
    gene_name (str): The gene name to query.
    organism_id (str): The species organism_id to search for. Default is 9606.
    
    Returns:
    str: The best match UniProt ID. Returns None if no match is found.
    """
    try:
        proid_seq = getseq(organism_id+'+'+gene_name, ignore_failures=False)
        uniprot_id = proid_seq[0].split('|')[1]
        head = proid_seq[0].replace(uniprot_id, uniprot_id.upper())
        seq = proid_seq[1]
        return [uniprot_id.upper(), head, seq]
    except:
        logger.warning(f"Unable to retrieve protein sequence for query gene name: {gene_name}")
        return None
    

def get_tgt_seq_from_uniprot_accession(query_ua):
    """
    Retrieve the best match UniProt ID for a given uniprot accession using the UniProt API.
    
    Parameters:
    uniprot_accession (str): The uniprot_accession to query.
    organism_id (str): The species organism_id to search for. Default is 9606.
    
    Returns:
    str: The best match UniProt ID. Returns None if no match is found.
    """
    try:
        proid_seq = getseq(query_ua, uniprot_id=True, ignore_failures=False)
        uniprot_id = proid_seq[0].split('|')[1]
        head = proid_seq[0].replace(uniprot_id, uniprot_id.upper())[1:]
        seq = proid_seq[1]
        return [uniprot_id.upper(), head, seq]
    except:
        logger.warning(f"Unable to retrieve protein sequence for query uniprot accession: {query_ua}")
        return None


def set_chemspider_key(key):
    os.environ['CHEMSPIDER_APIKEY'] = key


def check_chemspipy_service():
    try:
        from chemspipy import ChemSpider
    except ImportError:
        warnings.warn(
            "chemspipy is not installed. "
            "ChemSpider-based compound queries will be unavailable."
        )
        return None

    api_key = os.environ.get("CHEMSPIDER_APIKEY")
    if not api_key:
        warnings.warn(
            "CHEMSPIDER_APIKEY is not set. "
            "Please visit "
            "https://chemspipy.readthedocs.io/en/stable/guide/intro.html#obtaining-an-api-key",
            UserWarning,
        )
        return None

    try:
        cs = ChemSpider(api_key)
        # lightweight connectivity test
        _ = cs.search("glucose")
        return cs
    except Exception as e:
        warnings.warn(
            f"ChemSpider service connection failed: {e}",
            UserWarning,
        )
        return None
    

def check_pubchempy_service():
    try:
        import pubchempy as pcp
        _ = pcp.get_compounds('glucose', 'name')
        return pcp
    except Exception as e:
        warnings.warn(
            f"PubChem service connection failed: {e}",
            UserWarning,
        )
        return None


def get_cp_sms_from_compound_name(compound_name, server, server_name='pubchem'):
    """
    Retrieve compound SMILES from compound name using either ChemSpider or PubChemPy API.
    
    Parameters:
    compound_name (str): The compound name to query.
    service (str): The service to use. Can be either 'chemspider' or 'pubchem'. Default is 'pubchem'.
    
    Returns:
    str: The compound SMILES. Returns None if no match is found.
    """
    if server_name=='pubchem':
        try:
            compound = server.get_compounds(compound_name, 'name')[0]
            # is_valid = check_smiles(compound.isomeric_smiles)
            # return compound.isomeric_smiles if is_valid else None
            is_valid = check_smiles(compound.smiles)
            return compound.smiles if is_valid else None
        except:
            logger.print(f"Unable to retrieve SMILES for query compound name: {compound_name}")
            return None 
    elif server_name=='chemspider':
        try:
            compound = server.search(compound_name)[0]
            is_valid = check_smiles(compound.smiles)
            return compound.smiles if is_valid else None
        except:
            logger.print(f"Unable to retrieve SMILES for query compound name: {compound_name}")
            return None 
    else:
        raise ValueError('Compound service should be either pubchem or chemspider')


