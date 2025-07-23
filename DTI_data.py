import os
import pickle
import json
from collections import OrderedDict
import random
import glob
import warnings
from time import time

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# RDKit and molecule processing
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import rdkit

# DGL and graph processing
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
import networkx as nx

# Bio processing
from Bio.PDB import *
from Bio import SeqUtils
from Bio.pairwise2 import format_alignment
from Bio import pairwise2
import Bio

# Deep learning models
from transformers import BertModel, BertTokenizer
import deepchem

# Suppress warnings
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class DTIDataProcessor:
    """
    DeepMCL-DTI data processor, implements data preprocessing for a four-channel architecture
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize ProtBert model
        self._init_protbert()
        
        # Initialize subgraph feature extractor
        self.node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        
        # Atom type dictionary
        self.atom_type = {
            'C': 1, 'H': 2, 'O': 3, 'N': 4, 'S': 5, 'Li': 6, 'Mg': 7, 'F': 8, 
            'K': 9, 'Al': 10, 'Cl': 11, 'Au': 12, 'Ca': 13, 'Hg': 14, 'Na': 15, 
            'P': 16, 'Ti': 17, 'Br': 18
        }
        self.NOTINDICT = 19
        
        # SMILES character to index mapping
        self.smiles_chars = self._get_smiles_chars()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.smiles_chars)}
        
        print(f"Initialized DTI data processor, using device: {self.device}")
    
    def _init_protbert(self):
        """Initialize ProtBert model"""
        try:
            model_path = './inputs/ProtBert_model/'
            vocab_path = os.path.join(model_path, 'vocab.txt')
            
            self.tokenizer = BertTokenizer(vocab_path, do_lower_case=False)
            self.protbert_model = BertModel.from_pretrained(model_path)
            self.protbert_model = self.protbert_model.to(self.device)
            self.protbert_model.eval()
            
            print("ProtBert model loaded successfully")
        except Exception as e:
            print(f"ProtBert model loading failed: {e}")
            self.tokenizer = None
            self.protbert_model = None
    
    def _get_smiles_chars(self):
        """Get SMILES character set"""
        chars = [
            'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'H',
            '(', ')', '[', ']', '=', '#', '-', '+', '\\', '/',
            '@', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            'c', 'n', 'o', 's', 'p', 'f', 'l', 'r', 'i', 'h', ' '
        ]
        return chars
    
    def process_smiles_for_cnn(self, smiles, max_length=100):
        """
        Process SMILES string for CNN channel
        Args:
            smiles (str): SMILES string
            max_length (int): Maximum length, default 100
        Returns:
            torch.Tensor: 100x100 embedding matrix
        """
        # Convert SMILES to character index sequence
        char_indices = []
        for char in smiles[:max_length]:
            if char in self.char_to_idx:
                char_indices.append(self.char_to_idx[char])
            else:
                char_indices.append(self.char_to_idx[' '])  # Unknown characters replaced with space
        
        # Adaptively pool to specified length
        if len(char_indices) < max_length:
            # Pad
            char_indices.extend([self.char_to_idx[' ']] * (max_length - len(char_indices)))
        elif len(char_indices) > max_length:
            # Truncate
            char_indices = char_indices[:max_length]
        
        # Create 100x100 embedding matrix
        embedding_matrix = torch.zeros(max_length, max_length)
        vocab_size = len(self.smiles_chars)
        
        for i, char_idx in enumerate(char_indices):
            # Create one-hot encoding for each character
            if i < max_length:
                embedding_matrix[i, :min(vocab_size, max_length)] = F.one_hot(
                    torch.tensor(char_idx), num_classes=vocab_size
                ).float()[:max_length]
        
        return embedding_matrix
    
    def process_smiles_for_graphsage(self, smiles):
        """
        Process SMILES string for GraphSAGE channel
        Args:
            smiles (str): SMILES string
        Returns:
            dgl.DGLGraph: Subgraph
        """
        try:
            # Create subgraph using DGLLife
            graph = smiles_to_bigraph(smiles, node_featurizer=self.node_featurizer)
            graph = dgl.add_self_loop(graph)
            return graph
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def process_protein_for_protbert(self, sequence, max_length=1024):
        """
        Process protein sequence for ProtBert channel
        Args:
            sequence (str): Amino acid sequence
            max_length (int): Maximum length
        Returns:
            torch.Tensor: 1x1024 embedding vector
        """
        if self.protbert_model is None:
            print("ProtBert model not loaded")
            return torch.zeros(1, max_length)
        
        try:
            # Add spaces between amino acids
            spaced_sequence = " ".join(sequence)
            
            # Encode sequence
            encoded = self.tokenizer.batch_encode_plus(
                [spaced_sequence],
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get ProtBert embedding
            with torch.no_grad():
                outputs = self.protbert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                embeddings = outputs.last_hidden_state
            
            # Get sequence-level representation (excluding special tokens)
            seq_len = attention_mask.sum().item()
            if seq_len > 2:  # At least have [CLS] and [SEP]
                sequence_embedding = embeddings[0, 1:seq_len-1].mean(dim=0)
            else:
                sequence_embedding = embeddings[0, 0]  # Use only [CLS]
            
            return sequence_embedding.cpu().unsqueeze(0)  # 1x1024
            
        except Exception as e:
            print(f"Error processing sequence with ProtBert: {e}")
            return torch.zeros(1, max_length)
    
    def process_protein_for_convlstm(self, sequence, max_length=1000):
        """
        Process protein sequence for Bi-ConvLSTM channel
        Args:
            sequence (str): Amino acid sequence
            max_length (int): Maximum length, default 1000
        Returns:
            torch.Tensor: 1000x1000 embedding matrix
        """
        # Amino acid to index mapping
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}
        aa_to_idx['X'] = len(amino_acids)  # Unknown amino acid
        
        # Convert sequence to indices
        sequence_indices = []
        for aa in sequence[:max_length]:
            if aa in aa_to_idx:
                sequence_indices.append(aa_to_idx[aa])
        else:
                sequence_indices.append(aa_to_idx['X'])
        
        # Adaptively pool to specified length
        if len(sequence_indices) < max_length:
            # Pad
            sequence_indices.extend([aa_to_idx['X']] * (max_length - len(sequence_indices)))
        elif len(sequence_indices) > max_length:
            # Truncate
            sequence_indices = sequence_indices[:max_length]
        
        # Create 1000x1000 embedding matrix
        embedding_matrix = torch.zeros(max_length, max_length)
        vocab_size = len(aa_to_idx)
        
        for i, aa_idx in enumerate(sequence_indices):
            if i < max_length:
                # Create one-hot encoding for each amino acid
                one_hot = F.one_hot(torch.tensor(aa_idx), num_classes=vocab_size).float()
                embedding_matrix[i, :min(vocab_size, max_length)] = one_hot[:max_length]
        
        return embedding_matrix
    
    def create_drug_3d_graph(self, smiles):
        """
        Create 3D molecular graph for drug
        Args:
            smiles (str): SMILES string
        Returns:
            tuple: (positions, atomic_numbers) or (None, None)
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            
            if mol.GetNumAtoms() == 1:
                return None, None
            
            # Add hydrogen atoms
            mol = Chem.AddHs(mol)
            
            # Generate 3D structure
            try_count = 0
            while AllChem.EmbedMolecule(mol) == -1:
                try_count += 1
                if try_count >= 10:
                    return None, None
            
            # Optimize structure
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            
            # Get atom positions
            conformer = mol.GetConformer(id=0)
            positions = []
            atomic_numbers = []
            
            for i in range(mol.GetNumAtoms()):
                pos = list(conformer.GetAtomPosition(i))
                positions.append(pos)
                
                atom = mol.GetAtoms()[i]
                if atom.GetSymbol() in self.atom_type:
                    atomic_numbers.append(self.atom_type[atom.GetSymbol()])
                else:
                    atomic_numbers.append(self.NOTINDICT)
            
            return torch.tensor(positions), torch.tensor(atomic_numbers)
            
        except Exception as e:
            print(f"Error processing 3D molecular structure: {e}")
            return None, None
    
    def process_single_sample(self, smiles, sequence, label):
        """
        Process single sample, generate input data for four channels
        Args:
            smiles (str): SMILES string
            sequence (str): Protein sequence
            label (int): Label (0 or 1)
        Returns:
            dict: Dictionary containing data for four channels
        """
        sample_data = {}
        
        # Drug channel 1: GraphSAGE
        drug_graph = self.process_smiles_for_graphsage(smiles)
        if drug_graph is None:
            return None
        sample_data['drug_graph'] = drug_graph
        
        # Drug channel 2: CNN
        drug_cnn_input = self.process_smiles_for_cnn(smiles)
        sample_data['drug_cnn'] = drug_cnn_input
        
        # Protein channel 1: ProtBert
        protein_protbert = self.process_protein_for_protbert(sequence)
        sample_data['protein_protbert'] = protein_protbert
        
        # Protein channel 2: Bi-ConvLSTM
        protein_convlstm = self.process_protein_for_convlstm(sequence)
        sample_data['protein_convlstm'] = protein_convlstm
        
        # Label
        sample_data['label'] = torch.tensor([1, 0] if label == 1 else [0, 1], dtype=torch.float32)
        
        return sample_data
    
    def load_dataset(self, data_file, seq_pdb_file):
        """
        Load dataset
        Args:
            data_file (str): Data file path
            seq_pdb_file (str): Sequence-PDB mapping file path
        Returns:
            tuple: (train_data, valid_data, test_data)
        """
        print(f"Loading dataset: {data_file}")
        
        # Read data
        with open(data_file, 'r') as f:
            raw_data = f.read().strip().split('\n')
        
        # Read sequence-PDB mapping
        df = pd.read_csv(seq_pdb_file)
        
        # Shuffle data randomly
        random.shuffle(raw_data)
        
        # Split dataset
        total_size = len(raw_data)
        train_size = int(total_size * 0.8)
        valid_size = int(total_size * 0.9)
        
        train_raw = raw_data[:train_size]
        valid_raw = raw_data[train_size:valid_size]
        test_raw = raw_data[valid_size:]
        
        print(f"Dataset sizes - Train: {len(train_raw)}, Validation: {len(valid_raw)}, Test: {len(test_raw)}")
        
        return train_raw, valid_raw, test_raw
    
    def process_dataset(self, raw_data, dataset_name):
        """
        Process dataset
        Args:
            raw_data (list): Raw data list
            dataset_name (str): Dataset name
        Returns:
            list: Processed data
        """
        processed_data = []
        
        print(f"Starting to process {dataset_name} dataset...")
        
        for i, item in enumerate(raw_data):
            if i % 100 == 0:
                print(f"Processing progress: {i}/{len(raw_data)}")
            
            try:
                parts = item.split()
                if len(parts) < 3:
                    continue
                
                smiles = parts[0]
                sequence = parts[1]
                label = int(parts[2])
                
                # Process single sample
                sample_data = self.process_single_sample(smiles, sequence, label)
                if sample_data is not None:
                    processed_data.append(sample_data)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"{dataset_name} dataset processing complete, valid samples: {len(processed_data)}")
        return processed_data
    
    def save_processed_data(self, data, filename):
        """Save processed data"""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to: {filename}")


class DefaultConfig:
    """Default configuration class"""
    def __init__(self):
        self.max_sequence_length = 1000
        self.max_smiles_length = 100
        self.protbert_max_length = 1024


def main():
    """Main function"""
    # Initialize configuration
    config = DefaultConfig()
    
    # Initialize data processor
    processor = DTIDataProcessor(config)
    
    # Set file paths
    data_file = "data.txt"
    seq_pdb_file = "pdball.txt"
    
    try:
        # Load dataset
        train_raw, valid_raw, test_raw = processor.load_dataset(data_file, seq_pdb_file)
        
        # Process training set
        train_data = processor.process_dataset(train_raw, "训练")
        processor.save_processed_data(train_data, "human_train_deepmcl.pkl")
        
        # Process validation set
        valid_data = processor.process_dataset(valid_raw, "验证")
        processor.save_processed_data(valid_data, "human_valid_deepmcl.pkl")
        
        # Process test set
        test_data = processor.process_dataset(test_raw, "测试")
        processor.save_processed_data(test_data, "human_test_deepmcl.pkl")
        
        print("All data processing complete!")

    except Exception as e:
        print(f"Error during data processing: {e}")


if __name__ == "__main__":
    main()