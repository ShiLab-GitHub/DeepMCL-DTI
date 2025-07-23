from dgl.nn.pytorch.conv import GATConv, GraphConv, TAGConv, GINConv, APPNPConv, SAGEConv
from module import Conv1dLSTM
from module import Conv1dLSTM
from collections import defaultdict
from dgl.nn.pytorch.glob import MaxPooling, GlobalAttentionPooling
import torch.nn as nn
import torch.nn.functional as F
import random
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from rdkit import Chem
import deepchem
import torch
from transformers import BertModel, BertTokenizer
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from config import DefaultConfig

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
configs = DefaultConfig()
downloadFolderPath = './inputs/ProtBert_model/'
modelFolderPath = downloadFolderPath
modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
configFilePath = os.path.join(modelFolderPath, 'config.json')
vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False)
model = BertModel.from_pretrained(modelFolderPath)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:0')
model = model.to(device)
model = model.eval()

random.seed(0)

pk = deepchem.dock.ConvexHullPocketFinder()


node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64
def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25, "J": 26}

CHARPROTLEN = 26

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))

class DTISAGE(nn.Module):

    def __init__(self):
        super(DTISAGE, self).__init__()

        self.protein_graph_conv = nn.ModuleList()
        for i in range(5):
            self.protein_graph_conv.append(SAGEConv(31, 31, 'mean'))

        self.ligand_graph_conv = nn.ModuleList()
        self.ligand_graph_conv.append(SAGEConv(74, 60, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(60, 46, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(46, 32, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(32, 32, 'mean'))
        self.ligand_graph_conv.append(SAGEConv(32, 64, 'mean'))  # As required in the paper: 5-layer SAGEConv with 64-dimensional output

        self.pooling_ligand = nn.Linear(64, 1)  # Matches GraphSAGE output dimension
        self.pooling_protein = nn.Linear(31, 1)

        self.embed_smile = nn.Embedding(100, 100)
        self.embed_seq = nn.Embedding(1000, 1000)

        self.embed_fingerprint = nn.Embedding(len(fingerprint_dict), 100)

        self.W_gnn = nn.ModuleList([nn.Linear(100, 100)
                                    for _ in range(3)])
        self.W_rnn = Conv1dLSTM(in_channels=100,
                                                  out_channels=100,
                                                  kernel_size=3, num_layers=1, bidirectional=True,
                                                  dropout=0.4,
                                                  batch_first=True)
        self.W_rnn2 = Conv1dLSTM(in_channels=1000,
                                out_channels=31*32,
                                kernel_size=3, num_layers=1, bidirectional=True,
                                dropout=0.4,
                                batch_first=True)

        self.embedding_xt = nn.Embedding(25 + 1, 31)
        self.fc_xt1 = nn.Linear(32 * 121, 128)


        self.d = nn.Linear(100, 31)
        self.b = nn.Linear(1000, 31*32)

        self.bert1 = nn.Linear(1024, 32)
        self.bert2 = nn.Linear(500, 31)

        self.fc1 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(768, 512)
        self.out = nn.Linear(512, 128)


        self.dropout = 0.2

        self.bilstm = nn.LSTM(31, 31, num_layers=1, bidirectional=True, dropout=self.dropout)


        self.fc_in = nn.Linear(64, 2048)  # Original dimension, will be dynamically adjusted based on actual feature dimensions
        self.fc_out = nn.Linear(2048, 1)
        
        # Processing the feature dimension after interact-attention
        self.fc_interact = nn.Linear(192, 64)  # Maps 192 dimensions (128+64) to 64 dimensions

        self.W_s1 = nn.Linear(35, 45)
        self.W_s2 = nn.Linear(45, 30)



        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1024,
                      out_channels=32,
                      kernel_size=7, stride=1,
                      padding=7 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'),
            nn.LeakyReLU(negative_slope=.01),
            nn.BatchNorm1d(num_features=32,
                           eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(.5)
        )

        self.beta = 0.2
        self.dim = 64
        self.conv = 40
        self.drug_MAX_LENGH = 100
        self.drug_kernel = [4, 6, 8]
        self.protein_MAX_LENGH = 1000
        self.protein_kernel = [4, 8, 12]

        self.protein_embed = nn.Embedding(26, self.dim, padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=64, kernel_size=self.drug_kernel[2]),  # Changed to 64 dimensions to match GraphSAGE
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(86)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(62)
        self.attention_layer = nn.Linear(64, 64)  # Using a unified 64 dimensions
        self.protein_attention_layer = nn.Linear(32, 64)  # Output 64 dimensions to match drug_attention
        self.drug_attention_layer = nn.Linear(64, 64)  # Maintain 64 dimensions to match drug features
        
        # Interact-Attention module
        self.interact_W_drug = nn.Linear(128, 64)  # Drug feature MLP
        self.interact_W_protein = nn.Linear(128, 64)  # Protein feature MLP
        self.interact_attention_matrix = nn.Linear(64, 1)  # Attention matrix calculation
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.jiangwei = nn.Linear(160, 31)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2366, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)
    def attention_net(self, lstm_output):
       attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
       attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
       attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

       return attn_weight_matrix
    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
    def add_dense_block(self, block, in_channels):
        layer=[]
        layer.append(block(in_channels))
        D_seq=nn.Sequential(*layer)
        return D_seq
    def _make_transition_layer(self, layer, in_channels, out_channels):
            modules1 = []
            modules1.append(layer(in_channels, out_channels))
            return nn.Sequential(*modules1)


    def rnn(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
    def rnn2(self, xs):
        xs = torch.unsqueeze(xs, 0)
        xs = torch.unsqueeze(xs, 0)
        xs, h = self.W_rnn2(xs)
        xs = torch.relu(xs)
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)


    def forward(self, drug_graph=None, drug_cnn=None, protein_protbert=None, protein_convlstm=None, g=None):


        if drug_graph is not None:
            drugGraph = drug_graph.ndata['h']
        else:
            raise ValueError("The drug_graph parameter must be provided")

        for module in self.ligand_graph_conv:
            if g is not None:
                drugGraph = F.relu(module(g[0], drugGraph))
            else:
                drugGraph = F.relu(module(drug_graph, drugGraph))

        pool_ligand = GlobalAttentionPooling(self.pooling_ligand)
        if g is not None:
            ligand_rep = pool_ligand(g[0], drugGraph).view(-1, 64)  # Updated to 64 dimensions
        else:
            ligand_rep = pool_ligand(drug_graph, drugGraph).view(-1, 64)  # Updated to 64 dimensions
        ligand_rep_graph = torch.Tensor(ligand_rep)

        """Protein vector with bert."""


        batch_size = protein_protbert.size(0)
        bert_results = []
        
        for i in range(batch_size):
            # Get ProtBert features of a single sample [1, 1024]
            single_protbert = protein_protbert[i]  # [1, 1024]

            expanded_features = single_protbert.repeat(1024, 1)  # [1024, 1024]
            
            # Use adaptive pooling to adjust to [1024, 500]
            adaptive_pool = nn.AdaptiveAvgPool2d((1024, 500))
            features_pooled = adaptive_pool(expanded_features.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            
            # Through bert2: [1024, 500] -> [1024, 31]
            bert_feat = self.bert2(features_pooled)
            # permute: [1024, 31] -> [31, 1024]  
            bert_feat = bert_feat.permute(1, 0)
            # Through bert1: [31, 1024] -> [31, 32]
            bert_feat = self.bert1(bert_feat)
            # Permute again to get the correct shape: [31, 32] -> [32, 31]
            bert_feat = bert_feat.permute(1, 0)
            # Final shape: [1, 32, 31]
            final_bert = bert_feat.unsqueeze(0)
            bert_results.append(final_bert)
            
            protein_rep_bert = torch.cat(bert_results, dim=0)
        # protein_rep_bert = self
        # features = self.conv_encoder(protein_rep_bert)
        #
        # protein_rep_bert = protein_rep_bert.view(32,31)

        """protein vector with convLSTM."""
        
        batch_size = protein_convlstm.size(0)
        convlstm_results = []
        
        # Process each sample in the batch individually 
        for i in range(batch_size):
            seq_vectors = protein_convlstm[i]  # [1000, 1000] - 单个样本
            rnn_output = self.rnn2(seq_vectors)  # rnn2返回的原始形状
            b_output = self.b(rnn_output)  # 通过线性层 [1000] -> [31*32=992]
            # reshape为最终形状 [1, 32, 31]
            final_output = b_output.view(1, 32, 31)
            convlstm_results.append(final_output)
            
            # 合并批量结果
            protein_rep_ConvLSTM = torch.cat(convlstm_results, dim=0)  # [batch_size, 32, 31]

        '''CNN'''
        
        
            # 新格式：直接使用预处理的CNN特征
            # drug_cnn 的形状: [batch_size, 100, 100] - 已经是嵌入矩阵
        batch_size = drug_cnn.size(0)
        
        # 将drug_cnn从[batch_size, 100, 100]转换为CNN期望的[batch_size, 64, 100]
        drugembed = F.adaptive_avg_pool2d(drug_cnn.unsqueeze(1), (64, 100)).squeeze(1)
        drugConv = self.Drug_CNNs(drugembed)
        if g is not None:
            drug2d = ligand_rep_graph.view(1, 64, 1)  # GraphSAGE输出64维
            protbert = protein_rep_bert.view(1, 32, 31)
        else:
            batch_size = protein_rep_bert.size(0)
            drug2d = ligand_rep_graph.view(batch_size, 64, 1)  # GraphSAGE输出64维
            # protein_rep_bert 已经是正确的 [batch_size, 32, 31] 形状，无需再view
            protbert = protein_rep_bert
        
        drug_graphsage_feat = drug2d.squeeze(-1)  # [batch_size, 64]
        drug_cnn_feat = drugConv.mean(dim=-1)     # [batch_size, 64] - Update: CNN now outputs 64 dimensions
        drug_feature = torch.cat([drug_graphsage_feat, drug_cnn_feat], dim=1)  # [batch_size, 128] - Update: 64+64=128
        
        protein_bert_feat = protbert.mean(dim=-1)  # [batch_size, 32]
        protein_convlstm_feat = protein_rep_ConvLSTM.mean(dim=-1)  # [batch_size, 32]
        protein_feature = torch.cat([protein_bert_feat, protein_convlstm_feat], dim=1)  # [batch_size, 64]
        
        drugAll = torch.cat((drug2d, drugConv),dim=2)
        print(f"DEBUG: protbert shape before concat: {protbert.shape}")
        print(f"DEBUG: protein_rep_ConvLSTM shape before concat: {protein_rep_ConvLSTM.shape}")
        protAll = torch.cat((protbert, protein_rep_ConvLSTM),dim=2)

#############################################################################################
        # Interact-Attention mechanism 

        if drug_feature.size(1) != 128:
            drug_feature_padded = F.pad(drug_feature, (0, 128 - drug_feature.size(1)))
        else:
            drug_feature_padded = drug_feature
            
        if protein_feature.size(1) != 128:
            protein_feature_padded = F.pad(protein_feature, (0, 128 - protein_feature.size(1)))
        else:
            protein_feature_padded = protein_feature
        
        drug_att_vec = self.relu(self.interact_W_drug(drug_feature_padded))  # [1, 64]
        protein_att_vec = self.relu(self.interact_W_protein(protein_feature_padded))  # [1, 64]
        
        interaction = drug_att_vec * protein_att_vec 
        attention_weight = self.sigmoid(self.interact_attention_matrix(interaction))  # [1, 1]
        
        updated_drug_feature = drug_feature * self.beta + drug_feature * attention_weight
        updated_protein_feature = protein_feature * self.beta + protein_feature * attention_weight
        

        drug_att = self.drug_attention_layer(drugAll.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(protAll.permute(0, 2, 1))

        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, protAll.shape[-1], 1)
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, drugAll.shape[-1], 1, 1)
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))
        
        Protein_atte_resized = F.adaptive_avg_pool2d(Protein_atte.unsqueeze(1), (32, 62)).squeeze(1)  # [batch_size, 32, 62]

        drugAll = drugAll * self.beta + drugAll * Compound_atte
        protAll = protAll * self.beta + protAll * Protein_atte_resized

        drugAll = self.Drug_max_pool(drugAll).squeeze(2)
        protAll = self.Protein_max_pool(protAll).squeeze(2)

        final_pair = torch.cat((updated_drug_feature, updated_protein_feature), dim=1)

        # drug(128) + protein(64) = 192
        pair = self.fc_interact(final_pair)  # [batch_size, 192] -> [batch_size, 64]

        out = F.relu(self.fc_in(pair.view(-1, 64)))  # [batch_size, 64] -> [batch_size, 2048]
        out = self.fc_out(out)  # [batch_size, 2048] -> [batch_size, 1]
        
        out_2class = torch.cat([1-torch.sigmoid(out), torch.sigmoid(out)], dim=1)  # [batch_size, 2]
        out = out_2class
        return out

