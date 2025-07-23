# -*- coding: utf-8 -*-

import gc
import torch as th
import random
import time
from DeepMCL_model_davis import DTISAGE
from torch.optim.lr_scheduler import ExponentialLR
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, \
    average_precision_score, f1_score, auc, recall_score, precision_score
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

import dgl
from focalloss import FocalLoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_roce(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x:x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce

def show_result(DATASET, lable, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def test_precess(model, pbar, LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            sample_data, labels = data
            
            drug_graph = sample_data['drug_graph'].to(device)
            drug_cnn = sample_data['drug_cnn'].to(device)  
            protein_protbert = sample_data['protein_protbert'].to(device)
            protein_convlstm = sample_data['protein_convlstm'].to(device)
            labels = labels.to(device)

            predicted_scores = model(drug_graph, drug_cnn, protein_protbert, protein_convlstm)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, test_loss, Accuracy, Precision, Reacll, AUC, PRC


def test_model(dataset_load, save_path, DATASET, LOSS, dataset="Train", lable="best", save=True):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    T, P, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = \
        test_precess(model, test_pbar, LOSS)
    if save:
        with open(save_path + "/{}_{}_{}_prediction.txt".format(DATASET, dataset, lable), 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
    results = '{}_set--Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(lable, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test


def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


# New dataset loading class
class DTIDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample, sample['label']


def collate_fn(batch):
    """Batch processing function"""
    samples, labels = zip(*batch)
    
    # Extract channel data
    drug_graphs = [s['drug_graph'] for s in samples]
    drug_cnns = torch.stack([s['drug_cnn'] for s in samples])
    protein_protberts = torch.stack([s['protein_protbert'] for s in samples])
    protein_convlstms = torch.stack([s['protein_convlstm'] for s in samples])
    labels = torch.stack(list(labels))
    
    # Batch DGL graphs
    batched_graphs = dgl.batch(drug_graphs)
    
    return {
        'drug_graph': batched_graphs,
        'drug_cnn': drug_cnns,
        'protein_protbert': protein_protberts,
        'protein_convlstm': protein_convlstms
    }, labels


import os
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    """select seed"""
    SEED = random.randint(1, 1234)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Load new format data
    print("Loading new format DTI data...")
    with open("human_train_deepmcl.pkl", 'rb') as fp:
        train_data = pickle.load(fp)
    with open("human_valid_deepmcl.pkl", 'rb') as fp:
        valid_data = pickle.load(fp)
    with open("human_test_deepmcl.pkl", 'rb') as fp:
        test_data = pickle.load(fp)

    all_data = train_data + valid_data + test_data
    dataset = shuffle_dataset(all_data, SEED)
    
    print(f"Total data size: {len(dataset)}")
    print(f"Data format example: {list(dataset[0].keys())}")

    K_Fold = 5
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        
        # Create dataset and data loader
        train_dataset_obj = DTIDataset(train_dataset)
        test_dataset_obj = DTIDataset(test_dataset)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset_obj, 
            batch_size=32, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset_obj, 
            batch_size=32, 
            shuffle=False, 
            collate_fn=collate_fn
        )

        model = DTISAGE()
        model.to(device)
        MODEL_NAME = f"model-{int(time.time())}"
        optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
        criterion = FocalLoss()
        scheduler = ExponentialLR(optimizer, gamma=0.90)

        print("Model initialization completed")

        def fwd_pass(data_loader, train=False):
            if train:
                model.train()
            else:
                model.eval()
                
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in data_loader:
                if train:
                    optimizer.zero_grad()
                
                # Move to device
                drug_graph = batch_data['drug_graph'].to(device)
                drug_cnn = batch_data['drug_cnn'].to(device)
                protein_protbert = batch_data['protein_protbert'].to(device)
                protein_convlstm = batch_data['protein_convlstm'].to(device)
                batch_labels = batch_labels.to(device)
                
                # Forward pass
                outputs = model(drug_graph, drug_cnn, protein_protbert, protein_convlstm)
                loss = criterion(outputs, batch_labels.float())
                
                if train:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                correct += (predicted == batch_labels.argmax(dim=1).unsqueeze(1)).sum().item()
                total += batch_labels.size(0)
            
            accuracy = correct / total
            avg_loss = total_loss / len(data_loader)
            return accuracy, avg_loss

        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            train_acc, train_loss = fwd_pass(train_loader, train=True)
            
            if epoch % 10 == 0:
                test_acc, test_loss = fwd_pass(test_loader, train=False)
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            
            scheduler.step()

        print(f"Fold {i_fold + 1} training completed") 