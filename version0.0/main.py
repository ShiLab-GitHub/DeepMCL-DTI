# -*- coding: utf-8 -*-
"""
@Time:Created on 2022/11/05
@author: Yijie Guo
"""
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
            compounds, proteins, labels = data
            compounds = compounds.to(device)
            proteins = proteins.to(device)
            labels = labels.to(device)

            predicted_scores = model(compounds, proteins)
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


import os
import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')

if __name__ == "__main__":
    """select seed"""
    # SEED = 1234
    SEED = random.randint(1, 1234)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    # with open("DrugBank.txt", 'r') as fp:
    #      train_raw = fp.read().strip().split('\n')
    with open("Davis_part_train.pkl", 'rb') as fp:
        ds = pickle.load(fp)
    with open("Davis_part_test.pkl", 'rb') as fp:
        ds_test = pickle.load(fp)
    with open("Davis_part_val.pkl", 'rb') as fp:
        ds_val = pickle.load(fp)

    raw = ds+ds_test+ds_val
    # raw = ds_val

    dataset = shuffle_dataset(raw, SEED)
    K_Fold = 5
    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        X = [i[0] for i in train_dataset]
        y = [i[1][0] for i in train_dataset]
        p = [int(i[1][0]) for i in train_dataset]
        X_test = [i[0] for i in test_dataset]
        y_test = [i[1][0] for i in test_dataset]

        model = DTISAGE()
        model.to(device)
        MODEL_NAME = f"model-{int(time.time())}"
        optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
        # criterion = th.nn.BCELoss()
        criterion = FocalLoss()
        scheduler = ExponentialLR(optimizer, gamma=0.90)

        print("init done")

        def fwd_pass(X, y, train=False):
            if train:
                model.zero_grad()
            out = []

            for item in X:
                x = [0, 0, 0]
                x[0] = item[0].to(device)
                x[1] = item[1]
                x[2] = item[2]
                out.append(model(x))
                del x
            out = th.stack(out, 0).view(-1, 1).to(device)

            y = th.Tensor(y).view(-1, 1).to(device)

            loss = criterion(out, y)
            matches = [th.round(i) == th.round(j) for i, j in zip(out, y)]
            acc = matches.count(True) / len(matches)

            if train:
                loss.backward()
                optimizer.step()

            return acc, loss, out


        def test_func(model_f, y_label, X_test_f):
            y_pred = []
            y_label = th.Tensor(y_label)
            print("Testing:")
            print("-------------------")
            with tqdm(range(0, len(X_test_f), 1)) as tepoch:
                for i in tepoch:
                    with th.no_grad():
                        x = [0, 0, 0]
                        x[0] = X_test_f[i][0].to(device)
                        x[1] = X_test_f[i][1]
                        x[2] = X_test_f[i][2]
                        y_pred.append(model_f(x).cpu())

            y_pred = th.cat(y_pred, dim=0)
            y_pred_c = [round(i.item()) for i in y_pred]
            roce1 = get_roce(y_pred, y_label, 0.5)
            roce2 = get_roce(y_pred, y_label, 1)
            roce3 = get_roce(y_pred, y_label, 2)
            roce4 = get_roce(y_pred, y_label, 5)
            print("AUROC: " + str(roc_auc_score(y_label, y_pred)), end=" ")
            print("PRAUC: " + str(average_precision_score(y_label, y_pred)), end=" ")
            print("F1 Score: " + str(f1_score(y_label, y_pred_c)), end=" ")
            print("Precision Score:" + str(precision_score(y_label, y_pred_c)), end=" ")
            print("Recall Score:" + str(recall_score(y_label, y_pred_c)), end=" ")
            print("Balanced Accuracy Score " + str(balanced_accuracy_score(y_label, y_pred_c)), end=" ")
            print("0.5 re Score " + str(roce1), end=" ")
            print("1 re Score " + str(roce2), end=" ")
            print("2 re Score " + str(roce3), end=" ")
            print("5 re Score " + str(roce4), end=" ")
            print("-------------------")


        EPOCHS = 100
        BATCH_SIZE = 64
        with open("model.log", "a") as f:
            for epoch in range(EPOCHS):
                losses = []
                accs = []
                with tqdm(range(0, len(X), BATCH_SIZE)) as tepoch:
                    for i in tepoch:
                        tepoch.set_description(f"Epoch {epoch + 1}")
                        try:
                            batch_X = X[i: i + BATCH_SIZE]
                            batch_y = y[i: i + BATCH_SIZE]
                            # batch_p = p[i: i + BATCH_SIZE]
                        except:
                            gc.collect()
                            continue
                        acc, loss, _ = fwd_pass(batch_X, batch_y, train=True)

                        losses.append(loss.item())
                        accs.append(acc)
                        acc_mean = np.array(accs).mean()
                        loss_mean = np.array(losses).mean()
                        tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)
                        if i % 100000 == 0:
                            test_func(model, y_test, X_test)
                            f.write(
                                f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)}\n")
                    scheduler.step()
                print(f'Average Loss: {np.array(losses).mean()}')
                print(f'Average Accuracy: {np.array(accs).mean()}')






