import numpy as np
import random
import math

from sklearn.metrics import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

    
def word2idx(word, words):
    if word in words.keys():
        return int(words[word])
    
    return 0

def pad_seq(dataset, max_len):
    output = []
    for seq in dataset:
        pad = np.zeros(max_len)
        pad[:len(seq)] = seq
        output.append(pad)
        
    return np.array(output)

def rna2vec(seqset):
    letters = ['A', 'C', 'G', 'U', 'N']

    words = np.array([letters[i] + letters[j] + letters[k]
                     for i in range(len(letters))
                     for j in range(len(letters))
                     for k in range(len(letters))])

    words = {word: i + 1 for i, word in enumerate(words)}

    outputs = []

    for seq in seqset:
        output = []

        converted_seq = dna2rna(seq)  # Assuming dna2rna function converts DNA to RNA

        for i in range(0, len(converted_seq) - 2):  # -2 so we can index 3 letters
            output.append(word2idx(converted_seq[i:i + 3], words))

        if sum(output) != 0:
            # pad individual sequence
            padded_seq = np.pad(output, (0, 275 - len(output)), 'constant', constant_values=0)
            outputs.append(padded_seq)

    return np.array(outputs)

def rna2vec_pretraining(seqset):
    letters = ['A','C','G','U','N']
    
    words = np.array([letters[i]+letters[j]+letters[k]
                     for i in range(len(letters)) 
                     for j in range(len(letters)) 
                     for k in range(len(letters))])
    words = np.unique(words) 
    # print("words:", len(words))
    words = {word:i+1 for i, word in enumerate(words)}
    
    ss = ['S','H','M','I','B','X','E']
    words_ss = np.array([i + j + k for i in ss for j in ss for k in ss])
    words_ss = np.unique(words_ss)
    # print("words_ss:", len(words_ss))
    words_ss = {word:i+1 for i, word in enumerate(words_ss)}
    
    outputs = []
    outputs_ss = []
    for seq, ss in seqset:
        output = [] 
        output_ss = []
        conveted_seq = dna2rna(seq)

        for i in range(0, len(conveted_seq)-1):
            output.append(word2idx(conveted_seq[i:i+3], words))
            output_ss.append(word2idx(ss[i:i+3], words_ss))

            if len(output) == 275:
                outputs.append(np.array(output))
                outputs_ss.append(np.array(output_ss))
                
                output = []
                output_ss = []
                
        # We'll handle the padding of individual sequences before appending to the outputs list
        if len(output) > 0:
            padded_output = np.pad(output, (0, 275 - len(output)), 'constant', constant_values=0)
            outputs.append(padded_output)

            padded_output_ss = np.pad(output_ss, (0, 275 - len(output_ss)), 'constant', constant_values=0)
            outputs_ss.append(padded_output_ss)

    return np.array(outputs), np.array(outputs_ss)

def seq2vec(seqset, max_len, n_vocabs, n_target_vocabs, words, words_ss):
    word_max_len= 3
    
    outputs = []
    outputs_ss = []
    for seq, ss in seqset:
        output = []
        output_ss = [] 
        i = 0
        while i < len(seq):
            flag=False
            for j in range(word_max_len, 0, -1):
                if i+j <=len(seq):
                    if word2idx(seq[i:i+j], words)!= 0:
                        flag = True
                        output.append(word2idx(seq[i:i+j], words))
                        output_ss.append(word2idx(ss[i:i+j], words_ss))

                        if len(output)==max_len:
                            outputs.append(np.array(output))
                            outputs_ss.append(np.array(output_ss))
                            output = [] 
                            output_ss = []
                        i+=j
                        break 
            if flag==False: 
                i+=1
        if len(output) != 0: 
            outputs.append(np.array(output))
            outputs_ss.append(np.array(output_ss))
        
    return pad_seq(outputs, max_len), pad_seq(outputs_ss, max_len)

def tokenize_sequences(seqset, max_len, n_vocabs, words, word_max_len=3):
    outputs = []
    for seq in seqset:
        output = []
        i = 0
        while i < len(seq): 
            flag=False
            for j in range(word_max_len, 0, -1): 
                if i+j <=len(seq): 
                    if word2idx(seq[i:i+j], words)!= 0: 
                        flag = True 
                        output.append(word2idx(seq[i:i+j], words)) 
                        if len(output)==max_len: 
                            outputs.append(np.array(output))
                            output = [] 
                        i+=j 
                        break 
            if flag==False:
                i+=1
                
        if len(output) != 0: 
            outputs.append(np.array(output))
        
    return pad_seq(outputs, max_len)

def str2bool(seq):
    out = []
    for s in seq:
        if s == "positive": 
            out.append(1)
        elif s == "negative": 
            out.append(0)
            
    return np.array(out)

def dna2rna(seq):
    mapping = {'A':'A','C':'C','G':'G', 'U':'U', 'T':'U'}
    result = ""
    for s in seq:
        if s in mapping.keys():
            result += mapping[s]
        else:
            result += 'N'
            
    return result

class Custom_Dataset(Dataset):
    def __init__(self, x, y):
        super(Dataset, self).__init__()
        
        self.x = np.array(x, dtype=np.int64)
        self.y = np.array(y, dtype=np.int64)
        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.int64), torch.tensor(self.y[index], dtype=torch.int64)
    
class Masked_Dataset(Dataset):
    def __init__(self, x, y, max_len, masked_rate, mask_idx, isrna=False):
        self.x = np.array(x)
        self.y = np.array(y)
        self.box = np.array([i for i in range(max_len)])
        self.masked_rate = masked_rate
        self.mask_idx = mask_idx
        self.isrna = isrna
        
        if len(self.x) != len(self.y):
            raise
        else:
            self.len = len(self.x)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.int64)
        y = torch.tensor(self.y[index], dtype=torch.int64)
       
        masks = []
        x_masked = x.clone().detach()
        y_masked = x.clone().detach()
        
        seq_len = torch.sum(x_masked > 0)
        mask = random.sample(self.box[x_masked > 0].tolist(), int(seq_len * self.masked_rate))
        masks.append(mask)
        no_mask = [b for b in self.box[x_masked > 0].tolist() if b not in mask]
        
        mask = random.sample(mask, int(len(mask) * 0.8))
        x_masked[mask] = self.mask_idx #msk
        
        if self.isrna==True:
            x_masked[[m+1 for m in mask if m < 274]] = self.mask_idx
            x_masked[[m-1 for m in mask if m >0]] = self.mask_idx
        
        y_masked[no_mask] = 0
        
        return x_masked, y_masked, x, y
    
class API_Dataset(Dataset):
    def __init__(self, apta, prot, y):
        super(Dataset, self).__init__()
        
        self.apta = np.array(apta, dtype=np.int64)
        self.prot = np.array(prot, dtype=np.int64)
        self.y = np.array(y, dtype=np.int64)
        self.len = len(self.apta)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(self.apta[index], dtype=torch.int64), torch.tensor(self.prot[index], dtype=torch.int64), torch.tensor(self.y[index], dtype=torch.int64)

def find_opt_threshold(target, pred):
    result = 0
    best = 0
    
    for i in range(0, 1000):
        pred_threshold = np.where(pred > i/1000, 1, 0)
        now = f1_score(target, pred_threshold)
        if now > best:
            result = i/1000
            best = now
            
    return result

def argument_seqset(seqset):
    arg_seqset = []
    for s, ss in seqset:
        arg_seqset.append([s, ss]) 
        
        arg_seqset.append([s[::-1], ss[::-1]])

    return arg_seqset

def augment_apis(apta, prot, ys):
    aug_apta = []
    aug_prot = []
    aug_y = []
    for a, p, y in zip(apta, prot, ys):
        aug_apta.append(a) 
        aug_prot.append(p)
        aug_y.append(y)
        
        aug_apta.append(a[::-1]) 
        aug_prot.append(p)
        aug_y.append(y)
        
    return np.array(aug_apta), np.array(aug_prot), np.array(aug_y)


def get_dataset(filepath, prot_max_len, n_prot_vocabs, prot_words):
    with open(filepath,"rb") as fr:
        dataset = pickle.load(fr)
        dataset_train = np.array(dataset[dataset["dataset"]=="training dataset"])
        dataset_test = np.array(dataset[dataset["dataset"]=="test dataset"])

        arg_apta, arg_prot, arg_y = augment_apis(dataset_train[:, 1], dataset_train[:, 2], dataset_train[:, 0])
        datasets_train = [rna2vec(arg_apta), tokenize_sequences(arg_prot, prot_max_len, n_prot_vocabs, prot_words), str2bool(arg_y)]
        datasets_test = [rna2vec(dataset_test[:, 1]), tokenize_sequences(dataset_test[:, 2], prot_max_len, n_prot_vocabs, prot_words), str2bool(dataset_test[:, 0])]

    return datasets_train, datasets_test


def get_scores(target, pred):
    threshold = find_opt_threshold(target, pred)
    pred_threshold = np.where(pred > threshold, 1, 0)
    acc = accuracy_score(target, pred_threshold)
    roc_auc = roc_auc_score(target, pred)
    mcc = matthews_corrcoef(target, pred_threshold)
    f1 = f1_score(target, pred_threshold)
    pr_auc = average_precision_score(target, pred)
    cls_report = classification_report(target, pred_threshold)
    scores = {
        'threshold': threshold,
        'acc': acc,
        'roc_auc': roc_auc,
        'mcc': mcc, 
        'f1': f1, 
        'pr_auc': pr_auc,
        'cls_report': cls_report
    }
    return scores

