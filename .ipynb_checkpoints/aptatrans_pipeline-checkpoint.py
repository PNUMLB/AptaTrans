from sklearn.model_selection import train_test_split
from utils import tokenize_sequences, rna2vec, seq2vec, rna2vec_pretraining, get_dataset, get_scores, argument_seqset
from encoders import AptaTrans, Encoders, Token_Pretrained_Model
from utils import API_Dataset, Masked_Dataset
from mcts import MCTS
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import sqlite3
import timeit
import numpy as np
import pickle

class AptaTransPipeline:
    def __init__(
            self,
            d_model=128,
            d_ff=512,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            load_best_pt=True,
            device='cpu',
            seed=1004,
            ):
        
        self.seed = seed
        self.device = device
        self.n_apta_vocabs = 1 + 125 + 1 # pad + voca + msk
        self.n_apta_target_vocabs = 1 + 343 # pad + voca
        self.n_prot_vocabs = 1 + 713 + 1 # pad + voca + msk
        self.n_prot_target_vocabs = 1 + 584 # pad + voca

        self.apta_max_len = 275
        self.prot_max_len = 867

        with open('./data/protein_word_freq_3.pickle', 'rb') as fr:
            words = pickle.load(fr)
            words = words[words["freq"]>words.freq.mean()].seq.values
            self.prot_words = {word:i+1 for i, word in enumerate(words)}
    
        self.encoder_aptamer = Token_Pretrained_Model(
            n_vocabs=self.n_apta_vocabs,
            n_target_vocabs=self.n_apta_target_vocabs,
            d_ff=d_ff,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=self.apta_max_len
        ).to(device)

        self.encoder_protein = Token_Pretrained_Model(
            n_vocabs=self.n_prot_vocabs,
            n_target_vocabs=self.n_prot_target_vocabs,
            d_ff=d_ff,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_len=self.prot_max_len
        ).to(device)

        if load_best_pt:
            try:
                self.encoder_aptamer.load_state_dict(torch.load("./models/rna_pretrained_encoder.pt"))
                self.encoder_protein.load_state_dict(torch.load("./models/protein_pretrained_encoder.pt"))
                print('Best pre-trained models are loaded!')
            except:
                print('There are no best pre-trained model files..')
                print('You need to pre-train the ecoders!')


        self.model = AptaTrans(
            apta_encoder=self.encoder_aptamer.encoder, 
            prot_encoder=self.encoder_protein.encoder, 
            n_apta_vocabs=self.n_apta_vocabs, 
            n_prot_vocabs=self.n_prot_vocabs, 
            dropout=dropout,
            apta_max_len=self.apta_max_len,
            prot_max_len=self.prot_max_len
            ).to(device)
        
    def set_data_for_training(self, batch_size):
        datapath="./data/dataset_li.pickle"
        ds_train, ds_test = get_dataset(datapath, self.prot_max_len, self.n_prot_vocabs, self.prot_words)

        self.train_loader = DataLoader(API_Dataset(ds_train[0], ds_train[1], ds_train[2]), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(API_Dataset(ds_test[0], ds_test[1], ds_test[2]), batch_size=batch_size, shuffle=False)

    def set_data_rna_pt(self, batch_size, masked_rate=0.15):
        conn = sqlite3.connect("./data/bpRNA.db")
        results = conn.execute("SELECT * FROM RNA")
        fetch = results.fetchall()
        seqset = [[f[1], f[2]] for f in fetch if len(f[1]) <= 277]
        seqset = argument_seqset(seqset)

        train_seq, test_seq = train_test_split(seqset, test_size=0.05, random_state=self.seed)
        train_x, train_y = rna2vec_pretraining(train_seq)
        test_x, test_y = rna2vec_pretraining(test_seq)

        rna_train = Masked_Dataset(train_x, train_y, self.apta_max_len, masked_rate, self.n_apta_vocabs-1, isrna=True)
        rna_test = Masked_Dataset(test_x, test_y, self.apta_max_len, masked_rate, self.n_apta_vocabs-1, isrna=True)

        self.rna_train = DataLoader(rna_train, batch_size=batch_size, shuffle=True)
        self.rna_test = DataLoader(rna_test, batch_size=batch_size, shuffle=False)

    def set_data_protein_pt(self, batch_size, masked_rate=0.15):
        ss = ['', 'H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
        words_ss = np.array([i + j + k for i in ss for j in ss for k in ss[1:]])
        words_ss = np.unique(words_ss)
        words_ss = {word:i+1 for i, word in enumerate(words_ss)}

        conn = sqlite3.connect("./data/protein_ss_keywords.db")
        results = conn.execute("SELECT SEQUENCE, SS FROM PROTEIN")
        fetch = results.fetchall()
        seqset = [[f[0], f[1]] for f in fetch]

        train_seq, test_seq = train_test_split(seqset, test_size=0.05, random_state=self.seed)
        train_x, train_y = seq2vec(train_seq, self.prot_max_len, self.n_prot_vocabs, self.n_prot_target_vocabs, self.prot_words, words_ss)
        test_x, test_y = seq2vec(test_seq, self.prot_max_len, self.n_prot_vocabs, self.n_prot_target_vocabs, self.prot_words, words_ss)

        protein_train = Masked_Dataset(train_x, train_y, self.prot_max_len, masked_rate, self.n_prot_vocabs-1)
        protein_test = Masked_Dataset(test_x, test_y, self.prot_max_len, masked_rate, self.n_prot_vocabs-1)

        self.protein_train = DataLoader(protein_train, batch_size=batch_size, shuffle=True)
        self.protein_test = DataLoader(protein_test, batch_size=batch_size, shuffle=False)

    def train(self, epochs, lr = 1e-5):
        print('Training the model!')
        best_auc = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.BCELoss().to(self.device)

        for epoch in range(1, epochs+1):
            self.model.train()
            loss_train, pred_train, target_train = self.batch_step(self.train_loader, train_mode=True)         
            print("\n[EPOCH: {}], \tTrain Loss: {: .6f}".format(epoch, loss_train), end='')
            self.model.eval()
            with torch.no_grad():
                loss_test, pred_test, target_test = self.batch_step(self.test_loader, train_mode=False)
                scores = get_scores(target_test, pred_test)
                print("\tTest Loss: {: .6f}\tTest ACC: {:.6f}\tTest AUC: {:.6f}\tTest MCC: {:.6f}\tTest PR_AUC: {:.6f}\tF1: {:.6f}\n".format(loss_test, scores['acc'], scores['roc_auc'], scores['mcc'], scores['pr_auc'], scores['f1']))
            
            if scores['roc_auc'] > best_auc:
                best_auc = scores['roc_auc']
                torch.save(self.model.module.state_dict(), "./models/AptaTrans_best_auc.pt")
                print('Saved at ./models/AptaTrans_best_auc.pt!')
        print('Done!')

    def batch_step(self, loader, train_mode = True):
        loss_total = 0
        pred = np.array([])
        target = np.array([])
        for batch_idx, (apta, prot, y) in enumerate(loader):
            if train_mode:
                self.optimizer.zero_grad()

            y_pred = self.predict(apta, prot)
            y_true = torch.tensor(y, dtype=torch.float32).to(self.device)
            loss = self.criterion(torch.flatten(y_pred), y_true)

            if train_mode:
                loss.backward()
                self.optimizer.step()

            loss_total += loss.item()
            
            pred = np.append(pred, torch.flatten(y_pred).clone().detach().cpu().numpy())
            target = np.append(target, torch.flatten(y_true).clone().detach().cpu().numpy())
            mode = 'train' if train_mode else 'eval'
            print(mode + "[{}/{}({:.0f}%)]".format(batch_idx, len(loader), 100. * batch_idx / len(loader)), end = "\r", flush=True)
        loss_total /= len(loader)
        return loss_total, pred, target

    def predict(self, apta, prot):
        apta, prot = apta.to(self.device), prot.to(self.device)
        y_pred = self.model(apta, prot)
        return y_pred

    def pretain_aptamer(self, epochs, lr=1e-5):
        savepath = "./models/rna_pretrained_encoder.pt"
        self.encoder_aptamer = self.pretraining(self.encoder_aptamer, self.rna_train, self.rna_test, savepath, epochs, lr)

    def pretrain_protein(self, epochs, lr=1e-5):
        savepath = "./models/protein_pretrained_encoder.pt"
        self.encoder_protein = self.pretraining(self.encoder_protein, self.protein_train, self.protein_test, savepath, epochs, lr)

    def pretraining(self, model, train_loader, test_loader, savepath_model, epochs, lr=1e-5):
        print('Pre-training the model')

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        best = 1000

        start_time = timeit.default_timer()
        for epoch in range(1, epochs+1):
            model.train()
            model, loss_train, mlm_train, ssp_train = self.batch_step_pt(model, train_loader, train_mode=True)
            print("\n[EPOCH: {}], \tTrain Loss: {: .6f}\tTrain mlm: {: .6f}\tTrain ssp: {: .6f}".format(epoch, loss_train, mlm_train, ssp_train), end='')
            model.eval()
            with torch.no_grad():
                model, loss_test, mlm_test, ssp_test = self.batch_step_pt(model, test_loader, train_mode=False)
                print("\Test Loss: {: .6f}\tTest mlm: {: .6f}\tTest ssp: {: .6f} ".format(epoch, loss_test, mlm_test, ssp_test))
            
            terminate_time = timeit.default_timer()
            time = terminate_time-start_time
            print("the time is %02d:%02d:%2f" % ((time//3600), (time//60)%60, time%60))
            if (loss_test) < best:
                best = loss_test
                torch.save(model.module.state_dict(), savepath_model)
        
        return model


    def batch_step_pt(self, model, loader, train_mode=True):
        loss_total, loss_mlm, loss_ssp = 0, 0, 0
        for batch_idx, (x_masked, y_masked, x, y_ss) in enumerate(loader):
            if train_mode:
                self.optimizer.zero_grad()

            inputs_mlm, inputs = x_masked.to(self.device), x.to(self.device)
            y_pred_mlm, y_pred_ssp = model(inputs_mlm, inputs)

            l_mlm = self.criterion(torch.transpose(y_pred_mlm, 1, 2), y_masked.to(self.device))
            l_ssp = self.criterion(torch.transpose(y_pred_ssp, 1, 2), y_ss.to(self.device))
            loss = l_mlm * 2 + l_ssp

            loss_mlm += l_mlm
            loss_ssp += l_ssp
            loss_total += loss

            if train_mode:
                loss.backward()
                self.optimizer.step()
            mode = 'train' if train_mode else 'eval'
            print(mode + "[{}/{}({:.0f}%)]".format(batch_idx, len(loader), 100. * batch_idx / len(loader)), end = "\r", flush=True)
            
        loss_mlm /= len(loader)
        loss_ssp /= len(loader)
        loss_total /= len(loader)

        return model, loss_total, loss_mlm, loss_ssp
    
    def inference(self, apta, prot):
        print('Predict the Aptamer-Protein Interaction')
        try:
            print("load the best model for api!")
            self.model.load_state_dict(torch.load('./models/AptaTrans_best_auc.pt', map_location=self.device))
        except:
            print('there is no best model file.')
            print('You need to train the model for predicting API!')

        print('Aptamer : ', apta)
        print('Target Protein : ', prot)

        apta_tokenized = torch.tensor(rna2vec(np.array([apta])), dtype=torch.int64).to(self.device)
        prot_tokenized = torch.tensor(tokenize_sequences(list([prot]), self.prot_max_len, self.n_prot_vocabs, self.prot_words), dtype=torch.int64).to(self.device)

        y_pred = self.model(apta_tokenized, prot_tokenized)
        score = y_pred.detach().cpu().numpy()
        print('Score : ', score)

        return score

    def recommend(self, target, n_aptamers, depth, iteration, verbose=True):
        try:
            print("load the best model for api!")
            self.model.load_state_dict(torch.load('./models/AptaTrans_best_auc.pt', map_location=self.device))
        except:
            print('there is no best model file.')
            print('You need to train the model for predicting API!')

        candidates = []
        encoded_targetprotein = torch.tensor(tokenize_sequences(list([target]), self.prot_max_len, self.n_prot_vocabs, self.prot_words), dtype=torch.int64).to(self.device)
        mcts = MCTS(encoded_targetprotein, depth=depth, iteration=iteration, states=8, target_protein=target, device=self.device)

        for _ in range(n_aptamers):
            mcts.make_candidate(self.model)
            candidates.append(mcts.get_candidate())

            self.model.eval()
            with torch.no_grad():
                sim_seq = np.array([mcts.get_candidate()])
                apta = torch.tensor(rna2vec(sim_seq), dtype=torch.int64).to(self.device)
                score = self.model(apta, encoded_targetprotein)
                
            if verbose:
                print("candidate:\t", mcts.get_candidate(), "\tscore:\t", score)
                print("*"*80)
            mcts.reset()

        encoded_targetprotein = torch.tensor(tokenize_sequences(list([target]), self.prot_max_len, self.n_prot_vocabs, self.prot_words), dtype=torch.int64).to(self.device)
        for candidate in candidates:
            with torch.no_grad():
                sim_seq = np.array([candidate])
                apta = torch.tensor(rna2vec(sim_seq), dtype=torch.int64).to("cpu")
                score = self.model(apta, encoded_targetprotein)
                
                if verbose:
                    print(f'Candidate : {candidate}, Score: {score}')