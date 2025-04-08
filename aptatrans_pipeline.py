import torch
import torch.nn as nn
import numpy as np
import pickle
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from typing import Tuple
import os
import math

from utils import (
    tokenize_sequences, rna2vec, seq2vec, rna2vec_pretraining,
    get_dataset, get_scores, argument_seqset, API_Dataset, Masked_Dataset
)
from encoders import Token_Predictor, Encoders, CONVBlocks, Predictor, To_IteractionMap, AptaTransWrapper
from mcts import MCTS
import random

class Timer:
    """Class for timing blocks of code."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = timer()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        print(f"{self.name}: {timer() - self.start:.4f} seconds")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
class AptaTransPipeline:
    def __init__(
            self,
            dim: int = 128,
            mult_ff: int = 2,
            n_layers: int = 6,
            n_heads: int = 8,
            dropout: float = 0.1,
            channel_size: int = 64,
            save_name: str = 'default',
            load_best_pt: bool = True,
            load_best_model: bool = False,
            device: str = 'cpu',
            seed: int = 1004,
            data_dir: str = './data'
    ):
        """Initialize AptaTransPipeline with model configurations."""
        self.seed = seed
        seed_everything(seed)
        self.device = device
        self.data_dir = data_dir

        self.save_name = save_name
        os.makedirs(f"./models/{self.save_name}", exist_ok=True)

        self._initialize_constants()
        self._load_protein_words()
        self._initialize_encoders(dim, mult_ff, n_layers, n_heads, dropout, channel_size, load_best_pt, load_best_model)

    def _initialize_constants(self):
        """Initialize constants used in the pipeline."""
        self.n_apta_vocabs = 127
        self.n_apta_target_vocabs = 344
        self.n_prot_vocabs = 715
        self.n_prot_target_vocabs = 585
        self.apta_max_len = 275
        self.prot_max_len = 867

    def _load_protein_words(self):
        """Load protein words from a pickle file."""
        with open(os.path.join(self.data_dir, 'protein_word_freq_3.pickle'), 'rb') as fr:
            words = pickle.load(fr)
            words = words[words["freq"] > words.freq.mean()].seq.values
            self.prot_words = {word: i + 1 for i, word in enumerate(words)}

        letters = ['A', 'C', 'G', 'U', 'N']
        words = np.array([letters[i] + letters[j] + letters[k]
                         for i in range(len(letters))
                         for j in range(len(letters))
                         for k in range(len(letters))])
        self.apta_words = {word: i + 1 for i, word in enumerate(words)}

    def _initialize_encoders(self, dim, mult_ff, n_layers, n_heads, dropout, channel_size=64, load_best_pt=False, load_best_model=False):
        """Initialize the encoder models and other components."""
        self.encoder_aptamer = Encoders(
            n_vocabs=self.n_apta_vocabs,
            n_layers=n_layers,
            n_heads=n_heads,
            dim=dim,
            mult_ff=mult_ff,
            dropout=dropout,
            max_len=self.apta_max_len
        ).to(self.device)
        
        self.token_predictor_aptamer = Token_Predictor(
            n_vocabs=self.n_apta_vocabs,
            n_target_vocabs=self.n_apta_target_vocabs,
            dim=dim
        ).to(self.device)
        
        self.encoder_protein = Encoders(
            n_vocabs=self.n_prot_vocabs,
            n_layers=n_layers,
            n_heads=n_heads,
            dim=dim,
            mult_ff=mult_ff,
            dropout=dropout,
            max_len=self.prot_max_len
        ).to(self.device)
        
        self.token_predictor_protein = Token_Predictor(
            n_vocabs=self.n_prot_vocabs,
            n_target_vocabs=self.n_prot_target_vocabs,
            dim=dim
        ).to(self.device)
        
        self.to_im = To_IteractionMap().to(self.device)
        self.conv = CONVBlocks(out_channels=channel_size).to(self.device)
        self.predictor = Predictor(channel_size=channel_size).to(self.device)

        if load_best_pt:
            self._load_pretrained_models()

        if load_best_model:
            self._load_best_model()

    def _load_pretrained_models(self):
        """Load pre-trained models for aptamer and protein encoders."""
        try:
            self.encoder_aptamer.load_state_dict(torch.load(f"./models/{self.save_name}/pretrained_encoder_rna.pt"))
            self.encoder_protein.load_state_dict(torch.load(f"./models/{self.save_name}/pretrained_encoder_protein.pt"))
            print('Pre-trained models loaded successfully!')
        except FileNotFoundError:
            print('No pre-trained model files found. Pre-training required!')

    def set_data_for_training(self, batch_size: int):
        """Set data for training."""
        datapath = os.path.join(self.data_dir, 'dataset_li.pickle')
        ds_train, ds_test = get_dataset(datapath, self.prot_max_len, self.n_prot_vocabs, self.prot_words)
        self.train_loader = DataLoader(API_Dataset(ds_train[0], ds_train[1], ds_train[2]), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(API_Dataset(ds_test[0], ds_test[1], ds_test[2]), batch_size=batch_size, shuffle=False)

    def set_data_rna_pt(self, batch_size: int, masked_rate: float = 0.15):
        """Set data for RNA pre-training."""
        conn = sqlite3.connect(os.path.join(self.data_dir, 'bpRNA.db'))
        seqset = self._fetch_rna_sequences(conn)
        seqset = argument_seqset(seqset)
        train_seq, test_seq = train_test_split(seqset, test_size=0.05, random_state=self.seed)
        train_x, train_y = rna2vec_pretraining(train_seq)
        test_x, test_y = rna2vec_pretraining(test_seq)
        self.rna_train = self._create_dataloader(train_x, train_y, batch_size, masked_rate, self.apta_max_len, isrna=True)
        self.rna_test = self._create_dataloader(test_x, test_y, batch_size, masked_rate, self.apta_max_len, isrna=True)

    def _fetch_rna_sequences(self, conn):
        """Fetch RNA sequences from the database."""
        results = conn.execute("SELECT * FROM RNA")
        fetch = results.fetchall()
        return [[f[1], f[2]] for f in fetch if len(f[1]) <= 277]

    def _create_dataloader(self, x, y, batch_size, masked_rate, max_len, isrna=False):
        """Create a DataLoader for masked datasets."""
        dataset = Masked_Dataset(x, y, max_len, masked_rate, self.n_apta_vocabs - 1 if isrna else self.n_prot_vocabs - 1, isrna=isrna)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def set_data_protein_pt(self, batch_size: int, masked_rate: float = 0.15):
        """Set data for protein pre-training."""
        conn = sqlite3.connect(os.path.join(self.data_dir, 'protein_ss_keywords.db'))
        seqset = self._fetch_protein_sequences(conn)
        train_seq, test_seq = train_test_split(seqset, test_size=0.05, random_state=self.seed)
        train_x, train_y = seq2vec(train_seq, self.prot_max_len, self.n_prot_vocabs, self.n_prot_target_vocabs, self.prot_words, self._get_protein_words_ss())
        test_x, test_y = seq2vec(test_seq, self.prot_max_len, self.n_prot_vocabs, self.n_prot_target_vocabs, self.prot_words, self._get_protein_words_ss())
        self.protein_train = self._create_dataloader(train_x, train_y, batch_size, masked_rate, self.prot_max_len)
        self.protein_test = self._create_dataloader(test_x, test_y, batch_size, masked_rate, self.prot_max_len)

    def _fetch_protein_sequences(self, conn):
        """Fetch protein sequences from the database."""
        results = conn.execute("SELECT SEQUENCE, SS FROM PROTEIN")
        fetch = results.fetchall()
        return [[f[0], f[1]] for f in fetch]

    def _get_protein_words_ss(self):
        """Get protein words for secondary structure prediction."""
        ss = ['', 'H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
        words_ss = np.array([i + j + k for i in ss for j in ss for k in ss[1:]])
        words_ss = np.unique(words_ss)
        return {word: i + 1 for i, word in enumerate(words_ss)}

    def train(self, epochs: int, lr: float = 1e-5):
        """Train the model."""
        print('Training the model!')
        self.best_auc = 0
        self.best_epoch = 0
        self._initialize_optimizer_and_criterion(lr)

        for epoch in range(1, epochs + 1):
            self._train_epoch(epoch)

    def _initialize_optimizer_and_criterion(self, lr):
        """Initialize optimizer and criterion for training."""
        model_parameters = (
            list(self.encoder_aptamer.parameters()) +
            list(self.encoder_protein.parameters()) +
            list(self.to_im.parameters()) +
            list(self.conv.parameters()) +
            list(self.predictor.parameters())
        )
        self.optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=1e-5)
        self.criterion = nn.BCELoss().to(self.device)

    def _set_train_mode(self):
        """Set the model to train mode."""
        self.encoder_aptamer.train()
        self.encoder_protein.train()
        self.to_im.train()
        self.conv.train()
        self.predictor.train()

    def _set_eval_mode(self):
        """Set the model to evaluation mode."""
        self.encoder_aptamer.eval()
        self.encoder_protein.eval()
        self.to_im.eval()
        self.conv.eval()
        self.predictor.eval()

    def _train_epoch(self, epoch):
        """Train the model for a single epoch."""
        self._set_train_mode()
        loss_train, pred_train, target_train = self._batch_step(self.train_loader, train_mode=True)
        print(f"\n[EPOCH: {epoch}], \tTrain Loss: {loss_train:.6f}")

        self._set_eval_mode()
        with torch.no_grad():
            loss_test, pred_test, target_test = self._batch_step(self.test_loader, train_mode=False)
            scores = get_scores(target_test, pred_test)
            print(f"\nTest Loss: {loss_test:.6f}\tTest ACC: {scores['acc']:.6f}\tTest AUC: {scores['roc_auc']:.6f}\t"
                  f"Test MCC: {scores['mcc']:.6f}\tTest PR_AUC: {scores['pr_auc']:.6f}\tF1: {scores['f1']:.6f}")

        if scores['roc_auc'] > self.best_auc:
            self.best_epoch = epoch
            self.best_auc = scores['roc_auc']
            print(f"[{self.best_epoch}], best auc: {self.best_auc}, score: {scores['roc_auc']}")
            self._save_best_model()
        else:
            print(f'It is not best model (best aur roc: {self.best_auc}, best epoch: {self.best_epoch})')
            

    def _batch_step(self, loader: DataLoader, train_mode: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """Perform a single batch step."""
        loss_total = 0
        pred, target = [], []

        for batch_idx, (apta, prot, y) in enumerate(loader):
            if train_mode:
                self.optimizer.zero_grad()

            y_pred = self.predict(apta, prot)
            y_true = y.clone().detach().float().to(self.device)
            loss = self.criterion(torch.flatten(y_pred), y_true)

            if train_mode:
                loss.backward()
                self.optimizer.step()

            loss_total += loss.item()
            pred.extend(torch.flatten(y_pred).clone().detach().cpu().numpy())
            target.extend(torch.flatten(y_true).clone().detach().cpu().numpy())
            self._print_batch_progress(batch_idx, loader, train_mode, loss_total)

        return loss_total / len(loader), np.array(pred), np.array(target)

    def predict(self, apta: torch.Tensor, prot: torch.Tensor) -> torch.Tensor:
        """Predict the output using the model."""
        apta, prot = apta.to(self.device), prot.to(self.device)
        apta_encoded = self.encoder_aptamer(apta)
        prot_encoded = self.encoder_protein(prot)
        interaction_map = self.to_im(apta_encoded, prot_encoded)
        out = self.conv(interaction_map)
        out = self.predictor(out)
        return out

    def generate_interaction_map(self, apta, prot):
        """Generate interaction map for the given aptamer-protein pair."""
        apta, prot = apta.to(self.device), prot.to(self.device)
        apta_encoded = self.encoder_aptamer(apta)
        prot_encoded = self.encoder_protein(prot)
        interaction_map = self.to_im(apta_encoded, prot_encoded)
        return interaction_map

    def predict_proba(self, interaction_map):
        """Predict the probability of interaction."""
        out = torch.tensor(interaction_map).float().to(self.device)
        out = torch.unsqueeze(out, 1)
        out = torch.mean(out, 4)
        out = self.conv(out)
        out = self.predictor(out)
        out = np.array([[1 - o[0], o[0]] for o in out.clone().detach().cpu().numpy()])
        return out

    def pretrain_encoder_aptamer(self, epochs: int, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Pre-train the aptamer encoder."""
        savepath = f"./models/{self.save_name}/pretrained_encoder_rna.pt"
        model_parameters = list(self.encoder_aptamer.parameters()) + list(self.token_predictor_aptamer.parameters())
        self.optimizer_pt_apta = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
        self.criterion_mlm_apta = nn.CrossEntropyLoss().to(self.device)
        self.criterion_ssp_apta = nn.CrossEntropyLoss().to(self.device)
        best_loss = float('inf')

        with Timer("Total training time"):
            for epoch in range(1, epochs + 1):
                self.encoder_aptamer.train()
                self.token_predictor_aptamer.train()
                train_loss, train_mlm, train_ssp = self._run_epoch_pt_apta(self.rna_train, train_mode=True)
                print(f"\n[EPOCH {epoch}] Train Loss: {train_loss:.6f} Train MLM: {train_mlm:.6f} Train SSP: {train_ssp:.6f}", end='')

                with torch.no_grad():
                    self.encoder_aptamer.eval()
                    self.token_predictor_aptamer.eval()
                    test_loss, test_mlm, test_ssp = self._run_epoch_pt_apta(self.rna_test, train_mode=False)
                    print(f"\nTest Loss: {test_loss:.6f} Test MLM: {test_mlm:.6f} Test SSP: {test_ssp:.6f}")

                    if test_loss < best_loss:
                        best_loss = test_loss
                        torch.save(self.encoder_aptamer.state_dict(), savepath)

    def _run_epoch_pt_apta(self, loader: DataLoader, train_mode: bool) -> Tuple[float, float, float]:
        """Run a single epoch for pre-training."""
        total_loss, total_mlm, total_ssp = 0.0, 0.0, 0.0

        for batch_idx, (x_masked, y_masked, x, y_ss) in enumerate(loader):
            inputs_mlm, inputs, y_masked, y_ss = self._to_device(x_masked, x, y_masked, y_ss)
            y_masked = y_masked.long()
            y_ss = y_ss.long()

            mlm_apta = self.encoder_aptamer(inputs_mlm)
            ssp_apta = self.encoder_aptamer(inputs)
            y_pred_mlm, y_pred_ssp = self.token_predictor_aptamer(mlm_apta, ssp_apta)

            l_mlm = self.criterion_mlm_apta(y_pred_mlm.transpose(1, 2), y_masked)
            l_ssp = self.criterion_ssp_apta(y_pred_ssp.transpose(1, 2), y_ss)
            loss = l_mlm * 2 + l_ssp

            if torch.isnan(loss):
                print(f"NaN detected in loss at batch {batch_idx} [MLM: {l_mlm.item()}][SSP: {l_ssp.item()}]")
                self.debug_forward_pass(inputs, inputs_mlm, y_pred_mlm, y_pred_ssp, l_mlm, l_ssp, loss)
                continue

            if train_mode:
                self.optimizer_pt_apta.zero_grad()
                loss.backward()
                self.optimizer_pt_apta.step()

            total_mlm += l_mlm.item()
            total_ssp += l_ssp.item()
            total_loss += loss.item()
            self._print_batch_progress(batch_idx, loader, train_mode, total_loss)
        return total_loss / len(loader), total_mlm / len(loader), total_ssp / len(loader)

    def pretrain_encoder_protein(self, epochs: int, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Pre-train the protein encoder."""
        savepath = f"./models/{self.save_name}/pretrained_encoder_protein.pt"
        model_parameters = list(self.encoder_protein.parameters()) + list(self.token_predictor_protein.parameters())
        self.optimizer_pt_prot = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
        self.criterion_mlm_prot = nn.CrossEntropyLoss().to(self.device)
        self.criterion_ssp_prot = nn.CrossEntropyLoss().to(self.device)
        best_loss = float('inf')

        with Timer("Total training time"):
            for epoch in range(1, epochs + 1):
                train_loss, train_mlm, train_ssp = self._run_epoch_pt_prot(self.protein_train, train_mode=True)
                print(f"\n[EPOCH {epoch}] Train Loss: {train_loss:.6f} Train MLM: {train_mlm:.6f} Train SSP: {train_ssp:.6f}", end='')

                test_loss, test_mlm, test_ssp = self._run_epoch_pt_prot(self.protein_test, train_mode=False)
                print(f" Test Loss: {test_loss:.6f} Test MLM: {test_mlm:.6f} Test SSP: {test_ssp:.6f}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(self.encoder_protein.state_dict(), savepath)

    def _run_epoch_pt_prot(self, loader: DataLoader, train_mode: bool) -> Tuple[float, float, float]:
        """Run a single epoch for pre-training."""
        total_loss, total_mlm, total_ssp = 0.0, 0.0, 0.0

        for batch_idx, (x_masked, y_masked, x, y_ss) in enumerate(loader):
            inputs_mlm, inputs, y_masked, y_ss = self._to_device(x_masked, x, y_masked, y_ss)
            y_masked = y_masked.long()
            y_ss = y_ss.long()

            mlm_prot = self.encoder_protein(inputs_mlm)
            ssp_prot = self.encoder_protein(inputs)
            y_pred_mlm, y_pred_ssp = self.token_predictor_protein(mlm_prot, ssp_prot)

            l_mlm = self.criterion_mlm_prot(y_pred_mlm.transpose(1, 2), y_masked)
            l_ssp = self.criterion_ssp_prot(y_pred_ssp.transpose(1, 2), y_ss)
            loss = l_mlm * 2 + l_ssp

            if torch.isnan(loss):
                print(f"NaN detected in loss at batch {batch_idx} [MLM: {l_mlm.item()}][SSP: {l_ssp.item()}]")
                self.debug_forward_pass(inputs, inputs_mlm, y_pred_mlm, y_pred_ssp, l_mlm, l_ssp, loss)
                continue

            if train_mode:
                self.optimizer_pt_prot.zero_grad()
                loss.backward()
                self.optimizer_pt_prot.step()

            total_mlm += l_mlm.item()
            total_ssp += l_ssp.item()
            total_loss += loss.item()
            self._print_batch_progress(batch_idx, loader, train_mode, total_loss)
        return total_loss / len(loader), total_mlm / len(loader), total_ssp / len(loader)

    def _to_device(self, *tensors):
        """Move tensors to the device."""
        return [tensor.to(self.device) for tensor in tensors]

    def _has_nan_or_inf(self, *tensors):
        """Check for NaN or infinity in tensors."""
        for tensor in tensors:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Invalid value detected in tensors")
                return True
        return False

    def _print_batch_progress(self, batch_idx, loader, train_mode, values=None):
        """Print progress for the current batch."""
        mode = 'train' if train_mode else 'eval'
        print(f"{mode}[{batch_idx}/{len(loader)}({100. * batch_idx / len(loader):.0f}%)][Loss: {values:.4f}]    ", end='\r', flush=True)

    def debug_forward_pass(self, inputs, inputs_mlm, y_pred_mlm, y_pred_ssp, l_mlm, l_ssp, loss):
        """Print debug information for the forward pass."""
        print(f"Inputs MLM: {inputs_mlm}")
        print(f"Inputs: {inputs}")
        print(f"Y Pred MLM: {y_pred_mlm}")
        print(f"Y Pred SSP: {y_pred_ssp}")
        print(f"Loss MLM: {l_mlm}")
        print(f"Loss SSP: {l_ssp}")
        print(f"Total Loss: {loss}")

    def inference(self, apta: str, prot: str) -> np.ndarray:
        """Predict the aptamer-protein interaction."""
        print('Predicting the Aptamer-Protein Interaction')
        self._load_best_model()

        apta_tokenized = torch.tensor(rna2vec(np.array([apta])), dtype=torch.int64).to(self.device)
        prot_tokenized = torch.tensor(tokenize_sequences([prot], self.prot_max_len, self.n_prot_vocabs, self.prot_words), dtype=torch.int64).to(self.device)

        with torch.no_grad():
            self._set_eval_mode()
            y_pred = self.predict(apta_tokenized, prot_tokenized)
        score = y_pred.detach().cpu().numpy()

        return score

    def get_interactionmap(self, apta: str, prot: str, view=None, plot=False, top_k=10): 
        """Predict the aptamer-protein interaction."""
        print('Predicting the Aptamer-Protein Interaction')
        self._load_best_model()

        print(f'Aptamer: {apta}', end='\n\n')
        print(f'Target Protein: {prot}', end='\n\n')

        # Tokenization
        apta_tokenized = torch.tensor(rna2vec(np.array([apta])), dtype=torch.int64).to(self.device)
        prot_tokenized = torch.tensor(tokenize_sequences([prot], self.prot_max_len, self.n_prot_vocabs, self.prot_words), dtype=torch.int64).to(self.device)

        # Generate interaction map
        with torch.no_grad():
            self._set_eval_mode()
            im = self.generate_interaction_map(apta_tokenized, prot_tokenized)

        apta_tokenized = apta_tokenized[apta_tokenized != 0]
        prot_tokenized = prot_tokenized[prot_tokenized != 0]

        # Reverse tokens
        reversed_apta_words = {v: k for k, v in self.apta_words.items()}
        apta_tokens = [reversed_apta_words[value] for value in apta_tokenized.cpu().numpy()]
        reversed_prot_words = {v: k for k, v in self.prot_words.items()}
        prot_tokens = [reversed_prot_words[value] for value in prot_tokenized.cpu().numpy()]

        # Adjust interaction map size
        im = im[0, 0, :len(apta_tokenized), :len(prot_tokenized)]

        # View-based logic
        if view in ['apta', 'Apta', 'Aptamer', 'aptamer']:
            im = im.softmax(dim=0)
            interaction_scores = im.sum(axis=1)
            seq_tokens = apta_tokens
            apta_indices = np.argsort(interaction_scores.detach().cpu().numpy())[-top_k:]
            prot_indices = [i for i in range(len(prot_tokens))]
        elif view in ['prot', 'Prot', 'Protein', 'protein', 'target', 'Target']:
            im = im.softmax(dim=1)
            interaction_scores = im.sum(axis=0)
            seq_tokens = prot_tokens
            apta_indices = [i for i in range(len(apta_tokens))]
            prot_indices = np.argsort(interaction_scores.detach().cpu().numpy())[-top_k:]
        else:  # Combined view
            im_apta = im.softmax(dim=0)
            im_prot = im.softmax(dim=1)
            interaction_scores_apta = im_apta.sum(axis=1)
            interaction_scores_prot = im_prot.sum(axis=0)

            apta_indices = np.argsort(interaction_scores_apta.detach().cpu().numpy())[-top_k:]
            prot_indices = np.argsort(interaction_scores_prot.detach().cpu().numpy())[-top_k:]

        im = im.detach().cpu().numpy()
        
        # Plotting
        if plot:
            print(f'Plotting for Top {top_k} strongest interactions in the interaction map')
            plt.figure(figsize=(20, 8))
            plt.imshow(im, aspect='auto', cmap='viridis')
            plt.xticks(ticks=prot_indices, labels=[prot_tokens[i] for i in prot_indices], rotation=90)
            plt.yticks(ticks=apta_indices, labels=[apta_tokens[i] for i in apta_indices])
            plt.xlabel('Proteins')
            plt.ylabel('Aptamers')
            plt.colorbar()
            plt.show()    

            if view is not None:
                indices_top = set(apta_indices) if view in ['apta', 'Apta', 'Aptamer', 'aptamer'] else set(prot_indices)
                # Multi-line text plot
                default_color = "black"
                highlight_color = "red"
                num_cols = 50
                num_rows = math.ceil(len(seq_tokens) / num_cols)  
                fig, ax = plt.subplots(figsize=(20, num_rows * 0.5))  
                ax.set_xlim(0, num_cols)
                ax.set_ylim(0, num_rows)

                for i, text in enumerate(seq_tokens):
                    row = i // num_cols  
                    col = i % num_cols   
                    color = highlight_color if i in indices_top else default_color
                    ax.text(col + 0.5, num_rows - row - 0.5, text, ha="center", va="center", color=color, fontsize=8)
                ax.axis("off")
                plt.show()

        return im, apta_tokens, prot_tokens

    def _load_best_model(self):
        """Load the best model for API."""
        try:
            print("Loading the best model for API!")
            self.encoder_aptamer.load_state_dict(torch.load(f'./models/{self.save_name}/encoder_apta_best_auc.pt', map_location=self.device))
            self.encoder_protein.load_state_dict(torch.load(f'./models/{self.save_name}/encoder_prot_best_auc.pt', map_location=self.device))
            self.to_im.load_state_dict(torch.load(f'./models/{self.save_name}/to_im_best_auc.pt', map_location=self.device))
            self.conv.load_state_dict(torch.load(f'./models/{self.save_name}/conv_best_auc.pt', map_location=self.device))
            self.predictor.load_state_dict(torch.load(f'./models/{self.save_name}/predictor_best_auc.pt', map_location=self.device))
        except FileNotFoundError:
            print('No best model file found. Training required!')

    def _save_best_model(self):
        """Save the best model for API."""
        try:
            torch.save(self.encoder_aptamer.state_dict(), f"./models/{self.save_name}/encoder_apta_best_auc.pt")
            torch.save(self.encoder_protein.state_dict(), f"./models/{self.save_name}/encoder_prot_best_auc.pt")
            torch.save(self.to_im.state_dict(), f"./models/{self.save_name}/to_im_best_auc.pt")
            torch.save(self.conv.state_dict(), f"./models/{self.save_name}/conv_best_auc.pt")
            torch.save(self.predictor.state_dict(), f"./models/{self.save_name}/predictor_best_auc.pt")
            print('Saved the best model!')
        except Exception as e:
            print(f"Error saving the best model: {e}")

    def recommend(self, target: str, n_aptamers: int, depth: int, iteration: int, verbose: bool = True):
        """Recommend aptamers for a target protein."""
        self._load_best_model()
        self.classifier = AptaTransWrapper(
            self.encoder_aptamer, 
            self.encoder_protein, 
            self.to_im, 
            self.conv, 
            self.predictor
        ).to(self.device)
        candidates = []
        scores = []
        results = {}
        
        encoded_targetprotein = self._encode_target_protein(target)
        with torch.no_grad():
            self._set_eval_mode()
            mcts = MCTS(encoded_targetprotein, depth=depth, iteration=iteration, states=8, target_protein=target, device=self.device)

            for i in range(n_aptamers):
                mcts.make_candidate(self.classifier)
                candidate = mcts.get_candidate()
                score = self._evaluate_candidate(mcts, encoded_targetprotein, verbose)
                results[i] = {'candidate': candidate, 'score': score}

        self._final_evaluation(results, verbose)

        return results


    def _encode_target_protein(self, target):
        """Encode the target protein."""
        return torch.tensor(tokenize_sequences([target], self.prot_max_len, self.n_prot_vocabs, self.prot_words), dtype=torch.int64).to(self.device)

    def _evaluate_candidate(self, mcts, encoded_targetprotein, verbose):
        """Evaluate a single candidate."""
        self.classifier.eval()
        with torch.no_grad():
            sim_seq = np.array([mcts.get_candidate()])
            apta = torch.tensor(rna2vec(sim_seq), dtype=torch.int64).to(self.device)
            score = self.classifier(apta, encoded_targetprotein)

        if verbose:
            print(f"Candidate: {mcts.get_candidate()}\tScore: {score.cpu().numpy()}")
            print("*" * 80)
        mcts.reset()

        return score

    def _final_evaluation(self, results, verbose):
        """Perform final evaluation of candidates."""
        for i in results.keys():
            if verbose:
                print(f"Candidate: {results[i]['candidate']}, Score: {results[i]['score']}")
