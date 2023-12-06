import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
    
class Encoder(nn.Module):
    def __init__(self, d_ff=512, d_model=128, n_heads=8, dropout=.3, max_len=512):
        super(Encoder, self).__init__()
        #hyperparameters
        self.d_ff = d_ff
        self.d_model= d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.max_len = max_len

        #layers
        self.MultiHeadAttention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.LayerNorm = nn.LayerNorm(normalized_shape=[max_len, d_model], eps=1e-6)
        self.fc_ff = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc_model = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, padding_mask):
        residual = x
        x = self.MultiHeadAttention(x,x,x, key_padding_mask=padding_mask, need_weights=False)[0]
        
        x = self.dropout(x)
        x = self.LayerNorm(x + residual)
        
        residual = x
        x = self.fc_ff(x)
        x = self.relu(x)
        x = self.fc_model(x)
        x = self.dropout(x)
        x = self.LayerNorm(x + residual)
        
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        #hyperparameters

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:x.size(0)]
        
        return x

        

class Encoders(nn.Module):
    def __init__(self, n_vocabs, n_layers=6, d_ff=512, d_model=128, n_heads=8, dropout=.3, max_len=512):
        super(Encoders, self).__init__()
        #hyperparameters
        self.n_vocabs = n_vocabs
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_model= d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout
        self.max_len = max_len
        
        #layers
        self.Embedding = nn.Embedding(num_embeddings=n_vocabs, embedding_dim=d_model, padding_idx=0)
        self.PositionalEncoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoders = nn.ModuleList([Encoder(d_ff=d_ff, d_model=d_model, n_heads=n_heads, dropout=dropout, max_len=max_len) for _ in range(n_layers)])
    
    def forward(self, x):
        padding_mask = self.create_padding_mask(x)
        x = self.Embedding(x)
        
        x = self.PositionalEncoding(x)
        x[padding_mask] = torch.zeros(self.d_model).to(x.device)
        
        for i in range(self.n_layers):
            x = self.encoders[i](x, padding_mask)

        x[padding_mask] = torch.zeros(self.d_model).to(x.device)

        return x
    
    def create_padding_mask(self, x):
        mask = torch.eq(x, torch.zeros(x.size(), device=x.device))
        
        return mask
        

class Token_Pretrained_Model(nn.Module):
    def __init__(self, n_vocabs, n_target_vocabs, d_ff, d_model, n_layers, n_heads, dropout, max_len):
        super(Token_Pretrained_Model, self).__init__()
        self.encoder = Encoders(n_vocabs=n_vocabs, d_ff=d_ff, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout, max_len=max_len)
        
        self.fc1_mlm = nn.Linear(d_model, d_model)
        self.gelu_mlm = nn.GELU()
        self.norm_mlm = nn.LayerNorm(normalized_shape=[max_len, d_model], eps=1e-6)
        self.fc2_mlm = nn.Linear(d_model, n_vocabs)
        
        self.fc1_ssp = nn.Linear(d_model, d_model)
        self.gelu_ssp = nn.GELU()
        self.norm_ssp = nn.LayerNorm(normalized_shape=[max_len, d_model], eps=1e-6)
        self.fc2_ssp = nn.Linear(d_model, n_target_vocabs)
        
    def forward(self, inputs_mlm, inputs_ssp):
        enc_mlm = self.encoder(inputs_mlm)
        output_mlm = self.fc1_mlm(enc_mlm)
        output_mlm = self.gelu_mlm(output_mlm)
        output_mlm = self.norm_mlm(output_mlm)
        output_mlm = self.fc2_mlm(output_mlm)
        output_mlm = F.log_softmax(output_mlm, dim=1)
        
        enc_ssp = self.encoder(inputs_ssp)
        output_ssp = self.fc1_ssp(enc_ssp)
        output_ssp = self.gelu_ssp(output_ssp)
        output_ssp = self.norm_ssp(output_ssp)
        output_ssp = self.fc2_ssp(output_ssp)
        output_ssp = F.log_softmax(output_ssp, dim=1)
        
        return output_mlm, output_ssp

class Convolution_Block(nn.Module):
    def __init__(self, kernel_size):
        super(Convolution_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(kernel_size, kernel_size, (3, 3), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(kernel_size)
        
        self.conv2 = nn.Conv2d(kernel_size, kernel_size, (3, 3), padding='same')
        self.batchnorm2 = nn.BatchNorm2d(kernel_size)
                                 
        self.gelu = nn.GELU()
        
    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.batchnorm1(output)
        output = self.gelu(output)

        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = self.gelu(output)

        output = output + inputs
        
        return output
        

class Downsized_Convolution_Block(nn.Module):
    def __init__(self, input_kernel_size, output_kernel_size):
        super(Downsized_Convolution_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(input_kernel_size, output_kernel_size, (3, 3), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(output_kernel_size)
        
        self.conv2 = nn.Conv2d(output_kernel_size, output_kernel_size, (3, 3), padding='same')
        self.batchnorm2 = nn.BatchNorm2d(output_kernel_size)
        
        self.maxpool = nn. MaxPool2d((2, 2))
        self.gelu = nn.GELU()
        
    def forward(self, inputs):
        output = self.maxpool(inputs)
        
        output = self.conv1(output)
        output = self.batchnorm1(output)
        output = self.gelu(output)
        
        output = self.conv2(output)
        output = self.batchnorm2(output)
        output = self.gelu(output)
        
        return output
        
class AptaTrans(nn.Module):
    def __init__(self, apta_encoder, prot_encoder, n_apta_vocabs, n_prot_vocabs, dropout, apta_max_len, prot_max_len):
        super(AptaTrans, self).__init__()
        
        #hyperparameters
        self.n_apta_vocabs = n_apta_vocabs
        self.n_prot_vocabs = n_prot_vocabs
        self.apta_max_len = apta_max_len
        self.prot_max_len = prot_max_len
        self.dropout = dropout
        
        self.apta_encoder = apta_encoder
        self.prot_encoder = prot_encoder
        
        self.kernel_size = 64
        
        self.batchnorm_fm = nn.BatchNorm2d(1)

        self.conv = nn.Conv2d(1, self.kernel_size, (3, 3))
        self.batchnorm = nn.BatchNorm2d(self.kernel_size)
        self.gelu = nn.GELU()
        
        self.maxpool = nn.MaxPool2d((2,2))
        
        self.conv64_1 = Convolution_Block(64)
        self.conv64_2 = Convolution_Block(64)
        self.conv64_3 = Convolution_Block(64)
        
        self.dconv128 = Downsized_Convolution_Block(64, 128)
        self.conv128_1 = Convolution_Block(128)
        self.conv128_2 = Convolution_Block(128)

        self.dconv256 = Downsized_Convolution_Block(128, 256)
        self.conv256_1 = Convolution_Block(256)
        self.conv256_2 = Convolution_Block(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 34 * 108, 128) #128, 136, 432 #68 * 216 * 256 # 256 * 34 * 108 #256, 68, 255
    
        self.fc1 = nn.Linear(128, 1)
        
    def forward(self, apta, prot):
        apta = self.apta_encoder(apta)
        
        prot = self.prot_encoder(prot)
        prot = torch.transpose(prot, 1, 2)
    
        output = torch.bmm(apta, prot)
        output = torch.unsqueeze(output, 1)
        output = self.batchnorm_fm(output)
        
        output = self.conv(output)
        output = self.batchnorm(output)
        output = self.gelu(output)

        output = self.conv64_1(output)
        output = self.conv64_2(output)
        output = self.conv64_3(output)

        output = self.dconv128(output)
        output = self.conv128_1(output)
        output = self.conv128_2(output)

        output = self.dconv256(output)
        output = self.conv256_1(output)
        output = self.conv256_2(output)

        output = self.maxpool(output)
       # print(output.shape)
        output = self.flatten(output)
                                 
        output = self.fc(output)
        output = self.gelu(output)
        output = self.fc1(output)
        
        output = torch.sigmoid(output)
        
        return output
    
    def generate_interaction_map(self, apta, prot):
        with torch.no_grad():
            apta = self.apta_encoder(apta)
            prot = self.prot_encoder(prot)
            prot = torch.transpose(prot, 1, 2)

            interaction_map = torch.bmm(apta, prot)

            interaction_map = torch.unsqueeze(interaction_map, 1)
            interaction_map = self.batchnorm_fm(interaction_map)

        return interaction_map

    def conv_block_proba(self, interaction_map):
        with torch.no_grad():
            output = torch.tensor(interaction_map).float().to('cuda:0')
            output = torch.unsqueeze(output, 1)
            output = torch.mean(output, 4)

            output = self.conv(output)
            output = self.batchnorm(output)
            output = self.gelu(output)

            output = self.conv64_1(output)
            output = self.conv64_2(output)
            output = self.conv64_3(output)

            output = self.dconv128(output)
            output = self.conv128_1(output)
            output = self.conv128_2(output)

            output = self.dconv256(output)
            output = self.conv256_1(output)
            output = self.conv256_2(output)

            output = self.maxpool(output)
            output = self.flatten(output)
                                    
            output = self.fc(output)
            output = self.gelu(output)
            output = self.fc1(output)
            
            output = torch.sigmoid(output)
            output = np.array([[1 - o[0], o[0]]for o in output.clone().detach().cpu().numpy()])
            
            return output

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
        
        # aug_apta.append(a) 
        # aug_prot.append(p[::-1])
        # aug_y.append(y)
        
        # aug_apta.append(a[::-1]) 
        # aug_prot.append(p[::-1])
        # aug_y.append(y)
        
    return np.array(aug_apta), np.array(aug_prot), np.array(aug_y)


