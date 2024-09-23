import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
import math
import numpy as np
from sklearn.metrics import f1_score

class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        dim_hidden = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=8,
        dim_attn=16,
        dropout=0.,
    ):
        super().__init__()

        dim_inner = dim_attn * n_heads
        self.n_heads = n_heads

        self.scale = dim_attn ** -0.5

        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_k = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)

        self.to_out = nn.Linear(dim_inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n_heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.dropout(self.to_out(out))

class Encoder(nn.Module):
    def __init__(self, dim, n_heads, dim_attn, mult_ff, dropout_ff, dropout_attn):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, n_heads, dim_attn, dropout_attn)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult_ff, dropout_ff)

    def forward(self, x):
        out_attn = self.attention(x)
        x = self.norm_attn(out_attn + x)
        out_ff = self.ff(x)
        x = self.norm_ff(out_ff + x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return x + self.pe[:x.size(0)]

class Encoders(nn.Module):
    def __init__(self, n_vocabs, dim=128, n_layers=6, mult_ff=2, n_heads=8, dropout=0.3, max_len=512):
        super(Encoders, self).__init__()
        self.n_vocabs = n_vocabs
        self.n_layers = n_layers
        self.dim = dim
        self.Embedding = nn.Embedding(num_embeddings=n_vocabs, embedding_dim=dim, padding_idx=0)
        self.PositionalEncoding = PositionalEncoding(dim, max_len=max_len)
        self.encoders = nn.ModuleList([
            Encoder(dim=dim, n_heads=n_heads, dim_attn=dim // n_heads, mult_ff=mult_ff, dropout_ff=dropout, dropout_attn=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        padding_mask = self.create_padding_mask(x)
        x = self.Embedding(x)
        x = self.PositionalEncoding(x)
        x[padding_mask] = torch.zeros(self.dim).to(x.device)
        for encoder in self.encoders:
            x = encoder(x)
        x[padding_mask] = torch.zeros(self.dim).to(x.device)    
        return x

    def create_padding_mask(self, x):
        return x == 0

class Token_Predictor(nn.Module):
    def __init__(self, n_vocabs, n_target_vocabs, dim):
        super(Token_Predictor, self).__init__()
        self.fc1_mlm = nn.Linear(dim, n_vocabs)
        self.fc1_ssp = nn.Linear(dim, n_target_vocabs)

    def forward(self, x_mlm, x_ssp):
        output_mlm = self.fc1_mlm(x_mlm)
        output_ssp = self.fc1_ssp(x_ssp)
        return output_mlm, output_ssp

class To_IteractionMap(nn.Module):
    def __init__(self):
        super(To_IteractionMap, self).__init__()
        self.batchnorm = nn.BatchNorm2d(1)

    def forward(self, apta, prot):
        prot = torch.transpose(prot, 1, 2)
        interaction_map = torch.bmm(apta, prot)
        out = torch.unsqueeze(interaction_map, 1)
        out = self.batchnorm(out)
        return out

class Predictor(nn.Module):
    def __init__(self, channel_size=64):
        super(Predictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(channel_size*4, channel_size*2),
            nn.GELU(),
            nn.Linear(channel_size*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.predictor(x)
        return out
    
class Convolution_Block(nn.Module):
    def __init__(self, channels):
        super(Convolution_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 3), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding='same')
        self.batchnorm2 = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.gelu(out)
        out = out + x
        return out

class Downsized_Convolution_Block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Downsized_Convolution_Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, (3, 3), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, (3, 3), padding='same')
        self.batchnorm2 = nn.BatchNorm2d(output_channels)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.maxpool(x)
        out = self.conv1(out)
        out = self.batchnorm1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.gelu(out)
        return out

class CONVBlocks(nn.Module):
    def __init__(self, out_channels=64):
        super(CONVBlocks, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(1, self.out_channels, (3, 3))
                
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        self.gelu = nn.GELU()
        self.conv64 = nn.Sequential(
            Convolution_Block(self.out_channels),
            Convolution_Block(self.out_channels),
            Convolution_Block(self.out_channels)
        )
        self.dconv128 = Downsized_Convolution_Block(self.out_channels, self.out_channels * 2)
        self.conv128 = nn.Sequential(
            Convolution_Block(self.out_channels * 2),
            Convolution_Block(self.out_channels * 2)
        )
        self.dconv256 = Downsized_Convolution_Block(self.out_channels * 2, self.out_channels * 4)
        self.conv256 = nn.Sequential(
            Convolution_Block(self.out_channels * 4),
            Convolution_Block(self.out_channels * 4)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
    def forward(self, x):
        output = torch.squeeze(x, dim=2)  # Remove the extra dimension
        output = self.conv(output)
        output = self.batchnorm(output)
        output = self.gelu(output)
        output = self.conv64(output)

        output = self.dconv128(output)
        output = self.conv128(output)

        output = self.dconv256(output)
        output = self.conv256(output)

        output = self.avgpool(output)
        output = self.flatten(output)
        return output

class AptaTransWrapper(nn.Module):
    def __init__(self, encoder_apta, encoder_prot, to_interaction_map, convblocks, predictor):
        super(AptaTransWrapper, self).__init__()
        self.encoder_apta = encoder_apta
        self.encoder_prot = encoder_prot
        self.to_interaction_map = to_interaction_map
        self.convblocks = convblocks
        self.predictor = predictor

    def forward(self, apta, prot):
        out_apta = self.encoder_apta(apta)
        out_prot = self.encoder_prot(prot)
        interaction_map = self.to_interaction_map(out_apta, out_prot)
        out = self.convblocks(interaction_map)
        out = self.predictor(out)
        return out

    def generate_interaction_map(self, apta, prot):
        with torch.no_grad():
            out_apta = self.encoder_apta(apta)
            out_prot = self.encoder_prot(prot)
            out_prot = torch.transpose(out_prot, 1, 2)
            interaction_map = torch.bmm(out_apta, out_prot)
            interaction_map = torch.unsqueeze(interaction_map, 1)
            # interaction_map = self.batchnorm_fm(interaction_map)
        return interaction_map

    def conv_block_proba(self, interaction_map):
        with torch.no_grad():
            print(interaction_map.device)
            out = torch.tensor(interaction_map).float().to('cuda:0')
            output = torch.unsqueeze(out, 1)
            output = torch.mean(output, 4)
            out = self.convblocks(output)
            out = self.predictor(out)
            out = np.array([[1 - o[0], o[0]] for o in out.clone().detach().cpu().numpy()])
            return out

def find_opt_threshold(target, pred):
    result = 0
    best = 0
    for i in range(0, 1000):
        pred_threshold = np.where(pred > i / 1000, 1, 0)
        now = f1_score(target, pred_threshold)
        if now > best:
            result = i / 1000
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