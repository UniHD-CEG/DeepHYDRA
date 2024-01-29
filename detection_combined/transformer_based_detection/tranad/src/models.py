import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder

from .dlutils import *
torch.manual_seed(1)


# Proposed Model (TKDE 21)
class TranAD_Basic(nn.Module):
    def __init__(self, feats):
        super(TranAD_Basic, self).__init__()
        self.name = 'TranAD_Basic'
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (TKDE 21)
class TranAD_Transformer(nn.Module):
    def __init__(self, feats):
        super(TranAD_Transformer, self).__init__()
        self.name = 'TranAD_Transformer'
        self.batch = 128
        self.n_feats = feats
        self.n_hidden = 8
        self.n_window = 10
        self.n = 2 * self.n_feats * self.n_window
        self.transformer_encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
        self.transformer_decoder1 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
        self.transformer_decoder2 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2).flatten(start_dim=1)
        tgt = self.transformer_encoder(src)
        return tgt

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.transformer_decoder1(self.encode(src, c, tgt))
        x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
        x1 = self.fcn(x1)
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.transformer_decoder2(self.encode(src, c, tgt))
        x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
        x2 = self.fcn(x2)
        return x1, x2

# Proposed Model + Self Conditioning + MAML (TKDE 21)
class TranAD_Adversarial(nn.Module):
    def __init__(self, feats):
        super(TranAD_Adversarial, self).__init__()
        self.name = 'TranAD_Adversarial'
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode_decode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x = self.encode_decode(src, c, tgt)
        # Phase 2 - With anomaly scores
        c = (x - src) ** 2
        x = self.encode_decode(src, c, tgt)
        return x

# Proposed Model + Adversarial + MAML (TKDE 21)
class TranAD_SelfConditioning(nn.Module):
    def __init__(self, feats):
        super(TranAD_SelfConditioning, self).__init__()
        self.name = 'TranAD_SelfConditioning'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (TKDE 21)
class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.batch = 128
        self.n_feats = feats

        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=64, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=64, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=64, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())


    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory
    

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats):
        super(DAGMM, self).__init__()
        self.name = 'DAGMM'
        self.lr = 0.0001
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = 5 # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden = None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64, device='cuda:0')\
                                                        if hidden is not None else hidden
        
        # if hidden is not None:
        #     hidden.to(device)

        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std
        ## Decoder
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5 # USAD w_size = 5
        self.n = self.n_feats*self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g.view(1,-1))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)