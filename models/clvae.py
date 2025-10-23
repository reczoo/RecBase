import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers
from .RecBaseResidual import ResidualVectorQuantizer


class CLVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
        ):
        super(CLVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        
        self.epoch = 0

    def forward(self, x, level, use_sk=True):
        latent = self.encoder(x)
        x_q, rq_loss, indices = self.rq(latent,use_sk=use_sk,level=level)
        out = self.decoder(x_q)

        mu = latent.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        log_var = torch.var(latent, dim=1, keepdim=True)  # [batch_size, 1]

        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()


        return out, rq_loss, indices, kl_divergence

    @torch.no_grad()
    def get_rec_latent(self, x, use_sk=True):
        latent = self.encoder(x)
        x_q, rq_loss, indices = self.rq(latent,use_sk=use_sk,level=4)
        return x_q

    @torch.no_grad()
    def get_rec(self, x, use_sk=True):
        latent = self.encoder(x)
        x_q, rq_loss, indices = self.rq(latent,use_sk=use_sk,level=4)
        out = self.decoder(x_q)
        return out


    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        # import pdb;pdb.set_trace()
        x_e = self.encoder(xs) #[bs,4096] -> [bs,128]
        _, _, indices = self.rq(x_e, use_sk=use_sk, level=4)
        return indices

    def compute_loss(self, out, quant_loss, kl, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        # import pdb;pdb.set_trace()

        loss_total = loss_recon + self.quant_loss_weight * quant_loss# + 0.1 * kl

        return loss_total, loss_recon