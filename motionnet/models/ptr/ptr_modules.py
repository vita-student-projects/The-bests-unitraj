import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import special
from torch.distributions import MultivariateNormal, Laplace

class MapEncoderCNN(nn.Module):
    '''
    Regular CNN encoder for road image.
    '''
    def __init__(self, d_k=64, dropout=0.1, c=10):
        super(MapEncoderCNN, self).__init__()
        self.dropout = dropout
        self.c = c
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        # MAP ENCODER
        fm_size = 7
        self.map_encoder = nn.Sequential(
            init_(nn.Conv2d(3, 32, kernel_size=4, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, fm_size*self.c, kernel_size=2, stride=2)), nn.ReLU(),
            nn.Dropout2d(p=self.dropout)
        )
        self.map_feats = nn.Sequential(
            init_(nn.Linear(7*7*fm_size, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
        )
        self.fisher_information = None
        self.optimal_params = None

    def forward(self, roads):
        '''
        :param roads: road image with size (B, 128, 128, 3)
        :return: road features, with one for every mode (B, c, d_k)
        '''
        B = roads.size(0)  # batch size
        return self.map_feats(self.map_encoder(roads).view(B, self.c, -1))


class MapEncoderPts(nn.Module):
    '''
    This class operates on the road lanes provided as a tensor with shape
    (B, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        self.map_attr = map_attr
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False  # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        '''
        :param roads: (B, S, P, k_attr+1)  where B is batch size, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, d_k) where T_obs is the observation horizon. THis tensor is obtained from
        PTR's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B*S, P, -1).permute(1, 0, 2)

        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        road_seg_emb = road_seg_emb2.view(B, S, -1)

        return road_seg_emb.permute(1, 0, 2), road_segment_mask


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    '''
    This class operates on the output of PTR's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution.
    '''
    def __init__(self, d_k=64):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, 5))
        )
        self.min_stdev = 0.01

    def forward(self, agent_decoder_state):
        T = agent_decoder_state.shape[0]
        BK = agent_decoder_state.shape[1]
        pred_obs = self.observation_model(agent_decoder_state.reshape(-1, self.d_k)).reshape(T, BK, -1)

        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)
    

class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config


    def forward(self, out, gt,center_gt_final_valid_idx):

        return self.nll_loss_multimodes(out, gt,center_gt_final_valid_idx)

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        cov = torch.zeros((B, T, 2, 2)).to(pred.device)
        cov[:, :, 0, 0] = sigma_x ** 2
        cov[:, :, 1, 1] = sigma_y ** 2
        cov[:, :, 0, 1] = rho * sigma_x * sigma_y
        cov[:, :, 1, 0] = rho * sigma_x * sigma_y

        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4])

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        biv_gauss_dist = self.get_Laplace_dist(pred)
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, output, data, center_gt_final_valid_idx):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        modes_pred = output['predicted_probability']
        pred = output['predicted_trajectory'].permute(1, 2, 0, 3)
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
        log_lik = np.zeros((batch_sz, modes))
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik[:, kk] = -nll.cpu().numpy()

        priors = modes_pred.detach().cpu().numpy()
        log_posterior_unnorm = log_lik + np.log(priors)
        log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
        post_pr = np.exp(log_posterior)
        post_pr = torch.tensor(post_pr).float().to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()
