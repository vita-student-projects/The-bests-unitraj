import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import special
from torch.distributions import MultivariateNormal, Laplace
import sys

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FutureEncoder(nn.Module):
    def __init__(self, k_attr=2, d_k = 128):
        super(FutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(k_attr, 64), nn.ReLU(), nn.Linear(64, d_k))
        self.type_emb = nn.Embedding(4, d_k, padding_idx=0)

    def state_process(self, trajs, current_states):
        c = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, c, -1)
        # xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        # dxy = torch.diff(xy, dim=-2)
        # v = dxy / 0.1
        # theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        # T = trajs.shape[3]
        # size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        trajs = torch.cat([trajs], dim=-1) # (x, y, heading, vx, vy, w, l, h)

        return trajs

    def forward(self, trajs, current_states):
        # trajs = self.state_process(trajs, current_states)
        trajs = self.mlp(trajs.detach())
        # type = self.type_emb(current_states[:, :, None, 8].int())
        output = torch.max(trajs, dim=-2).values
        output = output# + type

        return output

class GMMPredictor(nn.Module):
    def __init__(self, future_len, d_k=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(nn.Linear(d_k, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*5))
        self.score = nn.Sequential(nn.Linear(d_k, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 5) # mu_x, mu_y, log_sig_x, log_sig_y, rho
        score = self.score(input).squeeze(-1)

        return res, score


class SelfTransformer(nn.Module):
    def __init__(self, num_heads=8, d_k=128, dropout=0.1):
        super(SelfTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_k, num_heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_k)
        self.norm_2 = nn.LayerNorm(d_k)
        self.ffn = nn.Sequential(nn.Linear(d_k, d_k*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_k*4, d_k), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class CrossTransformer(nn.Module):
    def __init__(self, num_heads=8, d_k=128, dropout=0.1):
        super(CrossTransformer, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_k, num_heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_k)
        self.norm_2 = nn.LayerNorm(d_k)
        self.ffn = nn.Sequential(nn.Linear(d_k, d_k*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_k*4, d_k), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output


class InitialDecoder(nn.Module):
    def __init__(self, c, neighbors, future_len, num_heads=8, d_k=128, dropout=0.1):
        super(InitialDecoder, self).__init__()
        self.c = c
        self.multi_modal_query_embedding = nn.Embedding(self.c, d_k)
        self.agent_query_embedding = nn.Embedding(neighbors+1, d_k)
        self.query_encoder = CrossTransformer(num_heads, d_k, dropout)
        self.predictor = GMMPredictor(future_len, d_k)
        self.register_buffer('modal', torch.arange(self.c).long())
        self.register_buffer('agent', torch.arange(neighbors+1).long())

    def forward(self, id, current_state, encoding, mask):
        # get query
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        agent_query = self.agent_query_embedding(self.agent[id])
        multi_modal_agent_query = multi_modal_query + agent_query[None, :]
        query = encoding[:, None, id] + multi_modal_agent_query

        # decode trajectories
        query_content = self.query_encoder(query, encoding, encoding, mask)
        predictions, scores = self.predictor(query_content)

        # post process
        predictions[..., :2] += current_state[:, None, None, :2]

        return query_content, predictions, scores
    
    
class InteractionDecoder(nn.Module):
    def __init__(self, future_encoder, future_len, num_heads=8, d_k=128, dropout=0.1):
        super(InteractionDecoder, self).__init__()
        self.interaction_encoder = SelfTransformer(num_heads, d_k, dropout)
        self.query_encoder = CrossTransformer(num_heads,d_k,dropout)
        self.future_encoder = future_encoder
        self.decoder = GMMPredictor(future_len, d_k)

    def forward(self, id, current_states, actors, scores, last_content, encoding, mask):
        B, N, M, T, _ = actors.shape
        
        # encoding the trajectories from the last level 
        multi_futures = self.future_encoder(actors[..., :2], current_states)
        futures = (multi_futures * scores.softmax(-1).unsqueeze(-1)).mean(dim=2) 
        # encoding the interaction using self-attention transformer   
        interaction = self.interaction_encoder(futures, mask[:, :N])

        # append the interaction encoding to the context encoding
        encoding = torch.cat([interaction, encoding], dim=1)
        mask = torch.cat([mask[:, :N], mask], dim=1).clone()
        mask[:, id] = True # mask the agent future itself from last level

        # decoding the trajectories from the current level
        query = last_content + multi_futures[:, id]
        query_content = self.query_encoder(query, encoding, encoding, mask)
        trajectories, scores = self.decoder(query_content)

        # post process
        trajectories[..., :2] += current_states[:, id, None, None, :2]

         
        # trajectories, scores = actors[:,id].clone(), scores[:,id].clone()
        # breakpoint()

        return query_content, trajectories, scores 

class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config


    def forward(self, out, gt, N_levels, center_gt_final_valid_idx):

        return self.nll_loss_multimodes(out, gt, N_levels, center_gt_final_valid_idx)

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

    def nll_loss_multimodes(self, output, data, N_levels, center_gt_final_valid_idx):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          pred: [K, T, B, 5]
          data: [B, T, 5]
          modes_pred: [B, K], prior prob over modes
          noise is optional
        """
        CURSOR_UP_ONE = '\x1b[1A'  # ANSI escape code to move cursor up by one line
        ERASE_LINE = '\x1b[2K'     # ANSI escape code to erase the line
        # breakpoint()
        N = output[f'level_{0}_probability'].size(1)
        for _ in range(0,(N_levels+1)*N+1):
            sys.stdout.write(CURSOR_UP_ONE)  # Move cursor up by one line
            sys.stdout.write(ERASE_LINE)     # Clear the line

        final_loss = 0.0
        for l in range(N_levels+1):
            for n in range(N):
                modes_pred = output[f'level_{l}_probability'][:,n]
                pred = output[f'level_{l}_trajectory'][:,n].permute(1, 2, 0, 3)
                mask = data[...,-1]

                entropy_weight = self.config['entropy_weight']
                kl_weight = self.config['kl_weight']
                use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

                modes = len(pred)
                nSteps, batch_sz, dim = pred[0].shape

                # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
                log_lik = np.zeros((batch_sz, modes))
                with torch.no_grad():
                    for kk in range(modes):
                        nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data[:,n], mask[:,n], rtn_loss=False)
                        log_lik[:, kk] = -nll.cpu().numpy()

                priors = modes_pred.detach().cpu().numpy()
                log_posterior_unnorm = log_lik + np.log(priors)
                log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
                post_pr = np.exp(log_posterior)
                post_pr = torch.tensor(post_pr).float().to(data.device)

                # Compute loss.
                loss = 0.0
                for kk in range(modes):
                    nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data[:,n], mask[:,n], rtn_loss=True) * post_pr[:, kk]
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
                    adefde_loss = self.l2_loss_fde(pred, data[:,n], mask[:,n], l)
                else:
                    adefde_loss = torch.tensor(0.0).to(data.device)

                # post_entropy
                # if n==0:
                final_loss = final_loss+(loss + kl_loss + adefde_loss)

        final_loss = final_loss/(N_levels+1)*N
        if np.isnan(final_loss.detach().cpu().numpy()):
            breakpoint()

        return final_loss

    def l2_loss_fde(self, pred, data, mask, level):

        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        
        min_ade, _ = ade_loss.min(dim=1)
        min_fde, _ = fde_loss.min(dim=1)
        print(f'\t level {level} : minADE = {min_ade.mean()} ||  minFDE = {min_fde.mean()}')
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()
