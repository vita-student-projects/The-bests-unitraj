from .gameformer_modules import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from motionnet.models.base_model.base_model import BaseModel
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim

    
class GameFormer(BaseModel):
    def __init__(self, config, k_attr=2, map_attr=2):

        super(GameFormer, self).__init__(config)

        self.config = config
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.T = config['future_len']
        self.past = config['past_len']
        self.fisher_information = None
        self.map_attr = map_attr
        self.k_attr = k_attr
        self.d_k = config['hidden_size']
        self.c = config['num_modes']

        self.L_enc = config['num_encoder_layers']
        self.dropout = config['dropout']
        self.residual = config['residual']
        self.num_heads = config['tx_num_heads']
        self.L_dec = config['num_decoder_layers']
        self.tx_hidden_size = config['tx_hidden_size']


        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))

        # ============================== GameFormer ENCODER ==============================
        self.ego_encoder = nn.LSTM(self.d_k, self.d_k, 2, batch_first=True)
        self.agent_encoder = nn.LSTM(self.d_k, self.d_k, 2, batch_first=True)
        attention_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dim_feedforward=self.d_k*4,
                                                     activation=F.gelu, dropout=self.dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, self.L_enc, enable_nested_tensor=False)

        # ============================== MAP ENCODER ==========================
        self.map_encoder = MapEncoderPts(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)

        # ========================== GameFormer DECODER ===========================
        self.future_encoder = FutureEncoder()
        self.initial_stage = InitialDecoder(self.c, 16, self.T, self.d_k)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(self.future_encoder, self.T,
                                                                   self.num_heads, self.d_k, self.dropout) for _ in range(self.L_dec)])  

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past)

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.c, 1).view(ego.shape[0] * self.c, -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks 
   
    def _forward(self, inputs):
        '''
        :param ego_in: [B, T_obs, k_attr+1] with last values being the existence mask.
        :param agents_in: [B, T_obs, M-1, k_attr+1] with last values being the existence mask.
        :param roads: [B, S, P, map_attr+1] representing the road network if self.use_map_lanes or
                      [B, 3, 128, 128] image representing the road network if self.use_map_img or
                      [B, 1, 1] if self.use_map_lanes and self.use_map_img are False.
        :return:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of
                                        Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        ego_in, agents_in, roads = inputs['ego_in'], inputs['agents_in'], inputs['roads']

        # Encode all input observations (k_attr --> d_k)
        ego, neighbors, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents = torch.cat((ego.unsqueeze(2), neighbors), dim=2)  # [B, T, N, k_attr]

        # encode each agent's dynamic state using a linear layer (k_attr --> d_k)
        agents_emb = self.agents_dynamic_encoder(agents)  # B, T, N, H  

        # # positional encoding
        # B,T,N,H = agents_emb.size() 
        # agents_emb = self.pos_encoder(agents_emb.permute(1,0,2,3).reshape(T,-1,H)) # T, B*N, H 
        # agents_emb = agents_emb.reshape(T,B,N,H).permute(1,0,2,3) # B, T, N, H    

        N = agents_emb.size(2)

        encoded_ego = self.ego_encoder(agents_emb[:,:,0])[0][:,-1] #B, H
        encoded_neighbors = [self.agent_encoder(agents_emb[:,:,i])[0][:,-1] for i in range(1,N)]
        
        encoded_agents = torch.stack([encoded_ego] + encoded_neighbors, dim=1)# [B, N, H]
        agents_masks = opps_masks[:,-1] # [B, N] 

        agents_map_features = []
        agents_road_segs_masks = []
        masks = []
        encodings = []
        agents_emb = agents_emb.permute(1,0,2,3) #T,B,N,H

        for n in range(N):
            agent_map_features, agent_road_segs_masks = self.map_encoder(roads, agents_emb[:,:,n])
            agent_map_features = agent_map_features.permute(1,0,2) # [B,S,H]
            agent_road_segs_masks = agent_road_segs_masks# [B,S]

            agents_map_features.append(agent_map_features)
            agents_road_segs_masks.append(agent_road_segs_masks)

            
            fusion_input = torch.cat([encoded_agents, agent_map_features], dim=1) # [B,N+S,H]
            mask = torch.cat([agents_masks, agent_road_segs_masks], dim=1)
            masks.append(mask)
            encoding = fusion_input #self.fusion_encoder(fusion_input, src_key_padding_mask=mask) # [B,N+S,H]
            encodings.append(encoding)
        
        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)

        # agents [B, T, N, k_attr]
        # encodings [B, N, N+S, H]
        # masks [B, N, N+S]
        # ========================== GameFormer DECODER ===========================
        current_states = agents[:,-1] # [B, N, k_attr]
    

        decoder_outputs = {}
        # level 0
        results = [self.initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([result[0] for result in results], dim=1)# [B, N, c, H]
        last_level = torch.stack([result[1] for result in results], dim=1)# [B, N, c, T, 5]
        last_scores = torch.stack([result[2] for result in results], dim=1)# [B, N, c]
        decoder_outputs['level_0_interactions'] = last_level
        decoder_outputs['level_0_scores'] = last_scores

        # level k reasoning
        for k in range(1, self.L_dec+1):
            interaction_decoder = self.interaction_stage[k-1]
            results = [interaction_decoder(i, current_states[:, :N], last_level, last_scores, \
                        last_content[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1)
            decoder_outputs[f'level_{k}_interactions'] = last_level
            decoder_outputs[f'level_{k}_scores'] = last_scores

        pred_obs = last_level[:,0] # [B, c, T, 5]
        mode_probs = F.softmax(last_scores[:,0], dim=1) # [B, c]

        x_mean = pred_obs[..., 0]
        y_mean = pred_obs[..., 1]
        x_sigma = F.softplus(pred_obs[..., 2]) + 0.01
        y_sigma = F.softplus(pred_obs[..., 3]) + 0.01
        rho = torch.tanh(pred_obs[..., 4]) * 0.9  # for stability


        out_dists = torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=3)

        output = {}
        output['predicted_probability'] = mode_probs #[B, c]
        output['predicted_trajectory'] = out_dists #[B, c, T, 5]        

        if len(np.argwhere(np.isnan(out_dists.detach().cpu().numpy()))) > 1:
            breakpoint()
        return output

    def forward(self, batch, batch_idx):
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'],inputs['obj_trajs_mask'] ,inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1,1,1,1).repeat(1,1,*agents_in.shape[-2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1,1,1).repeat(1,1,agents_mask.shape[-1])).squeeze(1)
        agents_in = torch.cat([agents_in[...,:2],agents_mask.unsqueeze(-1)],dim=-1)
        agents_in = agents_in.transpose(1,2)
        ego_in = torch.cat([ego_in[...,:2],ego_mask.unsqueeze(-1)],dim=-1)
        roads = torch.cat([inputs['map_polylines'][...,:2],inputs['map_polylines_mask'].unsqueeze(-1)],dim=-1)
        model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        output = self._forward(model_input)

        loss = self.get_loss(batch, output)

        return output, loss

    def get_loss(self, batch, prediction):
        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['center_gt_trajs'][...,:2],inputs['center_gt_trajs_mask'].unsqueeze(-1)],dim=-1)
        loss = self.criterion(prediction, ground_truth,inputs['center_gt_final_valid_idx'])
        return loss



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.config['learning_rate'],eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                           verbose=True)
        return [optimizer], [scheduler]


