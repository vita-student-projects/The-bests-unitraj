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
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_k, nhead=self.num_heads, dropout=self.dropout,
                                                          dim_feedforward=self.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # self.ego_encoder = nn.LSTM(self.d_k, self.d_k, 2, batch_first=True)
        # self.agent_encoder = nn.LSTM(self.d_k, self.d_k, 2, batch_first=True)
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

        # ============================== PTR DECODER ==============================
        # self.Q = nn.Parameter(torch.Tensor(self.T, 1, self.c, self.d_k), requires_grad=True)
        # nn.init.xavier_uniform_(self.Q)

        # self.tx_decoder = []
        # for _ in range(self.L_dec):
        #     self.tx_decoder.append(nn.TransformerDecoderLayer(d_model=self.d_k, nhead=self.num_heads,
        #                                                       dropout=self.dropout,
        #                                                       dim_feedforward=self.tx_hidden_size))
        # self.tx_decoder = nn.ModuleList(self.tx_decoder)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past)

        # ============================== OUTPUT MODEL ==============================
        # self.output_model = OutputModel(d_k=self.d_k)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        # self.P = nn.Parameter(torch.Tensor(self.c, 1, self.d_k), requires_grad=True)
        # nn.init.xavier_uniform_(self.P)

        # self.mode_map_attn = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads)

        # self.prob_decoder = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=self.dropout)
        # self.prob_predictor = init_(nn.Linear(self.d_k, 1))

        self.criterion = Criterion(self.config)

        self.fisher_information = None
        self.optimal_params = None

    # def generate_decoder_mask(self, seq_len, device):
    #     ''' For masking out the subsequent info. '''
    #     subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
    #     return subsequent_mask
    
    # def segment_map(self, map, map_encoding):
    #     stride = 10
    #     B, S, P, d_k = map_encoding.shape

    #     # segment map
    #     map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, stride))
    #     map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, d_k)

    #     # segment mask
    #     map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, S, P//stride, P//(P//stride))
    #     map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

    #     return map_encoding, map_mask

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
    
    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T,B,N,H = agents_emb.shape

        agents_emb = self.pos_encoder(agents_emb.reshape(T,-1,H)) # Shape: (T, B*N, H)

        agent_masks = agent_masks.permute(0,2,1).reshape(-1,T) # Shape: (B*N, T)
        agent_masks[:,-1][agent_masks.all(dim=1)] = False
        

        agents_emb = layer(agents_emb, src_key_padding_mask=agent_masks).reshape(T,B,N,H)
        return agents_emb

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        # Apply social attention layer
        T,B,N,H = agents_emb.shape

        agents_emb = agents_emb.permute(2,1,0,3).reshape(N,-1,H) #Shape: (N, B*T, H)
        agent_masks = agent_masks.reshape(-1,N)

        agents_emb = layer(agents_emb, src_key_padding_mask=agent_masks).reshape(N,B,T,H).permute(2,1,0,3)
    
        return agents_emb  

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

        B = ego_in.size(0)
        # Encode all input observations (k_attr --> d_k)
        ego, neighbors, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents = torch.cat((ego.unsqueeze(2), neighbors), dim=2)  # [B, T, N, k_attr]

        # encode each agent's dynamic state using a linear layer (k_attr --> d_k)
        agents_emb = self.agents_dynamic_encoder(agents).permute(1,0,2,3)  # T, B, N, H
        ######################## Your code here ########################
        # Apply temporal attention layers and then the social attention layers on agents_emb, each for L_enc times.
        for i in range(self.L_enc):
            agents_attn = agents_emb
            agents_attn = self.temporal_attn_fn(agents_attn, opps_masks, self.temporal_attn_layers[i])
            agents_attn = self.social_attn_fn(agents_attn, opps_masks, self.social_attn_layers[i])
            agents_emb = self.residual*agents_emb + (1-self.residual)*agents_attn
        ################################################################
        encoded_agents = agents_emb[-1]# [B, N, H]
        agents_masks = opps_masks[:,-1] # [B, N]

        N = agents_emb.size(2)

        agents_map_features = []
        agents_road_segs_masks = []
        masks = []
        encodings = []

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

        # map_features = torch.stack(agents_map_features, dim=1).permute(2,1,0,3) # [B,N,S,H]
        # road_segs_masks = torch.stack(agents_road_segs_masks, dim=1) # [B,N,S]
        
        # outputs
        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)
        # encoder_outputs = {
        #     'actors': agents,
        #     'encodings': encodings,
        #     'masks': masks
        # }

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


