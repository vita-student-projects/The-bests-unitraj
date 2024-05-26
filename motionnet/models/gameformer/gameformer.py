from .gameformer_modules import *

from motionnet.models.ptr.ptr import PTR
from motionnet.datasets import common_utils
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
        self.N_levels = config['num_levels']
        self.tx_hidden_size = config['tx_hidden_size']

        # ================= GameFormer Encoder =================
        self.encoder = PTR_Encoder(self.config, k_attr=k_attr, map_attr=map_attr)

        current_state_dict = self.encoder.state_dict()
        ptr_state_dict = torch.load(config['ptr_path'], map_location='cpu')['state_dict']
        for name, param in current_state_dict.items():
            if name in ptr_state_dict and param.size() == ptr_state_dict[name].size():
                current_state_dict[name] = ptr_state_dict[name]
            else:
                print(f"Skipped {name} from ptr.")


        # Load the updated state dict back into the model
        self.encoder.load_state_dict(current_state_dict, strict=False)

        # ================= GameFormer Encoder =================
        self.ego_encoder = nn.LSTM(self.k_attr, self.d_k, 2, batch_first=True)
        self.agent_encoder = nn.LSTM(self.k_attr, self.d_k, 2, batch_first=True)

        # ================= GameFormer Decoder =================
        self.initial_stage = InitialDecoder(self.c, 16, self.T, self.num_heads,d_k=self.d_k)
        self.future_encoder = FutureEncoder(k_attr=k_attr, d_k=self.d_k)
        self.interaction_stage = nn.ModuleList([InteractionDecoder(self.future_encoder, self.T,
                                                                   self.num_heads, self.d_k, self.dropout) for _ in range(self.N_levels)])

        self.criterion = Criterion(self.config)

  

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
        N = inputs['agents_in'].shape[2]
        agents= inputs['agents_in'][...,:2]

        encoded_ego = self.ego_encoder(agents[:,:,0])[0][:,-1] #B, H
        encoded_neighbors = [self.agent_encoder(agents[:,:,n])[0][:,-1] for n in range(1,N)]
        encoded_agents = torch.stack([encoded_ego]+encoded_neighbors,dim=1) #B, N, H
        
        out_dists = []
        mode_probs = []
        encodings = []
        masks = []

        agents_mask = torch.eq(agents[:,-1].sum(-1), 0) # B, N
        agents_mask[:,0] = False
        # agents_mask2 = torch.eq(agents.sum(2).sum(-1), 0)
        # breakpoint()
        test = agents_mask.any(dim=0)
        true_indices = torch.nonzero(test, as_tuple=True)[0]
        N_inter = true_indices[0].item() if true_indices.numel() > 2 else 2
        # breakpoint()
        N_inter = np.min((N_inter,3))
        for n in range(N_inter):
            ptr_inputs = {}
            ptr_inputs['agents_in'] = inputs['agents_in'].clone()
            ptr_inputs['roads'] = inputs['roads'].clone()

            center = ptr_inputs['agents_in'][:,-1,n,:2].clone()
            heading = ptr_inputs['agents_in'][:,-1,n,2].clone()
            
            ptr_inputs['agents_in'][...,:2] =  ptr_inputs['agents_in'][...,:2] - center.view(-1,1,1,2)
            ptr_inputs['agents_in'][...,:2] = self.rotate_points(ptr_inputs['agents_in'][...,:2], torch.tensor(np.pi/2)-heading)
            ptr_inputs['roads'][...,:2] = ptr_inputs['roads'][...,:2] - center.view(-1,1,1,2)
            ptr_inputs['roads'][...,:2] = self.rotate_points(ptr_inputs['roads'][...,:2], torch.tensor(np.pi/2)-heading)
            ptr_inputs['ego_in'] = ptr_inputs['agents_in'][:, :, n]      

            ptr_output = self.encoder(ptr_inputs)

            encodings.append(torch.cat([encoded_agents, ptr_output['map_features']], dim=1))
            masks.append(torch.cat([agents_mask, ptr_output['road_segs_masks']], dim=1))


        encodings = torch.stack(encodings, dim=1)
        masks = torch.stack(masks, dim=1)

        current_states = agents[:,-1]

        results = [self.initial_stage(i, current_states[:,i], encodings[:,i], masks[:,i]) for i in range(N_inter)]
        last_content = torch.stack([result[0] for result in results], dim=1)# [B, N, c, H]
        last_level = torch.stack([result[1] for result in results], dim=1) 
        last_scores = torch.stack([result[2] for result in results], dim=1)# [B, N, c, H]

        output = {}
        output['level_0_score'] = last_scores
        output['level_0_trajectory'], output['level_0_probability'] = self.get_probs(last_level, last_scores)
        

        #level k interaction
        for k in range(1, self.N_levels+1):
            interaction_decoder = self.interaction_stage[k-1]
            results = [interaction_decoder(i, current_states[:, :N_inter], last_level, last_scores, \
                        last_content[:, i], encodings[:,i], masks[:,i]) for i in range(N_inter)]
            last_content = torch.stack([result[0] for result in results], dim=1)
            last_level = torch.stack([result[1] for result in results], dim=1)
            last_scores = torch.stack([result[2] for result in results], dim=1) 

            output[f'level_{k}_score'] = last_scores
            output[f'level_{k}_trajectory'], output[f'level_{k}_probability'] = self.get_probs(last_level, last_scores)

        trajectories = []
        scores = []
        B = output[f'level_0_trajectory'].shape[0]
        for i in range(self.N_levels+1):
            trajectories.append(output[f'level_{i}_trajectory'][:,0])
            scores.append(output[f'level_{i}_score'][:,0])
        
        trajectories = torch.stack(trajectories, dim=2)
        scores = torch.stack(scores, dim=1)
        mean_scores = torch.mean(scores, dim=2)
        
        top_level = torch.topk(mean_scores, 1)[1].squeeze(1)
        # breakpoint()
        top_trajectories = torch.gather(trajectories, 2, top_level.view(-1,1,1,1,1).repeat(1,self.c,1,self.T,5)).squeeze(2)
        top_scores = torch.gather(scores, 1, top_level.view(-1,1,1).repeat(1,1,self.c)).squeeze(1)

        #trajectories[:,:,top_level].squeeze(2)#torch.gather(trajectories, 1, top_indices.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.T,5))
        # top_scores = scores[top_level]
        output[f'top_trajectory'] = top_trajectories
        output[f'top_score'] = top_scores

        output['predicted_trajectory'] = top_trajectories
        output['predicted_probability'] = F.softmax(top_scores, dim=-1)

        return output

    def forward(self, batch, batch_idx):
        model_input = {}
        inputs = batch['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'],inputs['obj_trajs_mask'] ,inputs['map_polylines']

        if torch.any(inputs['track_index_to_predict'] != 0):
            breakpoint()

        agents_heading = torch.atan2(agents_in[...,34], agents_in[...,33]).unsqueeze(-1)
        # breakpoint()
        agents_in = torch.cat([agents_in[...,:2],agents_heading, agents_mask.unsqueeze(-1)],dim=-1).transpose(1,2)
        roads = torch.cat([inputs['map_polylines'][...,:2],inputs['map_polylines_mask'].unsqueeze(-1)],dim=-1)
        # model_input['ego_in'] = ego_in
        model_input['agents_in'] = agents_in
        model_input['roads'] = roads
        output = self._forward(model_input)

        loss = self.get_loss(batch, output) #, bests_levels

        # # cheating
        # top_level = bests_levels[:,0]
        # top_trajectories = torch.gather(trajectories, 2, top_level.view(-1,1,1,1,1).repeat(1,self.c,1,self.T,5)).squeeze(2)
        # top_scores = torch.gather(scores, 1, top_level.view(-1,1,1).repeat(1,1,self.c)).squeeze(1)

        # output['top_trajectory'] = top_trajectories
        # output['top_score'] = top_scores


        # output['predicted_trajectory'] = output['top_trajectory'] #output[f'level_{self.N_levels}_trajectory'][:,0]
        # output['predicted_probability'] = F.softmax(output['top_score'], dim=-1) #output[f'level_{self.N_levels}_probability'][:,0]

        return output, loss
    
    def get_probs(self, last_level, last_scores):
        pred_obs = last_level # [B, c, T, 5]
        mode_probs = F.softmax(last_scores, dim=-1) # [B, c]

        x_mean = pred_obs[..., 0]
        y_mean = pred_obs[..., 1]
        x_sigma = F.softplus(pred_obs[..., 2]) + 0.01
        y_sigma = F.softplus(pred_obs[..., 3]) + 0.01
        rho = torch.tanh(pred_obs[..., 4]) * 0.9  # for stability
        out_dists = torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=-1)
        return out_dists, mode_probs

    def get_loss(self, batch, prediction):
        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['obj_trajs_future_state'][...,:2],inputs['obj_trajs_future_mask'].unsqueeze(-1)],dim=-1)
        loss = self.criterion(prediction, ground_truth, self.N_levels, inputs['track_index_to_predict'])
        return loss

    def configure_optimizers(self):
        all_params = list(self.parameters())
        ptr_params = list(self.encoder.parameters())
        main_params = list(set(all_params) - set(ptr_params))
        # optimizer = optim.Adam([{'params':all_params, 'lr':self.config['learning_rate'],'eps':0.0001}])
        optimizer = optim.Adam([{'params':main_params, 'lr':self.config['learning_rate'],'eps':0.0001},
                                {'params':ptr_params, 'lr':0.0, 'eps':0.0001}])
        
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                           verbose=True)
        return [optimizer], [scheduler]

    def rotate_points(self, trajs, theta):        
        # Create the rotation matrices for each angle in the batch
        cos, sin = torch.cos(theta), torch.sin(theta)
        rotation_matrices = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin,  cos], dim=-1)
        ], dim=-2)  # rotation_matrices will have shape [batch_size, 2, 2]
        
        B,T,N = trajs.shape[:3]
        # points should have shape [batch_size, ..., 2], where the last dimension are the xy coordinates
        # Apply the rotation matrix to points
        # We use matmul which handles batched matrix multiplication
        rotated_points = torch.matmul(trajs.reshape(B,-1,2),rotation_matrices).reshape(B,T,N,2)
        
        return rotated_points


class PTR_Encoder(BaseModel):
    def __init__(self, config, k_attr=2, map_attr=2):

        super(PTR_Encoder, self).__init__(config)

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

        # ============================== PTR ENCODER ==============================
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

        # ============================== MAP ENCODER ==========================
        self.map_encoder = MapEncoderPts(d_k=self.d_k, map_attr=self.map_attr, dropout=self.dropout)
        self.map_attn_layers = nn.MultiheadAttention(self.d_k, num_heads=self.num_heads, dropout=0.3)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self.d_k, dropout=0.0, max_len=self.past)

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
        Gets agents embeddings and agents mask, and applies the temporal attention layer per agent.
        Make sure to apply the agent mask in the layer function (you could use src_key_padding_mask argument).
        Also don't forget to use positional encoding.
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        ######################## Your code here ########################
        # Apply positional encoding
        T,B,N,H = agents_emb.shape

        agents_emb = self.pos_encoder(agents_emb.reshape(T,-1,H)) # Shape: (T, B*N, H)

        agent_masks = agent_masks.permute(0,2,1).reshape(-1,T) # Shape: (B*N, T)
        agent_masks[:,-1][agent_masks.all(dim=1)] = False
        

        agents_emb = layer(agents_emb, src_key_padding_mask=agent_masks).reshape(T,B,N,H)

        # pdb.set_trace()
        ################################################################
        return agents_emb

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        Gets agents embeddings and agents mask, and applies the social attention layer per time step.
        Make sure to apply the agent mask in the layer function (you could use src_key_padding_mask argument).
        You don't need to use positional encoding here.
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        ######################## Your code here ########################
        # Apply social attention layer
        T,B,N,H = agents_emb.shape

        agents_emb = agents_emb.permute(2,1,0,3).reshape(N,-1,H) #Shape: (N, B*T, H)
        agent_masks = agent_masks.reshape(-1,N)

        agents_emb = layer(agents_emb, src_key_padding_mask=agent_masks).reshape(N,B,T,H).permute(2,1,0,3)
        ################################################################
    
        return agents_emb    

    def forward(self, inputs):
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
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)  # [B, T, N, k_attr]

        # encode each agent's dynamic state using a linear layer (k_attr --> d_k)
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3)  # T, B, N, H

        ######################## Your code here ########################
        # Apply temporal attention layers and then the social attention layers on agents_emb, each for L_enc times.
        for i in range(self.L_enc):
            agents_attn = agents_emb
            agents_attn = self.temporal_attn_fn(agents_attn, opps_masks, self.temporal_attn_layers[i])
            agents_attn = self.social_attn_fn(agents_attn, opps_masks, self.social_attn_layers[i])
            agents_emb = self.residual*agents_emb + (1-self.residual)*agents_attn
        ################################################################

        ego_soctemp_emb = agents_emb[:, :, 0]  # take ego-agent encodings only.

        orig_map_features, orig_road_segs_masks = self.map_encoder(roads, ego_soctemp_emb)

        # return  [c, T, B, 5], [B, c]
        output = {}
        output['map_features'] = orig_map_features.transpose(0, 1)
        output['road_segs_masks'] = orig_road_segs_masks

        return output