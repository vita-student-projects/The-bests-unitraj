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

        self.ptr = PTR(config)
        # Load the state dict from the checkpoint into the submodel
        state_dict = torch.load('/home/avray/dlav/dlav_data/best_ptr.ckpt')["state_dict"]
        self.ptr.load_state_dict(state_dict)

        # ================= GameFormer Encoder =================
        self.ego_encoder = nn.LSTM(self.k_attr, self.d_k, 2, batch_first=True)
        self.agent_encoder = nn.LSTM(self.k_attr, self.d_k, 2, batch_first=True)

        # ================= GameFormer Decoder =================
        # self.future_encoder = FutureEncoder(k_attr=k_attr, d_k=self.d_k)
        # self.interaction_stage = nn.ModuleList([InteractionDecoder(self.future_encoder, self.T,
        #                                                            self.num_heads, self.d_k, self.dropout) for _ in range(self.N_levels)])

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

        encoded_ego = self.ego_encoder(agents[:,:,0])[0][-1] #B, H
        encoded_neighbors = [self.agent_encoder(agents[:,:,n])[0][-1] for n in range(1,N)]
        encoded_agents = torch.stack([encoded_ego]+encoded_neighbors,dim=1) #B, N, H
        
        out_dists = []
        mode_probs = []
        for n in range(2):
            ptr_inputs = inputs.copy()
            center = inputs['agents_in'][:,-1,n,:2].clone()
            heading = inputs['agents_in'][:,-1,n,2].clone()

            ptr_inputs['agents_in'][...,:2] -=  center.view(-1,1,1,2)
            ptr_inputs['agents_in'][...,:2] = self.rotate_points(ptr_inputs['agents_in'][...,:2], torch.tensor(np.pi/2)-heading)
            ptr_inputs['ego_in'] = ptr_inputs['agents_in'][:, :, n]      


            ptr_output = self.ptr._forward(ptr_inputs)

            out_dist = ptr_output['predicted_trajectory']
            out_dist[...,:2] = self.rotate_points(out_dist[...,:2], -torch.tensor(np.pi/2)+heading)
            out_dist[...,:2] += center.view(-1,1,1,2)
            mode_prob = ptr_output['predicted_probability']



            out_dists.append(out_dist)
            mode_probs.append(mode_prob)

        out_dists = torch.stack(out_dists, dim=1)
        mode_probs = torch.stack(mode_probs, dim=1)

        # return  [c, T, B, 5], [B, c]
        output = {}
        output['level_0_probability'] = mode_probs#.unsqueeze(1) # #[B, c]
        output['level_0_trajectory'] = out_dists#.unsqueeze(1) # [c, T, B, 5] to [B, c, T, 5] to be able to parallelize code
        if len(np.argwhere(np.isnan(out_dists.detach().cpu().numpy()))) > 1:
            breakpoint()
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

        loss = self.get_loss(batch, output)

        output['predicted_trajectory'] = output[f'level_{self.N_levels}_trajectory'][:,0]
        output['predicted_probability'] = output[f'level_{self.N_levels}_probability'][:,0]

        return output, loss

    def get_loss(self, batch, prediction):
        inputs = batch['input_dict']
        ground_truth = torch.cat([inputs['obj_trajs_future_state'][...,:2],inputs['obj_trajs_future_mask'].unsqueeze(-1)],dim=-1)
        loss = self.criterion(prediction, ground_truth, self.N_levels, inputs['track_index_to_predict'])
        return loss



    def configure_optimizers(self):
        all_params = list(self.parameters())
        ptr_params = list(self.ptr.parameters())
        main_params = list(set(all_params) - set(ptr_params))
        
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
