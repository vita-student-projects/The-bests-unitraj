ego, neighbors, opps_masks, env_masks = self.process_observations(ego_in, agents_in)
agents = torch.cat((ego.unsqueeze(2), neighbors), dim=2)  # [B, Tobs, N, k_attr]

encoded_ego = self.ego_encoder(ego) # [B, d_k]
encoded_neighbors = [self.agent_encoder(neighbors[:, :, i]) for i in range(neighbors.size(2))] # Na*[B, d_k]
encoded_agents = torch.stack([encoded_ego] + encoded_neighbors, dim=1) # [B, N, d_k]

agents_mask = opps_masks[:,-1]

encoded_roads = self.road_pts_lin(roads[:, :, :, :self.map_attr]) # [B, S, P, d_k]

masks = []
encodings = []
N = agents.size(2)

lanes, lanes_mask = self.segment_map(roads, encoded_roads) # [B, S*P/10, d_k], [B, S*P/10]
fusion_input = torch.cat([encoded_agents, lanes], dim=1) # [B, N+S*P/10, d_k]
mask = torch.cat([agents_mask, lanes_mask], dim=1) # [B, N+S*P/10]
# encoding = self.fusion_encoder(fusion_input, src_key_padding_mask=mask)

masks.append(mask)
# encodings.append(encoding)





class AgentEncoder(nn.Module):
    def __init__(self, k_attr=2, d_k=128):
        super(AgentEncoder, self).__init__()
        self.k_attr = k_attr
        self.d_k = d_k

        self.motion = nn.LSTM(self.k_attr, self.d_k, 2, batch_first=True)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :self.k_attr])
        output = traj[:, -1]

        return output
    
class LaneEncoder(nn.Module):
    def __init__(self, map_attr=2, d_k=128):
        super(LaneEncoder, self).__init__()
        self.map_attr = map_attr
        self.d_k = d_k

        # encdoer layer
        self.self_line = nn.Linear(self.map_attr, self.d_k)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :self.map_attr])
    
        # process
        output = self.position_encode(self.pointnet(self_line))

        return output