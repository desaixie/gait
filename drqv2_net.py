# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

from sklearn.linear_model import LogisticRegression as logistic
from torch.nn.utils.rnn import PackedSequence

from drqv2 import utils as drqutils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad
    
    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)  # (4,4,4,4)
        x = F.pad(x, padding, 'replicate')  # replicate padding on each side by 4
        eps = 1.0 / (h + 2 * self.pad)  # 1 / len_padded_side
        arange = torch.linspace(-1.0 + eps,  # remove one pixel on each end
                                1.0 - eps,
                                h + 2 * self.pad,  # len_padded_side
                                device=x.device,
                                dtype=x.dtype)[:h]  # first h out of h + 2*pad
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        
        grid = base_grid + shift
        return F.grid_sample(x,  # grid_sample uses bilinear interpolation by default
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        """ Assumes 84*84 input. """
        super().__init__()
        
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.repr_dim_tuple = (32, 35, 35)
        
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.apply(drqutils.weight_init)
    
    def forward(self, obs):
        # normalization is done in Encoder.forward(). Normalized from [0,255] to [-0.5, 0.5]
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)  # flatten
        return h


class Context(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """
    
    def __init__(self,
                 hidden_sizes=[50],  # TODO find out appropriate hidden size and sequqnce length
                 output_dim=None,
                 input_dim=None,
                 only_concat_context=0,
                 hidden_activation=F.relu,
                 history_length=1,
                 action_dim=None,
                 obsr_dim=None,
                 device='cpu'
                 ):
        
        super(Context, self).__init__()
        self.only_concat_context = only_concat_context
        self.hid_act = hidden_activation
        self.fcs = []  # list of linear layer
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.output_dim_final = output_dim  # count the fact that there is a skip connection
        self.output_dim_last_layer = output_dim // 2
        self.hist_length = history_length
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim
        
        #### build LSTM or multi-layers FF
        if only_concat_context == 3:
            # use LSTM or GRU
            self.recurrent = nn.GRU(self.input_dim,
                                    self.hidden_sizes[0],
                                    bidirectional=False,
                                    batch_first=True,
                                    num_layers=1)
    
    def init_recurrent(self, bsize=None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0], dtype=torch.float).to(self.device)
    
    def forward(self, history):
        '''
            history: PackedSequence of size (30*256, 57) i.e. (max_steps*batch_size, feature_dim+pos_dim*2+reward_dim)
            GRU memorizes useful information from all previous observation, action, and reward
            GRU is a (not shared) layer in actor and critic and is updated when actor/critic is updated
        '''
        
        if self.only_concat_context == 3:
            # init lstm/gru
            if isinstance(history, PackedSequence):
                batch_size = history.batch_sizes[0]
            else:
                batch_size = history.size()[0]
            hidden = self.init_recurrent(bsize=batch_size)
            
            # lstm/gru
            # history: (N, L=history_length, in). hidden: (N, n_layers=1, out=hidden_dim)
            _, hidden = self.recurrent(history, hidden)  # hidden is (1, B, hidden_size)
            out = hidden.squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)
            
            return out
        
        else:
            raise NotImplementedError
        
        return None


class Actor(nn.Module):
    def __init__(self, repr_dim, pos_shape, action_shape, feature_dim, hidden_dim,
                 use_context, context_hidden_dim, context_history_length, device,
                 batch_size, use_position, diversity, exc_hidden_size, no_hidden,
                 num_excluding_sequences, order_invariant, distance_obs,
                 smoothness, position_only_smoothness, smoothness_window,
                 rand_diversity_radius):
        super().__init__()
        self.batch_size = batch_size
        self.repr_dim = repr_dim  # flattened size of obs after conv layers
        self.feature_dim = feature_dim  # 50
        self.use_position = use_position
        self.use_context = use_context
        self.diversity = diversity
        self.exc_hidden_size = exc_hidden_size
        self.no_hidden = no_hidden
        self.order_invariant = order_invariant
        self.smoothness = smoothness
        self.position_only_smoothness = position_only_smoothness
        self.smoothness_window = smoothness_window

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())  # from repr_dim to feature_dim
        
        in_dim = feature_dim + 1  # 1: t
        if use_position:
            in_dim += pos_shape[0]
        if use_context:
            in_dim += context_hidden_dim[0]
        if diversity:
            if no_hidden:
                d_pose_shape = pos_shape[0]
                if distance_obs:
                    d_pose_shape += 1
                if rand_diversity_radius:
                    d_pose_shape += 1
                in_dim += num_excluding_sequences * d_pose_shape
            else:
                in_dim += exc_hidden_size
        if smoothness:
            if smoothness_window > 0:
                if no_hidden:
                    step_size_dim = 3 if self.position_only_smoothness else pos_shape[0]
                    in_dim += self.smoothness_window * step_size_dim
                else:
                    in_dim += exc_hidden_size
            else:
                in_dim += pos_shape[0]
                
        self.policy = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        
        if use_context:
            input_dim = feature_dim + action_shape[0] * 2 + 1 if self.use_position else feature_dim + action_shape[0] + 1
            self.context = Context(hidden_sizes=context_hidden_dim,
                                   input_dim=input_dim,
                                   output_dim=context_hidden_dim[0],
                                   history_length=context_history_length,
                                   only_concat_context=3,
                                   action_dim=action_shape[0],
                                   obsr_dim=feature_dim,
                                   device=device
                                   )
        if diversity:
            d_pose_shape = pos_shape[0]
            if distance_obs:
                d_pose_shape += 1
            if rand_diversity_radius:
                d_pose_shape += 1
            if order_invariant:
                self.seq = nn.Sequential(nn.Linear(d_pose_shape, exc_hidden_size),
                                         nn.ReLU(inplace=True))
            else:
                self.seq = nn.Sequential(nn.Linear(num_excluding_sequences * d_pose_shape, exc_hidden_size),
                                         nn.ReLU(inplace=True))
                
        if smoothness_window > 0:
            step_size_dim = 3 if self.position_only_smoothness else pos_shape[0]
            self.actions = nn.Sequential(nn.Linear(self.smoothness_window * step_size_dim, exc_hidden_size),
                                     nn.ReLU(inplace=True))
            
        self.apply(drqutils.weight_init)

    def encode_hisotry_obs(self, history, history_len=None):
        history_others, history_obs = history
        batch_size = history_others.size()[0]
        with torch.no_grad():
            if history_len is not None:  # history is a batch
                history_obs = self.trunk(history_obs.view(-1, self.repr_dim)).view(batch_size, -1, self.feature_dim)
                history = torch.nn.utils.rnn.pack_padded_sequence(torch.cat([history_others.data,
                                                                             history_obs], dim=-1), lengths=history_len.cpu(), batch_first=True, enforce_sorted=False)
            else:  # history is a single tensor
                history_obs = self.trunk(history_obs)
                history = torch.cat([history_others, history_obs], dim=-1)
                history = history.unsqueeze(0)  # insert batch dimension
        return history
    
    def forward(self, obs, pose, t, std, history_len=None, history=None, excluding_seq=None, avg_step_size=None):
        # excluding_seq: (batch_size, num_excluding_seqs, pose_shape)
        if self.diversity:
            if self.order_invariant:
                excluding_seq = excluding_seq.flatten(end_dim=1)
            else:
                excluding_seq = excluding_seq.flatten(start_dim=1)
        z_I = self.trunk(obs)
        
        z_list = [z_I, pose, t]
        if self.use_context:
            history = self.encode_hisotry_obs(history, history_len=history_len)
            z_C = self.context(history)
            z_list.append(z_C)
        if self.diversity:
            if self.no_hidden:
                z_list.append(excluding_seq)
            else:
                z_N = self.seq(excluding_seq)
                if self.order_invariant:
                    batch_size = obs.size(0)
                    z_N = z_N.unflatten(0, (batch_size, -1)).sum(dim=1)
                z_list.append(z_N)
        if self.smoothness:
            if self.smoothness_window > 0:
                # here avg_step_size is actually step_sizes (batch, window, shape), shape is 3 or 5
                step_sizes = avg_step_size.flatten(start_dim=1)  # (batch, window, shape) to (batch, window*shape)
                if self.no_hidden:
                    z_list.append(step_sizes)
                else:
                    z_S = self.actions(step_sizes)
                    z_list.append(z_S)
            else:
                z_list.append(avg_step_size)
        z = torch.cat(z_list, dim=-1)
        
        mu = self.policy(z)
        mu = torch.tanh(mu)  # network outputs mean
        std = torch.ones_like(mu) * std  # std provided
        
        dist = drqutils.TruncatedNormal(mu, std)
        return dist
    
    def get_conext_feats(self, history, history_len):
        history = self.encode_hisotry_obs(history, history_len=history_len)
        z = self.context(history)
        return z


class Critic(nn.Module):
    def __init__(self, repr_dim, pos_shape, action_shape, feature_dim, hidden_dim,
                 use_context, context_hidden_dim, context_history_length, device,
                 batch_size, use_position, diversity, exc_hidden_size, no_hidden,
                 num_excluding_sequences, order_invariant, distance_obs,
                 smoothness, position_only_smoothness, smoothness_window,
                 position_orientation_separate, rand_diversity_radius):
        super().__init__()
        self.batch_size = batch_size
        self.repr_dim = repr_dim
        self.feature_dim = feature_dim
        self.use_position = use_position
        self.use_context = use_context
        self.diversity = diversity
        self.exc_hidden_size = exc_hidden_size
        self.no_hidden = no_hidden
        self.order_invariant = order_invariant
        self.smoothness = smoothness
        self.position_only_smoothness = position_only_smoothness
        self.smoothness_window = smoothness_window
        self.position_orientation_separate = position_orientation_separate

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        in_dim = feature_dim + 1 + action_shape[0]  # 1: t
        if use_position:
            in_dim += pos_shape[0]
        if use_context:
            in_dim += context_hidden_dim[0]
        if diversity:
            if no_hidden:
                d_pose_shape = pos_shape[0]
                if distance_obs:
                    d_pose_shape += 1
                if rand_diversity_radius:
                    d_pose_shape += 1
                in_dim += num_excluding_sequences * d_pose_shape
            else:
                in_dim += exc_hidden_size
        if smoothness:
            if smoothness_window > 0:
                if no_hidden:
                    step_size_dim = 3 if self.position_only_smoothness else pos_shape[0]
                    in_dim += self.smoothness_window * (step_size_dim+1)
                else:
                    in_dim += exc_hidden_size
            else:
                in_dim += pos_shape[0] + 1  # 1 for step_size_diff
        # if use_context:
        #     if use_position:
        #         in_dim = feature_dim + pos_shape[0] + action_shape[0] + context_hidden_dim[0]
        #     else:
        #         in_dim = feature_dim + action_shape[0] + context_hidden_dim[0]
        # else:
        #     if use_position:
        #         in_dim = feature_dim + pos_shape[0] + action_shape[0]
        #     else:
        #         in_dim = feature_dim + action_shape[0]
        #     # feature_dim + pos_shape[0] + action_shape[0]
        self.Q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        
        self.Q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        
        if use_context:
            input_dim = feature_dim + action_shape[0] * 2 + 1 if self.use_position else feature_dim + action_shape[0] + 1
            self.context = Context(hidden_sizes=context_hidden_dim,
                                   input_dim=input_dim,
                                   output_dim=context_hidden_dim[0],
                                   history_length=context_history_length,
                                   only_concat_context=3,
                                   action_dim=action_shape[0],
                                   obsr_dim=feature_dim,
                                   device=device
                                   )
            
        if diversity:
            d_pose_shape = pos_shape[0]
            if distance_obs:
                d_pose_shape += 1
            if rand_diversity_radius:
                d_pose_shape += 1
            if order_invariant:
                self.seq = nn.Sequential(nn.Linear(d_pose_shape, exc_hidden_size),
                                         nn.ReLU(inplace=True))
            else:
                self.seq = nn.Sequential(nn.Linear(num_excluding_sequences * d_pose_shape, exc_hidden_size),
                                         nn.ReLU(inplace=True))
                
        if smoothness_window > 0:
            step_size_dim = 3 if self.position_only_smoothness else pos_shape[0]
            self.actions = nn.Sequential(nn.Linear(self.smoothness_window * (step_size_dim+1), exc_hidden_size),
                                         nn.ReLU(inplace=True))

        self.apply(drqutils.weight_init)

    def encode_hisotry_obs(self, history, history_len=None):
        history_others, history_obs = history
        batch_size = history_others.size()[0]
        with torch.no_grad():
            if history_len is not None:  # history is a batch
                history_obs = self.trunk(history_obs.view(-1, self.repr_dim)).view(batch_size, -1, self.feature_dim)
                history = torch.nn.utils.rnn.pack_padded_sequence(torch.cat([history_others.data,
                                                                             history_obs], dim=-1), lengths=history_len.cpu(), batch_first=True, enforce_sorted=False)
            else:  # history is a single tensor
                history_obs = self.trunk(history_obs)
                history = torch.cat([history_others, history_obs], dim=-1)
                history = history.unsqueeze(0)  # insert batch dimension
        return history
    
    def forward(self, obs, pose, t, action, history_len=None, history=None, excluding_seq=None, avg_step_size=None):
        # excluding_seq: (batch_size, num_excluding_seqs, pose_shape)
        if self.diversity:
            if self.order_invariant:
                excluding_seq = excluding_seq.flatten(end_dim=1)
            else:
                excluding_seq = excluding_seq.flatten(start_dim=1)
            
        z_I = self.trunk(obs)
        z_list = [z_I, pose, t, action]
        if self.use_context:
            history = self.encode_hisotry_obs(history, history_len=history_len)
            z_C = self.context(history)
            z_list.append(z_C)
        if self.diversity:
            if self.no_hidden:
                z_list.append(excluding_seq)
            else:
                z_N = self.seq(excluding_seq)
                if self.order_invariant:
                    batch_size = obs.size(0)
                    z_N = z_N.unflatten(0, (batch_size, -1)).sum(dim=1)
                z_list.append(z_N)
        if self.smoothness:
            if self.smoothness_window > 0:
                # here avg_step_size is actually step_sizes (batch, window, shape), shape is 3 or 5
                step_sizes = avg_step_size
                # critic considers step_size_diff (diff btw action and step_sizes(past actions))
                if self.position_only_smoothness:
                    action_position = action[:, :3]
                    step_size_diffs = torch.linalg.norm(action_position.unsqueeze(1) - step_sizes, dim=-1, keepdim=True)  # broaccasted to (batch, window, shape), (batch, window, 1) after norm
                elif self.position_orientation_separate:  # get mean of translation action diff and rotation action diff
                    action_p_o = action[:, :3], action[:, 3:]
                    step_sizes_p_o = step_sizes[:, :, :3], step_sizes[:, :, 3:]
                    step_size_diffs = (torch.linalg.norm(action_p_o[0].unsqueeze(1) - step_sizes_p_o[0], dim=-1, keepdim=True) + torch.linalg.norm(action_p_o[1].unsqueeze(1) - step_sizes_p_o[1], dim=-1, keepdim=True)) / 2.  # broaccasted to (batch, window, shape), (batch, window, 1) after norm
                else:
                    step_size_diffs = torch.linalg.norm(action.unsqueeze(1) - step_sizes, dim=-1, keepdim=True)  # broaccasted to (batch, window, shape), (batch, window, 1) after norm
                step_sizes = torch.cat([step_sizes, step_size_diffs], dim=-1)  # (batch, window, shape+1)
                step_sizes = step_sizes.flatten(start_dim=1)  # (batch, window, shape+1) to (batch, window*(shape+1))
                if self.no_hidden:
                    z_list.append(step_sizes)
                else:
                    z_S = self.actions(step_sizes)
                    z_list.append(z_S)
            else:
                step_size_diff = torch.linalg.norm(action - avg_step_size, dim=-1, keepdim=True)
                # avg_step_size = torch.cat([avg_step_size, step_size_diff], dim=-1)
                z_list.append(avg_step_size)
                z_list.append(step_size_diff)
        z = torch.cat(z_list, dim=-1)
        
        q1 = self.Q1(z)
        q2 = self.Q2(z)
        
        return q1, q2  # double Q learning?
    
    def get_conext_feats(self, history, history_len):
        history = self.encode_hisotry_obs(history, history_len=history_len)
        z = self.context(history)
        return z


class DrQV2Agent(nn.Module):
    def __init__(self, obs_shape, pos_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 use_context, context_hidden_dim, context_history_length, nstep, batch_size,
                 num_scenes, use_position, diversity, exc_hidden_size, no_hidden, num_excluding_sequences,
                 order_invariant, distance_obs, smoothness, position_only_smoothness, smoothness_window,
                 position_orientation_separate, rand_diversity_radius, constant_noise, no_aug):
        super().__init__()
        print(f"Creating agent on device {device}")
        from datetime import datetime
        self.create_time = datetime.now()
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_context = use_context
        self.nstep = nstep
        self.obs_shape = obs_shape
        self.batch_size = batch_size
        self.context_history_length = context_history_length
        self.num_scenes = num_scenes
        self.use_position = use_position
        self.diversity = diversity
        self.smoothness = smoothness
        self.constant_noise = constant_noise
        if self.constant_noise != -1:
            self.stddev_schedule = self.constant_noise
        self.no_aug = no_aug
        
        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, pos_shape, action_shape, feature_dim, hidden_dim,  # no actor_target
                           use_context, context_hidden_dim, context_history_length, device, batch_size,
                           use_position, diversity, exc_hidden_size, no_hidden, num_excluding_sequences, order_invariant,
                           distance_obs, smoothness, position_only_smoothness, smoothness_window,
                           rand_diversity_radius).to(device)
        
        self.critic = Critic(self.encoder.repr_dim, pos_shape, action_shape, feature_dim, hidden_dim,
                             use_context, context_hidden_dim, context_history_length, device, batch_size,
                             use_position, diversity, exc_hidden_size, no_hidden, num_excluding_sequences, order_invariant,
                             distance_obs, smoothness, position_only_smoothness, smoothness_window,
                             position_orientation_separate, rand_diversity_radius).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, pos_shape, action_shape, feature_dim, hidden_dim,
                                    use_context, context_hidden_dim, context_history_length, device, batch_size,
                                    use_position, diversity, exc_hidden_size, no_hidden, num_excluding_sequences, order_invariant,
                                    distance_obs, smoothness, position_only_smoothness, smoothness_window,
                                    position_orientation_separate, rand_diversity_radius).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        
        self.train()
        self.critic_target.train()
        
    def get_weights(self):
        # TODO only send actor and encoder weights
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    def act(self, obs, pos, t, excluding_seq, avg_step_size, step, eval_mode, history=None):
        """ select action"""
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        obs = self.encoder(obs)
        if step == -1:
            stddev = 0.2
        else:
            stddev = drqutils.schedule(self.stddev_schedule, step)
        
        pos = torch.as_tensor(pos, device=self.device).unsqueeze(0)
        t = torch.as_tensor(t, device=self.device).reshape((1, 1))  # from 0 dims to 2 dims
        if self.diversity:
            excluding_seq = torch.as_tensor(excluding_seq, device=self.device).unsqueeze(0)
        if self.smoothness:
            avg_step_size = torch.as_tensor(avg_step_size, device=self.device).unsqueeze(0)

        # process history_obs through encoder
        if history is not None:
            history_others, history_obs = history
            with torch.no_grad():
                history_obs = self.encoder(torch.as_tensor(history_obs, device=self.device))
            history = [torch.as_tensor(history_others, device=self.device), history_obs]
        
        dist = self.actor(obs, pos, t, stddev, history=history, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)  # exploration is done by providing a std to actor, and sample actions from a normal distribution centered at actor's output
            if step < self.num_expl_steps and step != -1:
                action.uniform_(-1.0, 1.0)  # random action
        return action.cpu().numpy()[0]
    
    def update_critic(self, obs, pose, t, action, reward, discount, next_obs, next_pos, next_t, step, history_len=None, next_history_len=None,
                      history=None, next_history=None, excluding_seq=None, next_excluding_seq=None, avg_step_size=None, next_avg_step_size=None,
                      apply_prox=False, beta_score=None):
        metrics = dict()
        
        with torch.no_grad():
            if step == -1:
                stddev = 0.2
            else:
                stddev = drqutils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, next_pos, next_t, stddev, history_len=next_history_len, history=next_history, excluding_seq=next_excluding_seq, avg_step_size=next_avg_step_size)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_pos, next_t, next_action, history_len=next_history_len, history=next_history, excluding_seq=next_excluding_seq, avg_step_size=next_avg_step_size)
            target_V = torch.min(target_Q1, target_Q2)  # Double Q learning
            target_Q = reward + (discount * target_V)  # standard Q learning

        Q1, Q2 = self.critic(obs, pose, t, action, history_len=history_len, history=history, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)  # critic_loss is a scalar, no need to .mean()
        
        if beta_score is not None:  # adaptation
            critic_loss = (critic_loss * beta_score).mean()
            # critic_loss_out += critic_loss.item()
        
        if apply_prox:
            critic_prox = self.get_prox_penalty(self.critic, self.ckpt['critic'])  # TODO ckpt?
            critic_loss = critic_loss + self.prox_coef * critic_prox
            # critic_prox_out += critic_prox.item()
        
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        
        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        
        return metrics
    
    def update_actor(self, obs, pose, t, step, history_len=None, history=None, excluding_seq=None, avg_step_size=None, apply_prox=False, beta_score=None):
        metrics = dict()
        
        if step == -1:
            stddev = 0.2
        else:
            stddev = drqutils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, pose, t, stddev, history_len=history_len, history=history, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, pose, t, action, history_len=history_len, history=history, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        Q = torch.min(Q1, Q2)
        
        if beta_score is not None:  # adaptation
            Q *= beta_score
        
        actor_loss = -Q.mean()  # standard DDPG actor upadte + Double Q leanring
        
        if apply_prox:  # adaptation
            # calculate proximal term
            actor_prox = self.get_prox_penalty(self.actor, self.ckpt['actor'])  # TODO ckpt?
            actor_loss = actor_loss + self.prox_coef * actor_prox
            # actor_prox_out += actor_prox.item()
        
        # optimize actor. Don't update encoder with actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        
        return metrics
    
    def update(self, replay_iter, step):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)  # batch is a tuple of cpu tensors
        tensor_batches = drqutils.to_torch(batch, self.device)
        _n = 0
        obs, pose, t, action, reward, discount, next_obs, next_pose, next_t = tensor_batches[_n:9]
        _n += 9
        history_len, next_history_len, history_others, history_obs, next_history_others, next_history_obs, \
            excluding_seq, next_excluding_seq, avg_step_size, next_avg_step_size = [None] * 10
        if self.use_context:
            history_len, history_others, history_obs, next_history_len, next_history_others, next_history_obs = tensor_batches[_n:_n+6]
            _n += 6
        if self.diversity:
            excluding_seq, next_excluding_seq = tensor_batches[_n:_n + 2]
            _n += 2
        if self.smoothness:
            avg_step_size, next_avg_step_size = tensor_batches[_n:_n + 2]
            _n += 2

        # augment
        if self.no_aug:
            pass
        else:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        
        # encode
        # Encoder is shared. However it is only updated once together with critic and only a single forward pass has grad enabled
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
        
        # process history_obs through encoder
        if self.use_context:
            with torch.no_grad():
                next_history_obs = self.encoder(next_history_obs.view(-1, *self.obs_shape)).view(self.batch_size, -1, self.encoder.repr_dim)  # batch_size, max_steps, 32*35*35 flattened CNN features
                next_history = [next_history_others, next_history_obs]
                
                history_obs = self.encoder(history_obs.view(-1, *self.obs_shape)).view(self.batch_size, -1, self.encoder.repr_dim)
                history = [history_others, history_obs]
        else:
            next_history, history = None, None
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        
        # update critic
        metrics.update(self.update_critic(obs, pose, t, action, reward, discount, next_obs, next_pose, next_t, step, history_len=history_len, next_history_len=next_history_len,
                                          history=history, next_history=next_history, excluding_seq=excluding_seq, next_excluding_seq=next_excluding_seq,
                                          avg_step_size=avg_step_size, next_avg_step_size=next_avg_step_size, beta_score=None))
        
        # update actor
        # TODO consider TD3's delayed policy update
        metrics.update(self.update_actor(obs.detach(), pose, t, step, history_len=history_len, history=history, excluding_seq=excluding_seq, avg_step_size=avg_step_size,
                                         beta_score=None))
        
        # update critic target
        drqutils.soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)
        
        return metrics
    
    ############################
    # MQL adaptation, proximal term and propensity score
    ############################
    
    def adapt(self,
              metatrain_replay_iter=None,
              eval_replay_iter=None,
              snap_iter_nums=5,  # 10
              main_snap_iter_nums=15,  # 100
              main_snap_bsize_mult=1,  # TODO 5 in main args
              evaluation_buffer_size=-1):
        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
            task_id is the eval/test scene name
        '''
        self.use_ess_clipping = True
        self.use_normalized_beta = True
        self.enable_beta_obs_cxt = True
        self.beta_clip = 1.0
        self.max_iter_logistic = 2000
        self.lam_csc = 0.5
        self.r_eps = np.float32(1e-7)  # this is used to avoid inf or nan in calculations
        self.prox_coef = 0.1
        self.prox_coef_init = self.prox_coef
        # self.train_tasks_list = set(self.sceneList.copy())
        #######
        # Reset optim at the beginning of the adaptation
        #######
        # for now, using default lr 1e-3. DrQv2's lr is 1e-4
        self.actor_opt = torch.optim.Adam(self.actor.parameters())
        self.critic_opt = torch.optim.Adam(self.critic.parameters())
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters())
        
        #######
        # Adaptaion step:
        # learn a model to correct covariate shift
        #######
        out_single = None
        
        # train covariate shift correction model
        csc_model, csc_info = self.train_cs(snap_replay_iter=eval_replay_iter,
                                            train_replay_iter=metatrain_replay_iter,
                                            adaptation_step=True,
                                            metatrain_batch_size=main_snap_bsize_mult * self.batch_size,
                                            evaluation_buffer_size=evaluation_buffer_size)
        
        # train td3 for a single task, the new task
        out_single = self.adapt_update(replay_iter=eval_replay_iter,
                                       iterations=snap_iter_nums,
                                       csc_model=None,
                                       apply_prox=False,
                                       # use the whole eval buffer
                                       current_batch_size=evaluation_buffer_size)
        # self.copy_model_params()
        
        # keep a copy of model params for task task_id
        out_single['csc_info'] = csc_info
        out_single['snap_iter'] = snap_iter_nums
        
        # traing TD3 on meta-training replay buffer
        # sampling_style is based on 'replay'
        # each train task has own buffer, so sample from each of them
        out = self.adapt_update(replay_iter=metatrain_replay_iter,
                                iterations=main_snap_iter_nums,
                                csc_model=csc_model,
                                apply_prox=True,
                                current_batch_size=main_snap_bsize_mult * self.batch_size)
        
        return out, out_single
    
    def adapt_update(self,
                     replay_iter=None,
                     iterations=None,
                     csc_model=None,
                     apply_prox=False,
                     current_batch_size=None):
        metrics = dict()
        step = -1  # marks adaptation
        
        for _ in range(iterations):
            batch = next(replay_iter)  # batch is a tuple of cpu tensors
            if self.use_context:
                obs, pos, action, reward, discount, next_obs, next_pos, history_len, history_others, history_obs, next_history_len, next_history_others, next_history_obs = drqutils.to_torch(
                    batch, self.device)
            else:
                obs, pos, action, reward, discount, next_obs, next_pos = drqutils.to_torch(
                    batch, self.device)
                history_len, next_history_len, history_others, history_obs, next_history_others, next_history_obs = None, None, None, None, None, None
            
            # augment
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            
            # encode
            # Encoder is shared. However it is only updated once together with critic and only a single forward pass has grad enabled
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)
            
            # process history_obs through encoder
            if self.use_context:
                with torch.no_grad():
                    next_history_obs = self.encoder(next_history_obs.view(-1, *self.obs_shape)).view(current_batch_size, -1, self.encoder.repr_dim)  # batch_size, max_steps, 32*35*35 flattened CNN features
                    next_history = [next_history_others, next_history_obs]
                    
                    history_obs = self.encoder(history_obs.view(-1, *self.obs_shape)).view(current_batch_size, -1, self.encoder.repr_dim)
                    history = [history_others, history_obs]
            else:
                next_history, history = None, None
            
            if self.use_tb:
                metrics['batch_reward'] = reward.mean().item()
            
            if csc_model is None:
                # propensity_scores dim is batch_size
                # no csc_model, so just do business as usual
                beta_score = torch.ones((current_batch_size, 1)).to(self.device)
            
            else:
                # propensity_scores dim is batch_size
                beta_score, clipping_factor = self.get_propensity(csc_model, history, history_len, obs, pos)
                beta_score = beta_score.to(self.device)
                # list_prox_coefs.append(self.prox_coef)
            
            # update critic
            metrics.update(
                self.update_critic(obs, pos, action, reward, discount, next_obs, next_pos, step, history_len=history_len, next_history_len=next_history_len, history=history, next_history=next_history, apply_prox=apply_prox, beta_score=beta_score))
            
            # update actor
            # TODO consider TD3's delayed policy update
            metrics.update(self.update_actor(obs.detach(), pos, step, history_len=history_len, history=history, apply_prox=apply_prox, beta_score=beta_score))
            
            # update critic target
            drqutils.soft_update_params(self.critic, self.critic_target,
                                        self.critic_target_tau)
        
        return metrics
    
    def copy_model_params(self):
        '''
            Keep a copy of actor and critic for proximal update. Call this before adaptation begins
        '''
        self.ckpt = {
            'actor': deepcopy(self.actor),
            'critic': deepcopy(self.critic)
        }
    
    def get_prox_penalty(self, model_t, model_target):
        '''
            This function calculates ||theta - theta_t||
            Keeps theta (current parameters) close to theta_t (meta-training parameters), reduce variance during adaptation
        '''
        param_prox = []
        for p, q in zip(model_t.parameters(), model_target.parameters()):
            # q should ne detached
            param_prox.append((p - q.detach()).norm() ** 2)
        
        result = sum(param_prox)
        
        return result
    
    def train_cs(self, snap_replay_iter=None, train_replay_iter=None, adaptation_step=False, metatrain_batch_size=-1, evaluation_buffer_size=-1):
        '''
            This function trains covariate shift correction model, a logistic classifier (-1: meta, 1: new task)
            snap_buffer is eval_eval_task buffer
        '''
        
        ######
        # fetch all_data
        ######
        if adaptation_step == True:
            # step 1: calculate how many samples per classes we need
            # in adaption step, all train task can be used
            # task_bsize = int(len(snap_buffer) / (len(self.train_tasks_list))) + 2
            # batch_size = len(snap_buffer) + 2
            # neg_tasks_ids = self.train_tasks_list
            metatrain_batch_size = min(evaluation_buffer_size, metatrain_batch_size)
        else:
            # step 1: calculate how many samples per classes we need
            # task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list) - 1)) + 2
            # neg_tasks_ids = list(self.train_tasks_list.difference(set([task_id])))
            raise
        
        # TODO record task id in replay buffer, sample evenly a batch of size task_bsize from each task's buffer
        # collect examples from other tasks and consider them as one class
        # negative: meta-train buffer. positive: snap buffer
        if self.use_context:
            batch = next(train_replay_iter)  # batch is a tuple of cpu tensors
            neg_obs, neg_pos, _, _, _, _, _, neg_history_len, neg_history_others, neg_history_obs, _, _, _ = drqutils.to_torch(
                batch, self.device)
            neg_obs, neg_pos, neg_history_len, neg_history_others, neg_history_obs = neg_obs[:metatrain_batch_size], neg_pos[:metatrain_batch_size], neg_history_len[:metatrain_batch_size], neg_history_others[:metatrain_batch_size], neg_history_obs[:metatrain_batch_size]
            batch = next(snap_replay_iter)  # batch is a tuple of cpu tensors
            pos_obs, pos_pos, _, _, _, _, _, pos_history_len, pos_history_others, pos_history_obs, _, _, _ = drqutils.to_torch(
                batch, self.device)
        else:
            raise
        
        ######
        # extract features: context features
        ######
        with torch.no_grad():
            neg_history_obs = self.encoder(neg_history_obs.view(-1, *self.obs_shape)).view(metatrain_batch_size, -1, self.encoder.repr_dim)
            neg_history = [neg_history_others, neg_history_obs]
            neg_obs = self.actor.trunk(self.encoder(neg_obs))  # (batch_size, self.actor.feature_dim (50))
            
            pos_history_obs = self.encoder(pos_history_obs.view(-1, *self.obs_shape)).view(evaluation_buffer_size, -1, self.encoder.repr_dim)
            pos_history = [pos_history_others, pos_history_obs]
            pos_obs = self.actor.trunk(self.encoder(pos_obs))
            
            # batch_size X context_hidden
            # self.actor.get_conext_feats outputs, [batch_size , context_size]
            # torch.cat ([batch_size , obs_dim], [batch_size , context_size]) ==> [batch_size, obs_dim + context_size ]
            if self.enable_beta_obs_cxt == True:
                snap_ctxt = torch.cat([pos_obs, pos_pos,
                                       self.actor.get_conext_feats(pos_history, pos_history_len)], dim=-1).cpu().data.numpy()
                neg_ctxt = torch.cat([neg_obs, neg_pos,
                                      self.actor.get_conext_feats(neg_history, neg_history_len)], dim=-1).cpu().data.numpy()
            
            else:
                # snap_ctxt = self.actor.get_conext_feats(pos_act_rew_obs).cpu().data.numpy()
                # neg_ctxt = self.actor.get_conext_feats(neg_act_rew_obs).cpu().data.numpy()
                raise
        
        ######
        # Train logistic classifiers
        ######
        x = np.concatenate((snap_ctxt, neg_ctxt))  # [b1 + b2] X D
        y = np.concatenate((-np.ones(snap_ctxt.shape[0]), np.ones(neg_ctxt.shape[0])))
        
        # model params : [1 , D] wehere D is context_hidden
        model = logistic(solver='lbfgs', max_iter=self.max_iter_logistic, C=self.lam_csc).fit(x, y)
        
        info = (snap_ctxt.shape[0], neg_ctxt.shape[0], model.score(x, y))
        return model, info
    
    def update_prox_w_ess_factor(self, cs_model, x, beta=None):
        '''
            This function calculates effective sample size (ESS):
            ESS = ||w||^2_1 / ||w||^2_2  , w = pi / beta
            ESS = ESS / n where n is number of samples to normalize
            x: is (n, D)
        '''
        n = x.shape[0]
        if beta is not None:
            # beta results should be same as using cs_model.predict_proba(x)[:,0] if no clipping
            w = ((torch.sum(beta) ** 2) / (torch.sum(beta ** 2) + self.r_eps)) / n
            ess_factor = np.float32(w.numpy())
        
        else:
            # step 1: get prob class 1
            p0 = cs_model.predict_proba(x)[:, 0]
            w = p0 / (1 - p0 + self.r_eps)
            w = (np.sum(w) ** 2) / (np.sum(w ** 2) + self.r_eps)
            ess_factor = np.float32(w) / n
        
        # since we assume task_i is class -1, and replay buffer is 1, then
        ess_prox_factor = 1.0 - ess_factor
        
        if np.isnan(ess_prox_factor) or np.isinf(ess_prox_factor) or ess_prox_factor <= self.r_eps:  # make sure that it is valid
            self.prox_coef = self.prox_coef_init
        
        else:
            self.prox_coef = ess_prox_factor
    
    def get_propensity(self, cs_model, history, history_len, obs, pos):
        '''
            This function returns propensity for current sample of data
            simply: exp(f(x))
            
            Beta (propensity score, or importance ratio) represents how much a meta-training sample
            looks like a new-task sample. It is used to weight (give importance to) meta-training
            samples during adaptation
            assumes history and obs comes from adapt update. history is [history_others, history_obs],
            with history_obs passed through encoder; obs also passed through encoder
        '''
        
        ######
        # extract features: context features
        ######
        with torch.no_grad():
            
            # batch_size X context_hidden
            if self.enable_beta_obs_cxt == True:
                obs = self.actor.trunk(obs)
                ctxt = torch.cat([obs, pos,
                                  self.actor.get_conext_feats(history, history_len)], dim=-1).cpu().data.numpy()
            
            else:
                # ctxt = self.actor.get_conext_feats(curr_pre_act_rew).cpu().data.numpy()
                raise
        
        # step 0: get f(x)
        f_prop = np.dot(ctxt, cs_model.coef_.T) + cs_model.intercept_
        
        # step 1: convert to torch
        f_prop = torch.from_numpy(f_prop).float()
        
        # To make it more stable, clip it
        f_prop = f_prop.clamp(min=-self.beta_clip)
        
        # step 2: exp(-f(X)), f_score: N * 1
        f_score = torch.exp(-f_prop)
        f_score[f_score < 0.1] = 0  # for numerical stability
        
        if self.use_normalized_beta == True:
            # get logistic regression prediction of class [-1] for current task
            lr_prob = cs_model.predict_proba(ctxt)[:, 0]
            # normalize using logistic_probs
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))
            f_score = (d_pmax_pmin * (f_score - torch.min(f_score))) / (
                        torch.max(f_score) - torch.min(f_score) + self.r_eps) + np.float32(np.min(lr_prob))
        
        # update prox coeff with ess.
        if self.use_ess_clipping == True:
            self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)
        
        return f_score, None
