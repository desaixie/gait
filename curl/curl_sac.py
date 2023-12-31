import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import curl.utils as utils
import drqv2.utils as drqutils
from curl.encoder import make_encoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
        diversity, exc_hidden_size, num_excluding_sequences, order_invariant, distance_obs,
        smoothness, position_only_smoothness, smoothness_window, rand_diversity_radius
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.diversity = diversity
        self.exc_hidden_size = exc_hidden_size
        self.order_invariant = order_invariant
        self.smoothness = smoothness
        self.position_only_smoothness = position_only_smoothness
        self.smoothness_window = smoothness_window
        
        in_dim = self.encoder.feature_dim + action_shape + 1  # 1: t
        if diversity:
            in_dim += exc_hidden_size
        if smoothness:
            if smoothness_window > 0:
                in_dim += exc_hidden_size
            else:
                in_dim += action_shape

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape)
        )
        
        if diversity:
            d_pose_shape = action_shape
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
            step_size_dim = 3 if self.position_only_smoothness else action_shape
            self.actions = nn.Sequential(nn.Linear(self.smoothness_window * step_size_dim, exc_hidden_size),
                                         nn.ReLU(inplace=True))

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, pos, t, compute_pi=True, compute_log_pi=True, detach_encoder=False,
            excluding_seq=None, avg_step_size=None
    ):
        z_I = self.encoder(obs, detach=detach_encoder)
        
        if self.diversity:
            if self.order_invariant:
                excluding_seq = excluding_seq.flatten(end_dim=1)
            else:
                excluding_seq = excluding_seq.flatten(start_dim=1)
        # z_I = self.trunk(obs)

        z_list = [z_I, pos, t]
        if self.diversity:
            z_N = self.seq(excluding_seq)
            if self.order_invariant:
                batch_size = obs.size(0)
                z_N = z_N.unflatten(0, (batch_size, -1)).sum(dim=1)
            z_list.append(z_N)
        if self.smoothness:
            if self.smoothness_window > 0:
                # here avg_step_size is actually step_sizes (batch, window, shape), shape is 3 or 5
                step_sizes = avg_step_size.flatten(start_dim=1)  # (batch, window, shape) to (batch, window*shape)
                z_S = self.actions(step_sizes)
                z_list.append(z_S)
            else:
                z_list.append(avg_step_size)
        z = torch.cat(z_list, dim=-1)

        # mu = self.policy(z)

        mu, log_std = self.trunk(z).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)  # noise from gaussian with mean 0 std 1
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        # assert obs.size(0) == action.size(0)

        # obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(z)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters,
        diversity, exc_hidden_size, num_excluding_sequences, order_invariant, distance_obs,
        smoothness, position_only_smoothness, smoothness_window, position_orientation_separate, rand_diversity_radius
    ):
        super().__init__()
        self.diversity = diversity
        self.exc_hidden_size = exc_hidden_size
        self.order_invariant = order_invariant
        self.smoothness = smoothness
        self.position_only_smoothness = position_only_smoothness
        self.smoothness_window = smoothness_window
        self.position_orientation_separate = position_orientation_separate


        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )
        
        in_dim = self.encoder.feature_dim + 1 + action_shape * 2  # 1: t
        if diversity:
            in_dim += exc_hidden_size
        if smoothness:
            if smoothness_window > 0:
                in_dim += exc_hidden_size
            else:
                in_dim += action_shape + 1  # 1 for step_size_diff

        self.Q1 = QFunction(in_dim, hidden_dim)
        self.Q2 = QFunction(in_dim, hidden_dim)

        if diversity:
            d_pose_shape = action_shape
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
            step_size_dim = 3 if self.position_only_smoothness else action_shape
            self.actions = nn.Sequential(nn.Linear(self.smoothness_window * (step_size_dim + 1), exc_hidden_size),
                                         nn.ReLU(inplace=True))
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, pose, t, action, detach_encoder=False, excluding_seq=None, avg_step_size=None):
        if self.diversity:
            if self.order_invariant:
                excluding_seq = excluding_seq.flatten(end_dim=1)
            else:
                excluding_seq = excluding_seq.flatten(start_dim=1)

        # detach_encoder allows to stop gradient propogation to encoder
        z_I = self.encoder(obs, detach=detach_encoder)
        z_list = [z_I, pose, t, action]
        if self.diversity:
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

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class CurlSacAgent(nn.Module):
    """CURL representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        diversity, exc_hidden_size, num_excluding_sequences, order_invariant, distance_obs,
        smoothness, position_only_smoothness, smoothness_window, position_orientation_separate, rand_diversity_radius,
        torch_random_crop, no_aug,
        image_size=84,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        curl_latent_dim=128,
        init_steps=1000,
    ):
        super().__init__()
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = image_size
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.pose_shape = action_shape
        self.init_steps = init_steps
        self.diversity = diversity
        self.smoothness = smoothness
        self.torch_random_crop = torch_random_crop
        if self.torch_random_crop:
            # self.batch_random_crop = utils.RandomCrop(self.image_size)
            self.batch_random_crop = utils.FastRandomCrop()
        self.no_aug = no_aug
        

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, diversity, exc_hidden_size, num_excluding_sequences, order_invariant,
            distance_obs, smoothness, position_only_smoothness, smoothness_window,
            rand_diversity_radius
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, diversity, exc_hidden_size, num_excluding_sequences, order_invariant,
            distance_obs, smoothness, position_only_smoothness, smoothness_window,
            position_orientation_separate, rand_diversity_radius
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, diversity, exc_hidden_size, num_excluding_sequences, order_invariant,
            distance_obs, smoothness, position_only_smoothness, smoothness_window,
            position_orientation_separate, rand_diversity_radius
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.no_aug:
            pass
        else:
            if self.encoder_type == 'pixel':
                # create CURL encoder (the 128 batch size is probably unnecessary)
                self.CURL = CURL(obs_shape, encoder_feature_dim,
                            self.curl_latent_dim, self.critic,self.critic_target, output_type='continuous').to(self.device)

                # optimizer for critic encoder for reconstruction loss
                self.encoder_optimizer = torch.optim.Adam(
                    self.critic.encoder.parameters(), lr=encoder_lr
                )

                self.cpc_optimizer = torch.optim.Adam(
                    self.CURL.parameters(), lr=encoder_lr
                )
            self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()
        
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.no_aug:
            pass
        else:
            if self.encoder_type == 'pixel':
                self.CURL.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        """ return mean only"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        """ return action sampled from SAC policy"""
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def act(self, obs, pose, t, excluding_seq, avg_step_size, step, eval_mode, history=None):
        """ select action"""
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        pose = torch.as_tensor(pose, device=self.device).unsqueeze(0)
        t = torch.as_tensor(t, device=self.device).reshape((1, 1))  # from 0 dims to 2 dims
        if self.diversity:
            excluding_seq = torch.as_tensor(excluding_seq, device=self.device).unsqueeze(0)
        if self.smoothness:
            avg_step_size = torch.as_tensor(avg_step_size, device=self.device).unsqueeze(0)
    
        mu, pi, _, _ = self.actor(obs, pose, t, compute_log_pi=False, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        if eval_mode:
            action = mu  # mean
        else:
            action = pi  # sampled from SAC policy
            if step < self.init_steps and step != -1:
                return np.random.rand(self.pose_shape).astype(np.float32) * 2. - 1.
        return action.cpu().numpy()[0]

    def update_critic(self, obs, pose, t, action, reward, discount, next_obs, next_pose, next_t, step,
                      excluding_seq=None, next_excluding_seq=None, avg_step_size=None, next_avg_step_size=None, ):
    # def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        metrics = dict()
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs, next_pose, next_t, excluding_seq=next_excluding_seq, avg_step_size=next_avg_step_size)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_pose, next_t, policy_action, excluding_seq=next_excluding_seq, avg_step_size=next_avg_step_size)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, pose, t, action, detach_encoder=self.detach_encoder, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        # if step % self.log_interval == 0:
        #     L.log('train_critic/loss', critic_loss, step)
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = current_Q1.mean().item()
        metrics['critic_q2'] = current_Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # Optimize the critic (including encoder, if args.detach_encoder is False)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return metrics

        # self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, pose, t, step, excluding_seq=None, avg_step_size=None):
    # def update_actor_and_alpha(self, obs, L, step):
        metrics = dict()
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, pose, t, detach_encoder=True, excluding_seq=excluding_seq, avg_step_size=avg_step_size)
        actor_Q1, actor_Q2 = self.critic(obs, pose, t, pi, detach_encoder=True, excluding_seq=excluding_seq, avg_step_size=avg_step_size)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # if step % self.log_interval == 0:
        #     L.log('train_actor/loss', actor_loss, step)
        #     L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        # if step % self.log_interval == 0:
        #     L.log('train_actor/entropy', entropy.mean(), step)

        metrics['actor_loss'] = actor_loss.item()
        # metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = entropy.mean().item()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        # TODO log this?
        # if step % self.log_interval == 0:
        #     L.log('train_alpha/loss', alpha_loss, step)
        #     L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return metrics

    def update_cpc(self, obs_anchor, obs_positive, cpc_kwargs, step):
        metrics = dict()
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_positive, ema=True)
        
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        
        # TODO log this
        # if step % self.log_interval == 0:
        #     L.log('train/curl_loss', loss, step)
        return metrics


    def update(self, replay_iter, step):
        metrics = dict()
    
        # if step % self.update_every_steps != 0:
        #     return metrics
    
        batch = next(replay_iter)  # batch is a tuple of cpu tensors

        # CURL uses random_cropped (requires np array) image for actor, critic, and encoder updates
        if self.no_aug:
            obs, next_obs, obs_positive = batch[0].to(self.device).float(), batch[6].to(self.device).float(), batch[-1].to(self.device).float()
        else:
            if self.torch_random_crop:
                obs, next_obs, obs_positive = batch[0].to(self.device).float(), batch[6].to(self.device).float(), batch[-1].to(self.device).float()
                obs = self.batch_random_crop(obs)
                next_obs = self.batch_random_crop(next_obs)
                obs_positive = self.batch_random_crop(obs_positive)
            else:
                obs, next_obs = batch[0].numpy(), batch[6].numpy()
                obs_positive = obs.copy()
                obs = utils.random_crop(obs, self.image_size)
                next_obs = utils.random_crop(next_obs, self.image_size)
                obs_positive = utils.random_crop(obs_positive, self.image_size)
                obs = torch.as_tensor(obs).to(self.device)
                next_obs = torch.as_tensor(next_obs).to(self.device)
                obs_positive = torch.as_tensor(obs_positive).to(self.device)

        tensor_batches = drqutils.to_torch(batch, self.device)
        _n = 0
        _, pose, t, action, reward, discount, _, next_pose, next_t = tensor_batches[_n:9]
        _n += 9
        history_len, next_history_len, history_others, history_obs, next_history_others, next_history_obs, \
            excluding_seq, next_excluding_seq, avg_step_size, next_avg_step_size = [None] * 10
        if self.diversity:
            excluding_seq, next_excluding_seq = tensor_batches[_n:_n + 2]
            _n += 2
        if self.smoothness:
            avg_step_size, next_avg_step_size = tensor_batches[_n:_n + 2]
            _n += 2
        # obs_positive = tensor_batches[-1]
        # if self.encoder_type == 'pixel':
        #     obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        # else:
        #     obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        # if step % self.log_interval == 0:
        #     L.log('train/batch_reward', reward.mean(), step)
        
        metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, pose, t, action, reward, discount, next_obs, next_pose, next_t, step,
                           excluding_seq=excluding_seq, next_excluding_seq=next_excluding_seq,
                           avg_step_size=avg_step_size, next_avg_step_size=next_avg_step_size))

        if step % self.actor_update_freq == 0:
            metrics.update(self.update_actor_and_alpha(obs, pose, t, step, excluding_seq=excluding_seq, avg_step_size=avg_step_size))

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )
        
        if self.no_aug:
            pass
        else:
            if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
                obs_anchor = obs
                metrics.update(self.update_cpc(obs_anchor, obs_positive, None, step))
        return metrics

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
 