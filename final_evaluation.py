# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import traceback

import warnings

import trajectory_visualize
from trajectory_visualize import plot3d_and_save_vid
import matplotlib
gui_env = ['QtAgg', 'TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        # traceback.print_exc()
        continue
print("final_evalutaion plt backend: ",matplotlib.get_backend())

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from torch import multiprocessing
from pathlib import Path
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


import hydra
import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True, sign=' ', floatmode='fixed')
from dm_env import specs  # https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py

# local import
import drqv2.utils as drqutils
from drqv2.logger import Logger
import saver_utils
from habitat_test import AestheticTourDMCWrapper, HabitatSimGymWrapper, SpaceMapping


def make_agent(obs_spec, pos_spec, action_spec, cfg):
    """ Created from config.yaml agent block"""
    cfg.obs_shape = obs_spec.shape
    cfg.pos_shape = pos_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        # don't access cfg, use the loaded self.cfg (accurate)
        self.scenes = ["room_0_, apartment_2_livingroom, office_3_, room_1_, apartment_0_bedroomUpperFloorMirror, office_0_"]
        self.scene_id = 0  # 0, 1, 2, 3, 4, or 5
        self.algo_id = 1  # 1 or 2 (DrQv2 or CURL)
        self.special = False
        self.fig4_mode = False  # robust
        self.fig6_mode = False  # diversity
        self.fig7_mode = False  # smoothness
        self.fig7_ablation_mode = False
        self.teaser_mode = True  # teaser
        self.user_study_mode = False  # 9 sequences per scene
        self.work_dir = Path.cwd()
        # self.snapshot_path = Path(cfg.snapshot_dir)
        self.snapshot_path = Path("../92 no hidden room0")
        if self.fig4_mode:
            if self.algo_id == 1:
                if self.scene_id == 0:
                    self.snapshot_path = Path("../70 no diversity room_0")
                elif self.scene_id == 1:
                    self.snapshot_path = Path("../74 no diversity apartment2")
                elif self.scene_id == 2:
                    self.snapshot_path = Path("../75 no diversity office3")
                else:
                    raise
            elif self.algo_id == 2:
                if self.scene_id == 0:
                    # self.snapshot_path = Path("../70 no diversity room_0")
                    pass
                elif self.scene_id == 1:
                    self.snapshot_path = Path("../10 no diversity apartment2")
                elif self.scene_id == 2:
                    self.snapshot_path = Path("../11 no diversity office3")
                else:
                    raise
        if self.fig6_mode:
            if self.algo_id == 1:
                if self.scene_id == 0:
                    # self.snapshot_path = Path("../70 no diversity room_0")
                    pass
                elif self.scene_id == 1:
                    self.snapshot_path = Path("../67 apartment2living")
                elif self.scene_id == 2:
                    self.snapshot_path = Path("../69 office_3_")
                elif self.scene_id == 3:
                    self.snapshot_path = Path("../76 default room_1")
                elif self.scene_id == 4:
                    self.snapshot_path = Path("../78 default apartment_0_bedroomMirror")
                elif self.scene_id == 5:
                    self.snapshot_path = Path("../77 default office_0")
                else:
                    raise
            elif self.algo_id == 2:
                if self.scene_id == 0:
                    self.snapshot_path = Path("../4 room_0_")
                elif self.scene_id == 1:
                    self.snapshot_path = Path("../5 apartment_2_livingroom")
                elif self.scene_id == 2:
                    self.snapshot_path = Path("../6 office_3_")
                elif self.scene_id == 3:
                    self.snapshot_path = Path("../16 default room1")
                elif self.scene_id == 4:
                    self.snapshot_path = Path("../18 default apartment0")
                elif self.scene_id == 5:
                    self.snapshot_path = Path("../17 default office0")
                else:
                    raise
        self.snapshot_name = 'best.pt'
        assert (self.snapshot_path / self.snapshot_name).exists(), f"The specified snapshot is not found: {self.snapshot_path / self.snapshot_name}"
        print(f'Resuming. Loaded snapshot at : {self.snapshot_path / self.snapshot_name}')
        print(f'workspace: {self.work_dir}')
        self.load_snapshot()
        
        # modify/access the loaded self.cfg
        if self.user_study_mode:
            if self.cfg.sceneList[0] == "room_0_":
                self.init_pose_idxs = np.array([0, 0, 0, 7, 7, 7, 8, 8, 8])
            elif self.cfg.sceneList[0] == "apartment_2_livingroom":
                self.init_pose_idxs = np.array([1, 1, 1, 4, 4, 4, 6, 6, 6])
            elif self.cfg.sceneList[0] == "office_3_":
                self.init_pose_idxs = np.array([2, 2, 2, 5, 5, 5, 9, 9, 9])
            self.cfg.num_eval_episodes = 9
        if not hasattr(self.agent.actor, 'no_hidden'):
            self.agent.actor.no_hidden = cfg.no_hidden
            self.agent.critic.no_hidden = cfg.no_hidden
            self.agent.critic_target.no_hidden = cfg.no_hidden
        if not hasattr(self.agent, 'no_aug'):
            self.agent.no_aug = cfg.no_aug
        print(self.cfg)

        self.use_context = self.cfg.agent.use_context
        self.use_rotation = self.cfg.use_rotation
        self.use_multiprocessing = self.cfg.use_multiprocessing
        self.cfg.replay_buffer_num_workers = multiprocessing.cpu_count()
        # cfg.replay_buffer_num_workers = 0  # for debug
        drqutils.set_seed_everywhere(self.cfg.seed)

        if self.cfg.device[:4] == "cuda":
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(self.cfg.device)
        self.num_scenes = self.cfg.num_scenes
        
        self.setup()

        self.trajectories = [[] for _ in range(self.num_scenes)]
        self.np_trajectories = [[] for _ in range(self.num_scenes)]
        self.eval_trajectories = [[] for _ in range(self.num_scenes)]
        self.np_eval_trajectories = [[] for _ in range(self.num_scenes)]
        self.empty = None

    def setup(self):
        """ create logger, env, replay buffer, and video recorder"""
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create envs
        pose_dim = 5 if self.cfg.use_rotation else 3
        self.cfg.pose_dim = pose_dim
        self.step_size = self.cfg.step_size[:pose_dim]
        
        observation_spec = specs.BoundedArray(self.cfg.state_dim, np.uint8, 0, 255, 'observation')
        pose_spec = specs.Array((pose_dim,), np.float32, 'pose')
        action_spec = specs.BoundedArray((pose_dim,), np.float32, -1.0, 1.0, "action")
        data_specs = (
            observation_spec,
            pose_spec,
            action_spec,
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
        )
        
        if not self.cfg.async_:
            self.train_env = AestheticTourDMCWrapper(self.cfg, data_specs)
            self.eval_env = self.train_env  # let train and eval share env, reduce resources used
            if self.user_study_mode:
                self.train_env.num_sequences = 3

    @property
    def global_step(self):
        return self._global_step
    
    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    def eval(self):
        scene_name = self.cfg.sceneList[self.cfg.scene_index]
        init_poses = np.load(f"../{scene_name}_init_poses.npz")["pose"]
        step, episode, total_rewards = 0, 0, [0] * self.num_scenes
        eval_until_episode = drqutils.Until(self.cfg.num_eval_episodes)
        if self.fig4_mode:
            if self.scene_id == 0:
                init_poses = np.load(f"../fig4_room0_init_poses.npz")["pose"]
            elif self.scene_id == 1:
                init_poses = np.load(f"../fig4_apartment2_init_poses.npz")["pose"]
            elif self.scene_id == 2:
                init_poses = np.load(f"../fig4_office3_init_poses.npz")["pose"]
            eval_until_episode = drqutils.Until(3)
            merge_plot_inputs = []
        elif self.fig6_mode:
            if self.algo_id == 1:
                if self.scene_id == 0:
                    init_poses_idxs = np.array([8,8,8])
                elif self.scene_id == 1:
                    init_poses_idxs = np.array([0, 0, 0])
                elif self.scene_id == 2:
                    init_poses_idxs = np.array([0, 0, 0])
                elif self.scene_id == 3:
                    init_poses_idxs = np.array([0, 0, 0])
                elif self.scene_id == 4:
                    init_poses_idxs = np.array([1, 1, 1])
                elif self.scene_id == 5:
                    init_poses_idxs = np.array([3, 3, 3])

            elif self.algo_id == 2:
                if self.scene_id == 0:
                    init_poses_idxs = np.array([8, 8, 8])
                elif self.scene_id == 1:
                    init_poses_idxs = np.array([5, 5, 5])
                elif self.scene_id == 2:
                    init_poses_idxs = np.array([2, 2, 2])
                elif self.scene_id == 3:
                    init_poses_idxs = np.array([1, 1, 1])
                elif self.scene_id == 4:
                    init_poses_idxs = np.array([1, 1, 1])
                elif self.scene_id == 5:
                    init_poses_idxs = np.array([3, 3, 3])
            init_poses = init_poses[init_poses_idxs]
            eval_until_episode = drqutils.Until(3)
            merge_plot_inputs = []
        elif self.fig7_mode:
            merge_plot_inputs = []

        self.cfg.gpu_aes_obs = False
        self.cfg.aes_obs_width = 512
        self.cfg.aes_obs_height = 512
        space_mapper = SpaceMapping(scene_name)
        gymenv = HabitatSimGymWrapper(self.cfg, space_mapper)  # for plotting
        
        if self.fig7_mode or self.special:
            if self.special:
                seq1 = np.load("../figure 6 apartment0/trajectories/2268000_s0_eval0.npz")
                merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *merge_plot_inputs[0], save_fig=True, fn=f"{0}")
                seq2 = np.load("../figure 6 apartment0/trajectories/2268000_s0_eval1.npz")
                merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *merge_plot_inputs[1], save_fig=True, fn=f"{1}")
                seq2 = np.load("../figure 6 apartment0/trajectories/2268000_s0_eval2.npz")
                merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *merge_plot_inputs[2], save_fig=True, fn=f"{2}")
            elif self.algo_id == 1:
                if self.scene_id == 0:
                    if self.fig7_ablation_mode:
                        seq1 = np.load("../eval 66 33/trajectories/2943060_s0_eval0.npz")
                        merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                        seq2 = np.load("../eval 71 no smoothness 74/trajectories/2686500_s0_eval0.npz")
                        merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                    else:
                        seq1 = np.load("../eval 66 33/trajectories/2943060_s0_eval0.npz")
                        merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                        seq2 = np.load("../eval 71 no smoothness 74/trajectories/2686500_s0_eval0.npz")
                        merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                elif self.scene_id == 1:
                    seq1 = np.load("../eval 67 20/trajectories/2938575_s0_eval0.npz")
                    merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                    seq2 = np.load("../eval 83 no smoothness apartment2/trajectories/2344515_s0_eval0.npz")
                    merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                elif self.scene_id == 2:
                    seq1 = np.load("../eval 69 35/trajectories/2934015_s0_eval0.npz")
                    merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                    seq2 = np.load("../eval 84 no smoothness office3/trajectories/2205000_s0_eval0.npz")
                    merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
            elif self.algo_id == 2:
                if self.scene_id == 0:
                    # seq1 = np.load("../eval 66 33/trajectories/2943060_s0_eval0.npz")
                    # merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                    # seq2 = np.load("../eval 71 no smoothness 74/trajectories/2686500_s0_eval0.npz")
                    # merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                    pass
                elif self.scene_id == 1:
                    seq1 = np.load("../eval 5 apartment2/trajectories/2637000_s0_eval0.npz")
                    merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                    seq2 = np.load("../eval 12 no smoothness apartment2/trajectories/1444500_s0_eval0.npz")
                    merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
                elif self.scene_id == 2:
                    seq1 = np.load("../eval 6 office3/trajectories/2848500_s0_eval0.npz")
                    merge_plot_inputs.append([seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']])
                    seq2 = np.load("../eval 13 no smoothness office3/trajectories/2452500_s0_eval0.npz")
                    merge_plot_inputs.append([seq2['pose'], seq2['action'], seq2['reward'], seq2['excluding_seq'], seq2['diversity_ratio'], seq2['smoothness_ratio']])
            merge_plot_inputs = zip(*merge_plot_inputs)
            merge_plot_labels = ["sequence 0 - with smoothness", "sequence 1 - without smoothness"]
            if self.special:
                merge_plot_labels = ["sequence 0 - no excluding region", "sequence 1 - one excluding region", "sequence 2 - two excluding regions"]
            plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *merge_plot_inputs, merge_plot=True, merge_plot_labels=merge_plot_labels, save_fig=False, save_interpolate_video=False, show_img=False, no_exclusion=True, fn=f"{100}")
        elif self.teaser_mode:
            seq1 = np.load("../figure 4.1/trajectories/2727090_s0_eval0.npz", allow_pickle=True)
            inputs = [seq1['pose'], seq1['action'], seq1['reward'], seq1['excluding_seq'], seq1['diversity_ratio'], seq1['smoothness_ratio']]
            plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *inputs, save_fig=False, save_interpolate_video=False, show_img=False, teaser_background=True, fn=f"{100}")

        else:
            while eval_until_episode(episode):
                # TODO init_poses[0] for testing random excluding seqs, init_poses[0] for testing other stuff
                # excluding pose is set according to config (random or last ending pose)
                # time_steps, histories = self.eval_env.reset(to_poses=[init_poses[0]])
                init_pose = init_poses[episode]
                curr_diversity_radius = None
                if self.user_study_mode:
                    init_pose = init_poses[self.init_pose_idxs[episode]]
                    curr_diversity_radius = 1. * torch.ones((self.cfg.num_excluding_sequences, 1))
                time_steps, histories = self.eval_env.reset(to_poses=[init_pose], curr_diversity_radius=curr_diversity_radius)
                self.append_to_trajectory(time_steps, eval=True)
                while not time_steps[0].last():
                    with torch.no_grad(), drqutils.eval_mode(self.agent):
                        action = [self.agent.act(time_steps[i].observation,
                                                 time_steps[i].pose,
                                                 time_steps[i].t,
                                                 time_steps[i].excluding_seq,
                                                 time_steps[i].avg_step_size,
                                                 self.global_step,
                                                 eval_mode=True,
                                                 history=histories[i]) for i in range(self.num_scenes)]
                    time_steps, histories = self.eval_env.step(action)
                    self.append_to_trajectory(time_steps, eval=True)
                    for i in range(self.num_scenes):
                        total_rewards[i] += time_steps[i].reward
                    step += 1
                    
                np_trajectory = list(zip(*self.np_eval_trajectories[0]))
                np_trajectory = [np.stack(np_trajectory[i]) for i in range(len(np_trajectory))]
                # np.savez(fname, pose=np_trajectory[0], reward=np_trajectory[1], discount=np_trajectory[2], action=np_trajectory[3], step_size=np_trajectory[4],
                #          diversity_ratio=np_trajectory[5], excluding_seq=np_trajectory[6], smoothness_ratio=np_trajectory[7], avg_step_size=np_trajectory[8])
                poses, rewards, _, actions = np_trajectory[:4]
                diversity_ratio, excluding_seq, smoothness_ratio = np_trajectory[5:8]
                if self.fig4_mode or self.fig6_mode:
                    merge_plot_inputs.append([poses, actions, rewards, excluding_seq, diversity_ratio, smoothness_ratio])
                # if self.teaser_mode:
                #     plot3d_and_save_vid(gymenv, self.cfg.max_timestep, poses, actions, rewards, excluding_seq, diversity_ratio, smoothness_ratio, save_fig=False, show_img=False, teaser_background=True, fn=f"{episode}")
                plot3d_and_save_vid(gymenv, self.cfg.max_timestep, poses, actions, rewards, excluding_seq, diversity_ratio, smoothness_ratio, save_fig=True, fn=f"{episode}")
                
                self.plot_trajectory(eval_i=episode)
                episode += 1

            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                avg_ep_rewards = []
                for i in range(self.num_scenes):
                    avg_ep_rewards.append(total_rewards[i] / episode)
                    log(f'episode_reward{i}', avg_ep_rewards[-1])
                log(f'episode_reward', np.average(np.array(avg_ep_rewards)))
                log('episode_length', step * self.cfg.action_repeat / episode)
                log('episode', self.global_episode)
                log('step', self.global_step)
            
            if self.fig4_mode:
                merge_plot_inputs = zip(*merge_plot_inputs)
                merge_plot_labels = ["sequence 0", "sequence 1", "sequence 2"]
                plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *merge_plot_inputs, merge_plot=True, merge_plot_labels=merge_plot_labels, save_fig=False, save_interpolate_video=False, show_img=False, fn=f"{100}")
            if self.fig6_mode:
                merge_plot_inputs = zip(*merge_plot_inputs)
                merge_plot_labels = ["sequence 0 - no excluding region", "sequence 1 - one excluding region", "sequence 2 - two excluding regions"]
                plot3d_and_save_vid(gymenv, self.cfg.max_timestep, *merge_plot_inputs, merge_plot=True, merge_plot_labels=merge_plot_labels, save_fig=False, save_interpolate_video=False, show_img=False, fn=f"{100}")

    
    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'cfg', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
    
    def load_snapshot(self):
        snapshot = self.snapshot_path / self.snapshot_name
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
    
    def append_to_trajectory(self, time_steps, eval=False):
        if eval or self.global_episode % self.cfg.eval_every_episodes < self.cfg.num_eval_episodes:
            for i in range(self.num_scenes):
                if self.use_rotation:
                    tile_to_plot = (time_steps[i].aes_obs.squeeze(0),
                                         f"p: {time_steps[i].pose[:3]}\n"
                                         f"   {time_steps[i].pose[3:] * np.array([180., 90.])}\n"
                                         f"r: {time_steps[i].reward:.4f} d: {time_steps[i].diversity_ratio:.2f} s: {time_steps[i].smoothness_ratio}\n"
                                         f"a: {time_steps[i].action[:3] * self.cfg.step_size[:3]}\n"
                                         f"   {time_steps[i].action[3:] * self.cfg.step_size[3:]}")
                else:
                    tile_to_plot = (torch.tensor(time_steps[i].observation, dtype=torch.float) / 255.,
                                    f"p: {time_steps[i].pose[:3]}\n"
                                    f"r: {time_steps[i].reward:.4f}\n"
                                    f"a: {time_steps[i].action[:3] * self.cfg.step_size[:3]}")
                if eval:
                    self.eval_trajectories[i].append(tile_to_plot)
                    self.np_eval_trajectories[i].append([time_steps[i].pose, time_steps[i].reward, time_steps[i].discount, time_steps[i].action, np.array(self.cfg.step_size),
                                                         time_steps[i].diversity_ratio, time_steps[i].excluding_seq, time_steps[i].smoothness_ratio, time_steps[i].avg_step_size])
                elif self.global_episode % self.cfg.eval_every_episodes < self.cfg.num_eval_episodes:
                    self.trajectories[i].append(tile_to_plot)
                    self.np_trajectories[i].append([time_steps[i].pose, time_steps[i].reward, time_steps[i].discount, time_steps[i].action, np.array(self.cfg.step_size),
                                                    time_steps[i].diversity_ratio, time_steps[i].excluding_seq, time_steps[i].smoothness_ratio, time_steps[i].avg_step_size])

    def plot_trajectory(self, eval_i=None, evaluatation_phase=""):
        """ trajectory contains (image, camerapos) tuples of an episode"""
        trajectories = self.eval_trajectories if eval_i is not None else self.trajectories
        np_trajectories = self.np_eval_trajectories if eval_i is not None else self.np_trajectories
        # only save trajectory plot and clear trajectories list once in a while
        if eval_i is not None or self.global_episode % self.cfg.eval_every_episodes < self.cfg.num_eval_episodes:
            for s_idx, trajectory in enumerate(trajectories):
                directory = self.work_dir / "trajectories"
                Path(directory).mkdir(parents=True, exist_ok=True)
                if self.empty is None:
                    self.empty = torch.ones_like(trajectory[0][0])  # a white, empty image
                to_plot = []  # 2D list containing tensors of each image
                ncol = 5
                i = 0
                while i < len(trajectory):
                    row = []
                    for _ in range(ncol):
                        tensor = saver_utils.draw_text_tensor(trajectory[i][0], trajectory[i][1])
                        # tensor = trajectory[i][0]
                        row.append(tensor)
                        i += 1
                        if i == len(trajectory):
                            break
                    while len(row) < ncol:  # fill last row if not full
                        row.append(self.empty)
                    to_plot.append(row)
                if evaluatation_phase != "":
                    # before or after
                    fname = directory / f"s{s_idx}_{evaluatation_phase}_eval{eval_i}"
                else:
                    fname = directory / f"{self.global_step}_s{s_idx}" if eval_i is None else directory / f"{self.global_step}_s{s_idx}_eval{eval_i}"
                saver_utils.save_tensors_image(str(fname) + ".png", to_plot)
                np_trajectory = list(zip(*np_trajectories[s_idx]))
                np_trajectory = [np.stack(np_trajectory[i]) for i in range(len(np_trajectory))]
                np.savez(fname, pose=np_trajectory[0], reward=np_trajectory[1], discount=np_trajectory[2], action=np_trajectory[3], step_size=np_trajectory[4],
                         diversity_ratio=np_trajectory[5], excluding_seq=np_trajectory[6], smoothness_ratio=np_trajectory[7], avg_step_size=np_trajectory[8])
            
            # clear trajectory
            if eval_i is not None:
                self.eval_trajectories = [[] for _ in range(self.num_scenes)]
                self.np_eval_trajectories = [[] for _ in range(self.num_scenes)]
            else:
                self.trajectories = [[] for _ in range(self.num_scenes)]
                self.np_trajectories = [[] for _ in range(self.num_scenes)]


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    print("before workspace")
    from final_evaluation import Workspace as W
    print("after workspace")
    import matplotlib
    gui_env = ['QtAgg', 'TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
    for gui in gui_env:
        try:
            print("testing", gui)
            matplotlib.use(gui, force=True)
            from matplotlib import pyplot as plt
            break
        except:
            # traceback.print_exc()
            continue
    print("final_evalutaion plt backend: ", matplotlib.get_backend())
    workspace = W(cfg)
    workspace.eval()
    

if __name__ == '__main__':
    main()
