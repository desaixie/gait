import copy
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from pathlib import Path
import hydra
import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True, sign=' ', floatmode='fixed')

# local import
import saver_utils
import drqv2.utils as drqutils
from aesthetics_model import AestheticsModel
from trajectory_visualize import plot3d_and_save_vid

simulation = "habitat-sim"
from habitat_test import HabitatSimGymWrapper, SpaceMapping, HabitatSimDMCWrapper, AestheticTourDMCWrapper


class Workspace:
    def __init__(self, cfg):
        self.max_episode = 100  # number of initial positions
        self.work_dir = Path.cwd()
        
        # read cfg and modify
        self.cfg = cfg
        drqutils.set_seed_everywhere(cfg.seed)
        
        if cfg.device[:4] == "cuda":
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(cfg.device)
        self.num_scenes = self.cfg.num_scenes
        self.position_dim = 5 if self.cfg.use_rotation else 3
        self.step_size = self.cfg.step_size[:self.position_dim]
        if self.cfg.sceneList[0] == "room_0_":
            self.init_pose_idxs = np.array([0, 0, 0, 7, 7, 7, 8, 8, 8])
        elif self.cfg.sceneList[0] == "apartment_2_livingroom":
            self.init_pose_idxs = np.array([1, 1, 1, 4, 4, 4, 6, 6, 6])
        elif self.cfg.sceneList[0] == "office_3_":
            self.init_pose_idxs = np.array([2, 2, 2, 5, 5, 5, 9, 9, 9])
        self.cfg.num_eval_episodes = 9
        self.train_env = AestheticTourDMCWrapper(self.cfg)
        self.eval_env = AestheticTourDMCWrapper(self.cfg)

        # gym env for plotting
        self.cfg.gpu_aes_obs = False
        self.cfg.aes_obs_width = 512
        self.cfg.aes_obs_height = 512
        scene_name = self.cfg.sceneList[self.cfg.scene_index]
        space_mapper = SpaceMapping(scene_name)
        self.gymenv = HabitatSimGymWrapper(self.cfg, space_mapper)  # for plotting
        self.num_sequences = 3
        self.train_env.num_sequences = 3
        self.eval_env.num_sequences = 3

        assert self.num_scenes == 1
        self.curr_pose = np.zeros((self.position_dim,), dtype=np.float32)
        self.curr_excluding_seqs = [np.ones((self.cfg.num_excluding_sequences, self.position_dim,), dtype=np.float32) * -1.5 for _ in range(self.num_scenes)]  # -1.5 gives >1 ratio with a pose at [-1.]*5
        self.curr_sequence_i = 1

        self.eval_trajectories = [[] for _ in range(self.num_scenes)]
        self.np_eval_trajectories = [[] for _ in range(self.num_scenes)]
        self.empty = None

    def MPC_random_step(self, a, curr_pose, curr_excluding_seqs, curr_sequence_i, curr_step_sizes):
        curr_diversity_radius = None
        curr_diversity_radius = 1. * torch.ones((self.cfg.num_excluding_sequences, 1))
        self.train_env.reset(to_poses=[curr_pose],
                             curr_excluding_seqs=curr_excluding_seqs, curr_sequence_i=curr_sequence_i, curr_step_sizes=curr_step_sizes, curr_diversity_radius=curr_diversity_radius)
        time_steps, _ = self.train_env.step([a])  # expects actions for num_scenes
        return time_steps[0].pose, time_steps[0].reward, self.train_env.excluding_seqs.copy(), self.train_env.sequence_i, copy.deepcopy(self.train_env.step_sizes)


    def optimize(self):
        starttime = time.time()
        
        scene_name = self.cfg.sceneList[self.cfg.scene_index]
        init_poses = np.load(f"../{scene_name}_init_poses.npz")["pose"]
        for i_eval in range(self.cfg.num_eval_episodes):
            print(f"\n Episode {i_eval}")
            ep_starttime = time.time()
            init_pose = init_poses[i_eval]
            curr_diversity_radius = None
            init_pose = init_poses[self.init_pose_idxs[i_eval]]
            curr_diversity_radius = 1. * torch.ones((self.cfg.num_excluding_sequences, 1))
            time_steps, histories = self.train_env.reset(to_poses=[init_pose], curr_excluding_seqs=self.curr_excluding_seqs, curr_sequence_i=self.curr_sequence_i, curr_diversity_radius=curr_diversity_radius)
            # time_steps, _ = self.train_env.reset()  # random initial pose
            curr_pose = time_steps[0].pose.copy()
            curr_excluding_seqs = self.train_env.excluding_seqs.copy()
            curr_sequence_i = self.train_env.sequence_i
            curr_step_sizes = copy.deepcopy(self.train_env.step_sizes)  # list. step_sizes: past 3 actions for smoothness
            poses, actions, rewards = [], [], []
            poses.append(time_steps[0].pose)
            actions.append(time_steps[0].action)
            rewards.append(time_steps[0].reward)
            print(f"\nstep 0: p: {time_steps[0].pose}, a: {time_steps[0].action}, r: {time_steps[0].reward}, exc_poses: {time_steps[0].excluding_seq}, seq_i: {curr_sequence_i}, recent actions: {curr_step_sizes}, "
                  f"d: {time_steps[0].diversity_ratio}, s: {time_steps[0].smoothness_ratio}\n")
            
            def DFS_helper(height, curr_pose, curr_excluding_seqs, curr_sequence_i, curr_step_sizes, first_action, cumulative_reward):
                cumulative_rewards = []
                for i_action in range(self.cfg.num_actions):
                    action = np.random.rand(5) * 2. - 1.
                    new_pose, reward, new_excluding_seqs, new_sequence_i, new_step_sizes = self.MPC_random_step(action, curr_pose, curr_excluding_seqs, curr_sequence_i, curr_step_sizes)

                    new_cumulative_reward = cumulative_reward + reward
                    if height == 1:
                        cumulative_rewards.append(new_cumulative_reward)
                    else:
                        cumulative_rewards.append(DFS_helper(height - 1, new_pose, new_excluding_seqs, new_sequence_i, new_step_sizes, first_action, new_cumulative_reward))
                return np.max(cumulative_rewards)
            
            for i_step in range(self.cfg.max_timestep):
                # MPC
                # while not done
                    # for unroll step in k
                        # for a in num actions
                            # sample random action in action space
                            # obs, reward = dynamics_model.step(action)
                            # record (s,a,r)
                            # record sum reward so far
                    # find argmax action of sum reward so far
                    # obs, reward, done = env.step(action)
                
                # predict reward for each action
                first_actions = []
                height = self.cfg.mpc_horizon
                cumulative_rewards = []
                for _ in range(self.cfg.num_actions):
                    first_action = np.random.rand(5) * 2. - 1.
                    first_actions.append(first_action)
                    new_pose, reward, new_excluding_seqs, new_sequence_i, new_step_sizes = self.MPC_random_step(first_action, curr_pose, curr_excluding_seqs, curr_sequence_i, curr_step_sizes)
                
                    cumulative_rewards.append(DFS_helper(height - 1, new_pose, new_excluding_seqs, new_sequence_i, new_step_sizes, first_action, reward))
                    
                # pick MPC action
                mpc_action = first_actions[np.argmax(cumulative_rewards)]
                
                # step
                _, _ = self.train_env.reset(to_poses=[curr_pose],
                                            curr_excluding_seqs=curr_excluding_seqs, curr_sequence_i=curr_sequence_i, curr_step_sizes=curr_step_sizes, curr_diversity_radius=curr_diversity_radius)
                time_steps, _ = self.train_env.step([mpc_action])
                
                # save copy
                reward = time_steps[0].reward
                curr_pose = time_steps[0].pose.copy()
                curr_excluding_seqs = self.train_env.excluding_seqs.copy()
                curr_sequence_i = self.train_env.sequence_i
                curr_step_sizes = copy.deepcopy(self.train_env.step_sizes)  # list
                print(f"\nstep {i_step+1}: p: {self.curr_pose}, a: {mpc_action}, r: {reward}, exc_poses: {time_steps[0].excluding_seq}, seq_i: {curr_sequence_i}, recent actions: {curr_step_sizes}, "
                      f"d: {time_steps[0].diversity_ratio}, s: {time_steps[0].smoothness_ratio}\n")
                
                poses.append(curr_pose)
                actions.append(mpc_action)
                rewards.append(reward)

            # save
            Path("mpc_trajectories").mkdir(parents=True, exist_ok=True)
            fname = f"mpc_trajectories/s{0}_eval{i_eval}"
            np.savez(fname, pose=poses, action=actions, reward=rewards)
            print(f"Episode {i_eval} finished for {self.cfg.max_timestep} actions, return: {np.sum(rewards)}, time used: {time.time() - ep_starttime}\n")
            self.verify_trajectory(i_eval, poses, actions, rewards)
            
            # for next sequence, because the part in env.step for last() action and in the next reset are not run
            if self.curr_sequence_i < self.num_sequences:  # record pose to be excluded in the following sequences
                self.curr_excluding_seqs[0][self.curr_sequence_i - 1] = self.curr_pose
                self.curr_sequence_i += 1
            else:  # reset every num_sequences
                self.curr_sequence_i = 1
                self.curr_excluding_seqs = [np.ones((self.cfg.num_excluding_sequences, self.position_dim,), dtype=np.float32) * -1.5 for _ in range(self.num_scenes)]  # -1.5 gives >1 ratio with a pose at [-1.]*5

        print(f"total time for {self.cfg.num_eval_episodes} episodes: {time.time() - starttime}")
        self.train_env.close()
    
    """ Revisit the trajectory, verify rewards and save plot and npz"""
    def verify_trajectory(self, i_eval, poses, actions, rewards):
        # traj = np.load(self.cfg.trajectory_path + "/trajectory.npz")
        # poses = traj["pose"]
        # actions = traj["action"]
        # rewards = traj["reward"]
        
        # revisit traj
        curr_diversity_radius = None
        curr_diversity_radius = 1. * torch.ones((self.cfg.num_excluding_sequences, 1))
        time_steps, _ = self.eval_env.reset(to_poses=[poses[0]], curr_diversity_radius=curr_diversity_radius)
        self.append_to_trajectory(time_steps)
        for i in range(self.cfg.max_timestep):
            time_steps, _ = self.eval_env.step([actions[i+1]])
            self.append_to_trajectory(time_steps)
            gap = abs(rewards[i + 1] - time_steps[0].reward)
            if gap > 0.01:
                print(f"Step {i+1}: reward gap {gap} at pose {time_steps[0].pose} {poses[i + 1]}")

        np_trajectory = list(zip(*self.np_eval_trajectories[0]))
        np_trajectory = [np.stack(np_trajectory[i]) for i in range(len(np_trajectory))]
        # np.savez(fname, pose=np_trajectory[0], reward=np_trajectory[1], discount=np_trajectory[2], action=np_trajectory[3], step_size=np_trajectory[4],
        #          diversity_ratio=np_trajectory[5], excluding_seq=np_trajectory[6], smoothness_ratio=np_trajectory[7], avg_step_size=np_trajectory[8])
        poses, rewards, _, actions = np_trajectory[:4]
        diversity_ratio, excluding_seq, smoothness_ratio = np_trajectory[5:8]
        plot3d_and_save_vid(self.gymenv, self.cfg.max_timestep, poses, actions, rewards, excluding_seq, diversity_ratio, smoothness_ratio, save_fig=True, fn=f"{i_eval}")
        self.plot_trajectory(i_eval)

    def append_to_trajectory(self, time_steps, eval=True):
        if eval:
            for i in range(self.num_scenes):
                if self.cfg.use_rotation:
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

    def plot_trajectory(self, eval_i=0):
        """ trajectory contains (image, camerapos) tuples of an episode"""
        trajectories = self.eval_trajectories
        np_trajectories = self.np_eval_trajectories
        # only save trajectory plot and clear trajectories list once in a while
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
            fname = directory / f"s{s_idx}_eval{eval_i}"
            saver_utils.save_tensors_image(str(fname) + ".png", to_plot)
            np_trajectory = list(zip(*np_trajectories[s_idx]))
            np_trajectory = [np.stack(np_trajectory[i]) for i in range(len(np_trajectory))]
            np.savez(fname, pose=np_trajectory[0], reward=np_trajectory[1], discount=np_trajectory[2], action=np_trajectory[3], step_size=np_trajectory[4],
                     diversity_ratio=np_trajectory[5], excluding_seq=np_trajectory[6], smoothness_ratio=np_trajectory[7], avg_step_size=np_trajectory[8])
        
            # clear trajectory
            self.eval_trajectories = [[] for _ in range(self.num_scenes)]
            self.np_eval_trajectories = [[] for _ in range(self.num_scenes)]


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from MPC import Workspace as W
    workspace = W(cfg)
    workspace.optimize()
    
        
if __name__ == '__main__':
    main()
