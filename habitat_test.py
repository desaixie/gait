import copy
import os
import re
import time
from copy import deepcopy
import torch.multiprocessing as mp
import ray
from collections import deque
from typing import Any, NamedTuple, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from numpy.linalg import inv
from PIL import Image
import torch
from gym import spaces
from dm_env import specs, StepType
import matplotlib.pyplot as plt
from mathutils import Matrix

import habitat_sim
from habitat_sim.agent import ActionSpec
from habitat_sim.registry import registry
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.agent.controls.default_controls import _rotate_local
from habitat_sim.scene import SceneNode
import magnum
import quaternion

# local imports
from aesthetics_model import AestheticsModel
from subproc_vec_env import SubprocVecEnv, SubprocEnv
from base_vec_env import CloudpickleWrapper
from loadBoundingBox import loadBoundingBox
import drqv2.utils as drqutils
from drqv2.replay_buffer import ReplayBufferStorage

from svox2util import pose_spherical
from pathlib import Path

class HabitatSimGymWrapper:
    # def __init__(self, max_episode, max_timestep=30, step_size=1, state_dim=(3, 84, 84),
    #              pose_dim=3, outside=False, use_rotation=False, uniform_sample=False,
    #              scene_name="room_0", space_mapper=None, gpu_device_id=0, mesh_name="mesh", use_position=True):
    def __init__(self, cfg, space_mapper=None):
        print("Initializing Habitat Simulator")
        self.space_mapper = space_mapper if space_mapper is not None else cfg.space_mapper
        self.scene_name = cfg.scene_name
        self.use_position = cfg.use_position

        self.state_dim = cfg.state_dim
        scene_filename = cfg.scene_name[:[m.start() for m in re.finditer('_', cfg.scene_name)][-1]]  # scene_filename only up to before the last _
        self.sim_settings = make_default_settings(cfg.state_dim[1], cfg.state_dim[2], scene_filename, cfg.gpu_device_id,
                                                  cfg.mesh_name, cfg.camera_fov, cfg.gpu_aes_obs, cfg.aes_obs_width, cfg.aes_obs_height)  # trim scene names for apartment
        self.sim = make_simulator_from_settings(self.sim_settings)
        self.agent = self.sim.agents[0]
        
        self.step_size = cfg.step_size
        self.pose_dim = cfg.pose_dim
        self.action_space = spaces.Box(-np.ones((cfg.pose_dim,), dtype=np.float32), np.ones((cfg.pose_dim,), dtype=np.float32), shape=(cfg.pose_dim,))
        self.max_episode = cfg.max_episode
        self.max_timestep = cfg.max_timestep
        self.i_episode = 0
        self.t = 0
        self.initialPositionList = self.GenerateInitialPosList(cfg.max_episode, outside=cfg.outside)  # same set of initial positions will be used accross different runs
        self.uniformInitialPoseList = self.GenerateInitialPosList(cfg.max_episode, outside=cfg.outside, uniform=True)  # same set of initial positions will be used accross different runs
        
        self.use_rotation = cfg.use_rotation
        self.soft_bound = cfg.soft_bound

        # dense sampling:
        self.uniform_sample = cfg.uniform_sample
        if cfg.use_rotation:
            self.rotation = np.zeros((2,), dtype=np.float32)  # keep record of rotation within current episode
            self.zero_rotation = None

        # for evaluation index
        self.evaluation_uniform_index = 0

        self.fixed_initial_pose = cfg.fixed_initial_pose
        if self.fixed_initial_pose == "None":
            self.fixed_initial_pose = None
        if self.fixed_initial_pose is not None:
            print("Using fixed initial pose")

    """ to_pose is [-1,1] normalized"""
    def reset(self, eval_i=None, uniform=False, to_pose=None, apply_filter=False, to_quat=None):
        self.t = 0
        if to_pose is not None:
            initial_pose = np.copy(to_pose)
        elif eval_i is not None:
            if self.i_episode == self.max_episode:
                self.i_episode = 0
            # initialPose = self.initialPositionList[self.i_episode]  # fixed set of initial positions
            initial_pose = np.random.rand(self.pose_dim) * 2.0 - 1.0  # random initial pose for each episode
            self.i_episode += 1
        else:
            if uniform:
                initial_pose = self.uniformInitialPoseList[eval_i]
            else:
                # randomize
                initial_pose = np.random.rand(self.pose_dim) * 2.0 - 1.0
        if self.fixed_initial_pose is not None:
            initial_pose = np.copy(self.fixed_initial_pose)
        
        # for dense sampling
        if self.uniform_sample:
            initial_pose[:3] = np.array([-1.,-1.,-1.])
            initial_pose[3:] = 0

        # convert to world coordinates
        initial_pose[:3] = self.space_mapper.MapToWorldPosition(initial_pose[:3])
        if self.use_rotation:
            initial_pose[3:] = self.space_mapper.MapToAngle(initial_pose[3:])
        initial_pose = initial_pose.astype(np.float32)

        # need to reset rotation then do rotate to the yaw, pitch angles
        if self.use_rotation and to_quat is None:  # if to_quat is not None, skip rotation and set quaternion later
            self.rotation[:] = 0.
            state = self.agent.get_state()
            state.position = np.array([0.0, 0.0, 0.0])
            state.rotation = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
            self.agent.set_state(state)
            
            action_angle = initial_pose[3:]
            self.rotation += action_angle
        
            if self.rotation[0] < -180.0:
                self.rotation[0] = self.rotation[0] + ((int)((-180.0 - self.rotation[0] + 1.0) / 360.0) + 1) * 360.0
            if self.rotation[0] > 180.0:
                self.rotation[0] = self.rotation[0] - ((int)((self.rotation[0] + 1.0 - 180.0) / 360.0) + 1) * 360.0
        
            if self.rotation[1] < -90.0:
                self.rotation[1] = -90.0
            elif self.rotation[1] >= 90.0:
                self.rotation[1] = 90.0
            _rotate_local(self.agent.scene_node, theta=self.rotation[0], axis=1)  # yaw rotation, arround y, in degrees.
            _rotate_local(self.agent.scene_node, theta=self.rotation[1], axis=0)  # pitch rotation, in degrees.
    
        # translation
        state = self.agent.get_state()
        state.position = initial_pose[:3]
        if to_quat is not None:
            # state.rotation = quaternion.as_quat_array(to_quat)
            if type(to_quat) != quaternion.quaternion:
                print(type(to_quat))
                to_quat = quaternion.as_quat_array(to_quat)
            state.rotation = to_quat

        self.agent.set_state(state)
        obs = self.sim.step("stay", apply_filter=apply_filter)
        
        # return
        img = obs['color_sensor_1st_person'][:, :, :3]  # remove the alpha channel
        aes_obs = obs["color_sensor_1st_person_aes"][:, :, :3]
        pose = self.agent.get_state().position.astype(np.float32)
        if self.use_rotation:
            pose = np.concatenate([pose, self.rotation])
        done = False

        return img, pose, done, aes_obs

    """ Important: first rotate, then translate! """
    def step(self, action, apply_filter=False):
        self.t += 1

        action = np.copy(action)  # don't modify model's output
        if not self.uniform_sample:
            action *= self.step_size
        action[:3] = self.space_mapper.MapToWorldTranslation(action[:3])
        if self.use_rotation:
            action[3:] = self.space_mapper.MapToRotation(action[3:])

        # rotation
        if self.use_rotation:
            # first reset rotation
            _rotate_local(self.agent.scene_node, theta=-self.rotation[1], axis=0)  # pitch rotation, in degrees. positive=look_left
            _rotate_local(self.agent.scene_node, theta=-self.rotation[0], axis=1)  # yaw rotation, in degrees. positive=look_up
            
            # clipping
            self.rotation += action[3:]
            if self.rotation[0] < -180.0:
                self.rotation[0] = self.rotation[0] + (int((-180.0 - self.rotation[0] + 1.0) / 360.0) + 1) * 360.0
            if self.rotation[0] > 180.0:
                self.rotation[0] = self.rotation[0] - (int((self.rotation[0] + 1.0 - 180.0) / 360.0) + 1) * 360.0
            self.rotation[1] = np.clip(self.rotation[1], -90., 90.)
            
            # then do new rotation
            _rotate_local(self.agent.scene_node, theta=self.rotation[0], axis=1)  # yaw rotation, in degrees. positive=look_up
            _rotate_local(self.agent.scene_node, theta=self.rotation[1], axis=0)  # pitch rotation, in degrees. positive=look_left

        # translation
        state = self.agent.get_state()
        if self.use_position and self.soft_bound:
            state.position += action[:3]
        else:  # hard bound, reject out of bound translation
            temp_position = state.position + action[:3]
            temp_position_norm = self.space_mapper.normalize_position(temp_position)
            temp_position_norm = np.clip(temp_position_norm, -1., 1.)
            temp_position = self.space_mapper.MapToWorldPosition(temp_position_norm)
            state.position = temp_position
        self.agent.set_state(state)
        
        # return
        obses = self.sim.step("stay", apply_filter=apply_filter)
        img = obses['color_sensor_1st_person'][:, :, :3]  # remove the alpha channel
        aes_obs = obses["color_sensor_1st_person_aes"][:, :, :3]
        pose = self.agent.get_state().position.astype(np.float32)
        if self.use_rotation:
            pose = np.concatenate([pose, self.rotation])
        done = self.t == self.max_timestep
        
        return img, pose, done, aes_obs
    
    def close(self):
        self.sim.close()
    
    def GenerateInitialPosList(self, count, outside=False, uniform=False):
        if outside:
            # (-0.3,0.3) or (0.7,1.3)
            if uniform:
                raise
            else:
                return 0.6*np.random.rand(count, 3) - 0.3 + np.random.randint(low=0, high=2, size=(count, 3))
            # return 0.3 * np.random.rand(count, 3) + np.random.randint(low=0, high=1, size=(count, 3))
        else:
            if uniform:
                side = 1. / (count ** (1. / 3))
                uniform_pos = np.mgrid[0:1.:side, 0:1.:side, 0:1.:side][:, 1:, 1:, 1:].reshape(3,-1).T
                ret = np.random.rand(uniform_pos.shape[0], self.pose_dim)
                ret[:,:3] = uniform_pos
                ret[:,3:] = ret[:,3:]*2.0-1.0
                return ret
            else:
                return np.random.rand(count, self.pose_dim)

    def switchScene(self, scene_name="room_0", space_mapper=None):
        print("Switch to a new scene.")
        self.space_mapper = space_mapper
        self.scene_name = scene_name
        
        # update simulator
        self.sim_settings = make_default_settings(self.state_dim[0], self.state_dim[1], scene_name)
        new_cfg = make_cfg(self.sim_settings)
        self.sim.close(destroy=False)
        self.sim.reconfigure(new_cfg)

        # update agent
        self.agent = self.sim.agents[0]
    
    def get_zero_rotation(self):
        if self.zero_rotation is None:
            self.zero_rotation = np.copy(self.agent.state.rotation)
        return self.zero_rotation
    
    def detectBoundaries(self):
        """ Assumes room in the shape of a box. Find corners by move in x and z directions with apply_filter=True"""
        directions = [[1,0,0], [-1,0,0], [0,0,1], [0,0,-1]]
        _, last_pos, _, _ = self.step((np.zeros((3,)), True), apply_filter=True)
        corners = []
        for d in directions:
            _, curr_pos, _, _ = self.step(np.array(d).astype(float), apply_filter=True)
            while not np.array_equal(curr_pos, last_pos):
                last_pos = curr_pos
                _, curr_pos, _, _ = self.step(np.array(d).astype(float), apply_filter=True)
            corners.append(curr_pos)
        return corners
    
    def visualizeHeight(self, start, end, output_path):
        """ Saves images when agent moves in the y direction, in order to find room's height upper bound."""
        step_size = (end - start) / 30
        img, pos, _, _ = self.step((np.array([0., start-step_size*2, 0.]), True), apply_filter=False)
        for i in range(30):
            save_np_img(os.path.join(output_path, f"{i}.png"), img)
            action = np.array([0., 1., 0.]) * step_size
            # action = np.ones((3,)) * 0.25
            img, pos, done, obs = self.step(action, apply_filter=False)


class HabitatSimDMCWrapper:
    """imitate drqv2.dmc.ExtendedTimeStepWrapper"""
    # def __init__(self, max_episode, max_timestep=30, step_size=1, state_dim=(128,128,3), pose_dim=3,
    #              outside=False, aesthetics_model=None, use_rotation=False, use_context=False, hist_len=0, uniform_sample=False,
    #              boundingbox_dir="", sceneList=["room_0"], scene_index=0, gpu_device_id=0, mesh_name="mesh", use_position=True):
    def __init__(self, cfg, scene_index, aesthetics_model=None):
        self.max_episode = cfg.max_episode
        self.max_timestep = cfg.max_timestep
        self.step_size = np.array(cfg.step_size)
        self.state_dim = cfg.state_dim
        self.pose_dim = cfg.pose_dim
        self.outside = cfg.outside
        self.use_rotation = cfg.use_rotation
        self.uniform_sample = cfg.uniform_sample
        self.use_position = cfg.use_position

        assert cfg.sceneList is not None, "The scene list is None!"
        self.scene_list = cfg.sceneList
        self.boundingbox_dir = cfg.boundingbox_dir
        self.space_mapper = SpaceMapping(cfg.sceneList[scene_index])
        # self.env = HabitatSimGymWrapper(max_episode, max_timestep, step_size, state_dim, pose_dim,
        #                                 outside, use_rotation, uniform_sample, sceneList[scene_index],
        #                                 self.space_mapper, gpu_device_id, mesh_name, use_position)
        self.env = HabitatSimGymWrapper(cfg, self.space_mapper)
        self.aesthetics_model = aesthetics_model if aesthetics_model is not None else cfg.aesthetics_model
        self.observation_spec = specs.BoundedArray(cfg.state_dim, np.uint8, 0, 255, 'observation')
        self.pose_spec = specs.Array((cfg.pose_dim,), np.float32, 'pose')
        self.action_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "action")
        # self.gamma = 0.99
        self.discount = cfg.discount
        self.use_rotation = cfg.use_rotation
        self.map_to_positive = cfg.map_to_positive
        self.min_reward_abs = abs(self.space_mapper.maxmin_scores[1])
        self.negative_reward = cfg.negative_reward

        self.use_context = cfg.use_context
        self.hist_len = cfg.agent.context_history_length
        if cfg.use_context:
            # history contains list of np array of concatenated action+reward+obs+pos
            self.history_obs = deque(maxlen=self.hist_len)
            self.history_others = deque(maxlen=self.hist_len)
            self.hist_obs_dim = cfg.state_dim
            self.hist_others_dim = cfg.pose_dim * 2 + 1

    def reset(self, eval_i=None, uniformSampling=False, to_pose=None):
        # self.discount = 1.

        img, pos, done, aes_obs = self.env.reset(eval_i, uniform=uniformSampling, to_pose=to_pose)  # if call super().reset() here, inside Gym.reset(), self is DMC instead of Gym, so self.step((action, True)) will call DMC.step() instead of Gym.step()
        img = np.moveaxis(img, -1, 0)
        pos[:3] = self.space_mapper.normalize_position(pos[:3])
        if self.use_rotation:
            pos[3:] = self.space_mapper.normalize_angle(pos[3:])
        t = self.env.t / self.max_timestep  # normalized

        step_type = StepType.FIRST
        action = np.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)
        reward = 0.

        aesthetic_img = aes_obs.float() / 255.0 # (240,240,3) GPU tensor
        aesthetic_img = aesthetic_img.unsqueeze(0).permute(0, 3, 1, 2)  # from NHWC to NCHW
        time_step = ExtendedTimeStep(step_type, reward, self.discount, img, pos, action, t=t, aes_obs=aesthetic_img)

        history = None
        if self.use_context:
            # reset history
            self.history_obs.clear()
            self.history_others.clear()
            self.history_obs.append(img)
            if self.use_position:
                self.history_others.append(np.concatenate([action, np.full((1,), reward, dtype=np.float32), pos]))
            else:
                self.history_others.append(np.concatenate([action, np.full((1,), reward, dtype=np.float32)]))
            history = [np.array(self.history_others), np.array(self.history_obs)]
        return time_step, history
    
    def switchScene(self, newSceneName=None):
        if newSceneName != None:
            self.space_mapper = SpaceMapping(self.boundingbox_dir, newSceneName,np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([180.0, 90.0]), np.array([45.0, 45.0]))
            self.env.switchScene(scene_name=newSceneName, space_mapper=self.space_mapper)
        else:
            self.scene_index = self.scene_index + 1
            self.space_mapper = SpaceMapping(self.boundingbox_dir, self.scene_list[self.scene_index],np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), np.array([180.0, 90.0]), np.array([45.0, 45.0]))
            self.env.switchScene(scene_name=self.scene_list[self.scene_index], space_mapper=self.space_mapper)

    def step(self, action, apply_filter=False):
        img, pos, done, aes_obs = self.env.step(action, apply_filter=apply_filter)
        img = np.moveaxis(img, -1, 0)
        pos[:3] = self.space_mapper.normalize_position(pos[:3])
        if self.use_rotation:
            pos[3:] = self.space_mapper.normalize_angle(pos[3:])
        t = self.env.t / self.max_timestep  # normalized

        step_type = StepType.LAST if done else StepType.MID
        
        # self.discount *= self.gamma
        
        # calculate reward
        aesthetic_img = aes_obs.float() / 255.0 # (240,240,3) GPU tensor
        aesthetic_img = aesthetic_img.unsqueeze(0).permute(0, 3, 1, 2)  # from NHWC to NCHW
        currscore, reward = self.aesthetics_model(aesthetic_img, pos[:3], done)
        # map to positive
        if self.map_to_positive and reward != self.negative_reward:
            reward += self.min_reward_abs
        # aesthetic score normalization
        # no normalization for single train
        #if reward != -10:
        #    reward = self.space_mapper.normalize_score(reward)
        
        discount = 0. if done else self.discount
        time_step = ExtendedTimeStep(step_type, reward, discount, img, pos, action, t=t, aes_obs=aesthetic_img)
        
        history = None
        if self.use_context:
            # don't include current obs, action, reward into history
            history = [np.array(self.history_others), np.array(self.history_obs)]
            # append to history
            self.history_obs.append(img)
            if self.use_position:
                self.history_others.append(np.concatenate([action, np.full((1,), reward, dtype=np.float32), pos]))
            else:
                self.history_others.append(np.concatenate([action, np.full((1,), reward, dtype=np.float32)]))
        return time_step, history
    
    def close(self):
        self.env.close()
    

class MultiSceneWrapper:
    def __init__(self, cfg, data_specs=None):
    # def __init__(self, max_episode, max_timestep=30, step_size=1, state_dim=(128, 128, 3), pose_dim=3,
    #              outside=False, use_rotation=False, use_context=False, hist_len=0, uniform_sample=False,
    #              boundingbox_dir="", num_scenes=1, sceneList=["room_0"], scene_index=-1, use_multiprocessing=False,
    #              GPU_IDs=[], log_dir=None, data_specs=None, mesh_name="mesh"):
        self.num_scenes = cfg.num_scenes
        self.scene_names = cfg.sceneList
        self.observation_spec = specs.BoundedArray(cfg.state_dim, np.uint8, 0, 255, 'observation')
        self.pose_spec = specs.Array((cfg.pose_dim,), np.float32, 'pose')
        self.action_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "action")
        
        self.use_multiprocessing = cfg.use_multiprocessing
        if cfg.use_multiprocessing:
    
            # SubprocEnvs would receive the same i. Solution is to use mp.current_process() in SubprocEnv
            self.envs = SubprocVecEnv([lambda : SubprocEnv(max_episode, max_timestep, step_size, state_dim,
                                                      position_dim, outside, use_rotation, use_context,
                                                      hist_len, uniform_sample, boundingbox_dir, num_scenes, sceneList, i,
                                                           GPU_IDs, work_dir, data_specs, mesh_name)
                                       for i in range(self.num_scenes)])
        
        else:
            self.envs = []
            aesthetics_model = AestheticsModel(negative_reward=-10.)
            gpu_device_id = 0
            if cfg.num_scenes == 1:
                # single scene
                cfg.scene_name = cfg.sceneList[cfg.scene_index]
                self.envs.append(HabitatSimDMCWrapper(cfg, cfg.scene_index, aesthetics_model))
                # self.envs.append(HabitatSimDMCWrapper(max_episode, max_timestep, step_size, state_dim,
                #                                       pose_dim, outside, aesthetics_model, use_rotation, use_context,
                #                                       hist_len, uniform_sample, boundingbox_dir, sceneList, scene_index, gpu_device_id, mesh_name))
            else:
                gpu_device_ids = cfg.gpu_device_id
                for i, scene_name in enumerate(self.scene_names):
                    cfg.scene_name = cfg.sceneList[cfg.scene_index]
                    cfg.gpu_device_id = gpu_device_ids[i]
                    self.envs.append(HabitatSimDMCWrapper(cfg, i, aesthetics_model))
                    # self.envs.append(HabitatSimDMCWrapper(max_episode, max_timestep, step_size, state_dim,
                    #                                       pose_dim, outside, aesthetics_model, use_rotation, use_context,
                    #                                       hist_len, uniform_sample, boundingbox_dir, sceneList, i, gpu_device_id, mesh_name))
                # self.envs[-1].env.sim.close()
                

    def reset(self, eval_i=None, uniformSampling=False, to_poses=None):
        """ Assumes scenes have episodes of same length and always reset together
            returns time_steps: list, histories: list"""
        if self.use_multiprocessing:
            return self.envs.reset(eval_i, uniformSampling, to_pose=to_poses)
        if to_poses is not None:
            return zip(*[e.reset(eval_i, uniformSampling, to_pose=to_poses[i]) for i, e in enumerate(self.envs)])
        return zip(*[e.reset(eval_i, uniformSampling, to_pose=to_poses) for i, e in enumerate(self.envs)])

    def step(self, actions, apply_filter=False):
        """ returns time_steps: list, histories: list """
        if self.use_multiprocessing:
            return self.envs.step(actions)
        return zip(*[e.step(actions[i], apply_filter) for i, e in enumerate(self.envs)])
    
    def close(self):
        for e in self.envs:
            e.close()
    

class AestheticTourDMCWrapper:
    def __init__(self, cfg, data_specs=None):
        self.cfg = cfg
        # if cfg.max_timestep != 31:
        #     raise NotImplementedError
        self.pose_dim = cfg.pose_dim
        self.num_scenes = cfg.num_scenes
        self.scene_names = cfg.sceneList
        self.observation_spec = specs.BoundedArray(cfg.state_dim, np.uint8, 0, 255, 'observation')
        self.pose_spec = specs.Array((cfg.pose_dim,), np.float32, 'pose')
        self.t_spec = specs.Array((1,), np.float32, 't')
        self.action_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "action")
        self.pose_shape = cfg.pose_dim
        d_pose_shape = cfg.pose_dim
        if cfg.distance_obs:
            d_pose_shape += 1
        if cfg.rand_diversity_radius:
            d_pose_shape += 1
        self.excluding_seq_spec = specs.BoundedArray((cfg.num_excluding_sequences, d_pose_shape,), np.float32, -1.0, 1.0, "excluding_seq")
        self.avg_step_size_spec = specs.BoundedArray((cfg.pose_dim,), np.float32, -1.0, 1.0, "avg_step_size")
        self.env = MultiSceneWrapper(cfg, data_specs)
        
        self.diversity = cfg.diversity
        self.num_excluding_sequences = cfg.num_excluding_sequences
        self.num_sequences = self.num_excluding_sequences + 1
        self.sequence_i = 0
        self.excluding_seqs = [np.ones((self.pose_dim,), dtype=np.float32) * -1.5 for _ in range(self.num_scenes)]  # -1.5 gives >1 ratio with a pose at [-1.]*5
        self.diversity_radius = cfg.diversity_radius
        self.distance_obs = cfg.distance_obs
        self.rand_exc_pose = cfg.rand_exc_pose
        self.rand_diversity_radius = cfg.rand_diversity_radius
        
        self.smoothness = cfg.smoothness
        self.step_sizes = [[np.zeros((5,), dtype=np.float32)] for _ in range(self.num_scenes)]
        self.smoothness_threshold = cfg.smoothness_threshold
        self.smoothness_window = cfg.smoothness_window
        self.position_only_smoothness = cfg.position_only_smoothness
        self.separate_step_sizes = cfg.separate_step_sizes
        self.weighted_window = cfg.weighted_window
        if self.smoothness_window > 0:
            self.step_size_dim = 3 if self.position_only_smoothness else cfg.pose_dim
            self.avg_step_size_spec = specs.BoundedArray((self.smoothness_window, self.step_size_dim), np.float32, -1.0, 1.0, "avg_step_size")
        self.position_orientation_separate = cfg.position_orientation_separate
        if self.position_orientation_separate:
            assert self.position_only_smoothness == False

    def reset(self, eval_i=None, to_poses=None, curr_excluding_seqs=None, curr_sequence_i=None, curr_step_sizes=None, curr_diversity_radius=None):
        self.sequence_i += 1
        if self.sequence_i == 1:
            self.excluding_seqs = [np.ones((self.num_excluding_sequences, self.pose_dim,), dtype=np.float32) * -1.5 for _ in range(self.num_scenes)]  # -1.5 gives >1 ratio with a pose at [-1.]*5
        if curr_excluding_seqs is not None:  # for CMA-ES
            self.excluding_seqs = curr_excluding_seqs.copy()
        elif self.diversity and self.rand_exc_pose:
            self.excluding_seqs = [np.random.rand(self.num_excluding_sequences, self.pose_dim).astype(np.float32) * 2. - 1. for _ in range(self.num_scenes)]
            
        if curr_sequence_i is not None:
            self.sequence_i = curr_sequence_i
        self.step_sizes = [[np.zeros((5,), dtype=np.float32)] for _ in range(self.num_scenes)]
        if curr_step_sizes is not None:
            self.step_sizes = copy.deepcopy(curr_step_sizes)
        if self.rand_diversity_radius:
            self.diversity_radius = np.random.rand(self.num_excluding_sequences, 1).astype(np.float32) + 0.3
        if curr_diversity_radius is not None:
            self.diversity_radius = curr_diversity_radius

        time_steps, histories = self.env.reset(eval_i=eval_i, to_poses=to_poses)
        ret_time_steps = []
        for i, t_s in enumerate(time_steps):  # num scenes
            if self.diversity:
                exc_seq = self.excluding_seqs[i]
                if self.distance_obs:
                    difs = t_s.pose - self.excluding_seqs[i]  # (5,), (4,5)
                    distances = np.linalg.norm(difs, axis=1, keepdims=True)
                    exc_seq = np.concatenate([exc_seq, distances], axis=1)
                if self.rand_diversity_radius:
                    exc_seq = np.concatenate([exc_seq, self.diversity_radius], axis=1)
            else:
                exc_seq = None

            avg_step_size = np.zeros((self.smoothness_window, self.step_size_dim), dtype=np.float32) if self.smoothness_window > 0 else np.zeros(self.pose_dim, dtype=np.float32)
            ret_time_steps.append(ExtendedTimeStep(t_s.step_type, t_s.reward, t_s.discount, t_s.observation, t_s.pose, t_s.action, t_s.t, exc_seq, t_s.aes_obs, 1., avg_step_size, 1.))
        return ret_time_steps, histories

    def step(self, actions):
        time_steps, histories = self.env.step(actions)
        ret_time_steps = []
        for i, t_s in enumerate(time_steps):
            diversity_ratio = 1.
            exc_seq = None
            smoothness_ratio = 1.
            avg_step_size = np.zeros((self.smoothness_window, self.step_size_dim), dtype=np.float32) if self.smoothness_window > 0 else np.zeros(self.pose_dim, dtype=np.float32)
            
            # diversity
            if self.diversity:
                difs = t_s.pose - self.excluding_seqs[i]  # (5,), (4,5)
                distances = np.linalg.norm(difs, axis=1, keepdims=True)
                diversity_ratio = float(self._diversity_reward_ratio(distances))
                exc_seq = self.excluding_seqs[i]
                if self.distance_obs:
                    exc_seq = np.concatenate([exc_seq, distances], axis=1)
                if self.rand_diversity_radius:
                    exc_seq = np.concatenate([exc_seq, self.diversity_radius], axis=1)

                if time_steps[i].last():
                    if self.sequence_i < self.num_sequences:  # record pose to be excluded in the following sequences
                        if not self.rand_exc_pose:
                            self.excluding_seqs[i][self.sequence_i - 1] = t_s.pose
                    else:  # reset every num_sequences
                        self.sequence_i = 0

            # smoothness
            if self.smoothness:
                step_sizes = self.step_sizes[i]  # issue was here, step_sizes after stack became (1,1,5) instead of (1,5)
                if self.smoothness_window > 0:
                    step_sizes = np.stack(step_sizes[-self.smoothness_window:])  # (window, 5)
                    diffs = np.expand_dims(actions[i], axis=0) - step_sizes  # broadcasted to (window, 5)
                    if self.position_only_smoothness:
                        assert len(diffs.shape) == 2 and diffs.shape[-1] == self.pose_shape
                        diffs = diffs[:, :3]
                        step_sizes = step_sizes[:, :3]
                    if self.position_orientation_separate:
                        step_size_diff = np.linalg.norm(diffs[:, :3], axis=-1), np.linalg.norm(diffs[:, 3:], axis=-1)  # (window,)
                        step_sizes_in = step_sizes[:, :3], step_sizes[:, 3:]
                        smoothness_ratio = self._separate_position_orientation_smoothness_reward_ratio(step_size_diff, step_sizes_in)
                        # action_norm = np.linalg.norm(actions[i])
                        # smoothness_ratio = self._smoothness_v2(step_size_diff, action_norm)
                    else:
                        step_size_diff = np.linalg.norm(diffs, axis=-1)
                        smoothness_ratio = self._separate_smoothness_reward_ratio(step_size_diff, step_sizes)
                    # for time_step
                    avg_step_size = step_sizes
                    num_padding = self.smoothness_window - len(step_sizes)
                    avg_step_size = np.concatenate([np.zeros((num_padding, step_sizes.shape[-1]), dtype=np.float32), avg_step_size], axis=0)
                else:
                    if len(self.step_sizes[i]) == 0:
                        avg_step_size = np.zeros(5, dtype=np.float32)
                    else:
                        avg_step_size = np.sum(step_sizes, axis=0) / len(self.step_sizes[i])  # (5,)
                    diff = actions[i] - avg_step_size
                    if self.position_only_smoothness:
                        assert diff.shape == (self.pose_shape,)
                        diff = diff[:3]
                    step_size_diff = np.linalg.norm(diff)
                    smoothness_ratio = float(self._smoothness_reward_ratio(step_size_diff, avg_step_size))

            ori_reward = t_s.reward
            if ori_reward == self.cfg.negative_reward:
                aes_tour_reward = ori_reward  # don't modify negative out of bound reward
            else:
                if self.cfg.map_to_positive:
                    # this increase the integral of gaussian function by 1.333 times
                    aes_tour_reward = ori_reward * (0.5 + 0.5 * diversity_ratio) * (0.5 + 0.5 * smoothness_ratio)
                else:
                    if self.cfg.new_reward:
                        # new_reward = old_reward - abs(old_reward) * (1 - diversity_ratio * smoothness_ratio)
                        if ori_reward >= 0:
                            aes_tour_reward = ori_reward * diversity_ratio * smoothness_ratio  # old
                        else:
                            aes_tour_reward = ori_reward - 0.5 * (1 - diversity_ratio * smoothness_ratio) * abs(ori_reward)
                            # aes_tour_reward = ori_reward
                    else:  # old reward, r * D * S
                        aes_tour_reward = ori_reward * diversity_ratio * smoothness_ratio  # old
            ret_time_steps.append(ExtendedTimeStep(t_s.step_type, aes_tour_reward, t_s.discount, t_s.observation, t_s.pose, t_s.action, t_s.t, exc_seq, t_s.aes_obs, diversity_ratio, avg_step_size, smoothness_ratio))
            
            if self.smoothness:
                self.step_sizes[i].append(t_s.action)

        return ret_time_steps, histories
        
    def close(self):
        self.env.close()
    
    def _diversity_reward_ratio(self, distances):
        if self.cfg.avg_distance:
            # if self.rand_diversity_radius:
            #     return torch.minimum((distances / self.diversity_radius), 1.).mean() # mean((3,) / (3,))
            # return np.minimum(distances / self.diversity_radius, 1.).mean()
            sigma = self.diversity_radius / 3.
            gaussian_diversity = 1 - np.exp(- np.square(distances) / (2 * np.square(sigma)))
            return gaussian_diversity.mean()

        if self.rand_diversity_radius:
            return min((distances / self.diversity_radius).min(), 1.)  # (3,) / (3,)
        return min(distances.min() / self.diversity_radius, 1.)
    
    def _smoothness_reward_ratio(self, distance, avg):
        # return min(self.smoothness_threshold / diff, 1.)
        
        a = 1.  # amplitude of reward ratio is 1.
        d = np.square(distance)  # (x-b)^2
        c = max(np.linalg.norm(avg) / 2., 0.1)  # avoid dividing by zero
        return a * np.exp(- d / (2. * np.square(c)))
    
    def _separate_smoothness_reward_ratio(self, distances, step_sizes, rotation=False):
        """ distances: (window,), step_sizes: (window, 5)"""
        a = 1.  # amplitude of reward ratio is 1.
        d = np.square(distances)  # (x-b)^2  (window,)
        denom = 1. if rotation else 2.
        c = np.maximum(np.linalg.norm(step_sizes, axis=-1) / denom, 0.1)  # avoid dividing by zero  (window,)
        gaussians = a * np.exp(- d / (2. * np.square(c)))  # (window,)
        if self.weighted_window == "None":
            # return np.average(gaussians)
            return (gaussians.min() + gaussians.mean()) / 2.
        else:
            ret = 1.
            offset = self.smoothness_window - len(step_sizes)
            for i in range(len(step_sizes)):
                ret *= gaussians[i] / self.weighted_window[i+offset] + (1. - 1. / self.weighted_window[i+offset])
            return ret
    
    def _separate_position_orientation_smoothness_reward_ratio(self, distances, step_sizes):
        rotation = [False, True]
        return np.average([self._separate_smoothness_reward_ratio(distances[i], step_sizes[i], rotation[i]) for i in range(2)])

    def _smoothness_v2(self, distances, action_norm):
        """ don't consider low smoothness for zero-action. Therefore, use constant sigma wrt smoothness_radius
            at radius, smoothness is zero"""
        a = 1.  # amplitude of reward ratio is 1.
        d = np.square(distances[0])  # (x-b)^2  (window,)
        c = self.cfg.smoothness_radius_trans / 3.
        gaussian_trans = a * np.exp(- d / (2. * np.square(c))).mean()  # (window,)
        
        a = 1.  # amplitude of reward ratio is 1.
        d = np.square(distances[1])  # (x-b)^2  (window,)
        c = self.cfg.smoothness_radius_rot / 3.
        gaussian_rot = a * np.exp(- d / (2. * np.square(c))).mean()  # (window,)
        gaussian_zero = self._smoothness_avoid_zero(action_norm)
        return gaussian_zero * (gaussian_trans + gaussian_rot) / 2.
        
    def _smoothness_avoid_zero(self, action_norm):
        a = 1.  # amplitude of reward ratio is 1.
        c = self.cfg.zero_radius / 3.
        gaussian = 1 - a * np.exp(- np.square(action_norm) / (2. * np.square(c)))  # (window,)
        return gaussian

class Runner:
    """ Wrapper of a single environment inside a SubprocVecEnv. Creates a DMCWrapper Env,
    an aesthetics_model, and a replay_storage in a sub process given scene_index and GPU_IDs"""
    
    # beginning part of __init__ arguments is the same as MultiSceneWrapper
    def __init__(self, max_episode, max_timestep=30, step_size=1, state_dim=(128, 128, 3), position_dim=3,
                 outside=False, use_rotation=False, use_context=False, hist_len=0, uniform_sample=False,
                 boundingbox_dir="", num_scenes=1, sceneList=["room_0"], scene_index=-1, GPU_IDs=None,
                 work_dir=None, data_specs=None, mesh_name="mesh", main_agent=None, cfg=None, agent_id=None,
                 use_position=True):
        
        # Spreads scenes evenly on available GPUs
        scene_index = mp.current_process()._identity[0] - 1
        gpu_device_id = GPU_IDs[scene_index % len(GPU_IDs)]
        device = f"cuda:{gpu_device_id}"
        cfg.agent.device = device
        print(f"i: {scene_index}, GPU: {gpu_device_id}, scne_name: {sceneList[scene_index]}")
        
        if use_position:
            aesthetics_model = AestheticsModel(negative_reward=-10., device=device)
        else:
            aesthetics_model = AestheticsModel(negative_reward=None, device=device)
        self.env = HabitatSimDMCWrapper(max_episode, max_timestep, step_size, state_dim,
                                        position_dim, outside, aesthetics_model, use_rotation, use_context,
                                        hist_len, uniform_sample, boundingbox_dir, sceneList, scene_index,
                                        gpu_device_id, mesh_name, use_position)
        # create replay storage
        # TODO replay_storage and env in the same thread for now, since main thread updates before
        #  we receive STEP, action
        self.replay_storage = ReplayBufferStorage(data_specs, work_dir / 'buffer', num_scenes=1, scene_index=scene_index)

        #self.agent = main_agent
        self.main_encoder, self.main_actor = main_agent
        self.agent_id = agent_id
        from rlcam_drqv2_mql import make_agent  # avoid circular import
        self.agent = make_agent(
            self.env.observation_spec,
            self.env.pose_spec,
            self.env.action_spec,
            cfg.agent)
        
        self.global_step = 0
        self.global_step_per_episode = cfg.agent.update_every_steps * cfg.num_async_update_iters

    def reset(self, eval_i=None, uniformSampling=False):
        """ Assumes scenes have episodes of same length and always reset together
            returns time_steps: list, histories: list"""
        return self.env.reset(eval_i, uniformSampling)
    
    def step(self, action, apply_filter=False):
        """ returns time_steps: list, histories: list """
        return self.env.step(action, apply_filter)
    
    def add_to_storage(self, time_step):
        self.replay_storage.add([time_step])
    
    def run_episodes(self, eval_i=None, n_episodes=1):
        time_steps = []
        total_episode_reward = 0
        for _ in range(n_episodes):
            time_step, history = self.env.reset(eval_i)
            self.replay_storage.add([time_step])
            if eval_i is not None:
                time_steps.append(time_step)
            while not time_step.last():
                # sample action
                with torch.no_grad(), drqutils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            time_step.pose,
                                            self.global_step,
                                            eval_mode=eval_i is not None,
                                            history=history)
            
                # take env step
                time_step, history = self.env.step(action)
                if eval_i is not None:
                    time_steps.append(time_step)
                
                total_episode_reward += time_step.reward
                self.replay_storage.add([time_step])  # TODO async add
                
            # sync every episode
            self.sync_network()
            self.global_step += self.global_step_per_episode
        if eval_i is not None:
            return time_steps, total_episode_reward / n_episodes
        return total_episode_reward / n_episodes
    
    def sync_network(self):
        self.agent.encoder.load_state_dict(self.main_encoder.state_dict())
        self.agent.actor.load_state_dict(self.main_actor.state_dict())

    def close(self):
        self.env.close()


def make_async_runners(max_episode, max_timestep, step_size, state_dim, position_dim, outside,
                       use_rotation, use_context, hist_len, uniform_sample, boundingbox_dir, num_scenes,
                       sceneList, scene_index, GPU_IDs, work_dir, data_specs, mesh_name, agent, cfg,
                       agent_id, use_position):
    """ Avoid referencing self when passing self.max_episode, ... to avoid 'TypeError: can't pickle _thread.lock objects' """
    return AsyncRunners(num_scenes,
                        (max_episode, max_timestep, step_size, state_dim,
                        position_dim, outside, use_rotation, use_context,
                        hist_len, uniform_sample, boundingbox_dir,
                        num_scenes, sceneList, -1,
                        GPU_IDs, work_dir, data_specs, mesh_name, agent, deepcopy(cfg), agent_id, use_position),
            # used spawn here. forkserver makes copies even with main_encoder.share_memory(). fork causes CUDA reinitialize error
                        "spawn")


class AsyncRunners:
    # def __init__(self, env_fns: List[Callable[[], Runner]], start_method: Optional[str] = None):
    def __init__(self, n_envs, runner_args, start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
    
        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            # used spawn here. forkserver makes copies even with main_encoder.share_memory(). fork causes CUDA reinitialize error
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)
    
        # one pipe for each SubprocEnv
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, runner_args)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
            
    def all_ready(self):
        """ Call this before the first run_episode, to wait for all children to finish their __init__"""
        results = [remote.recv() for remote in self.remotes]
        for r in results:
            if r != "ready":
                raise

    def run_episodes(self, eval_i=None, n_episodes=1):
        command = "run_episodes" if eval_i is None else "run_eval_episodes"
        command_data = (command, (eval_i, n_episodes))
        for remote in self.remotes:
            remote.send(command_data)
        self.waiting = True
    
    def finish_episodes(self, eval_i=None):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        if eval_i is not None:
            time_stepss, episode_rewards = zip(*results)
            return time_stepss, episode_rewards
        else:
            return results
        
    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


def _worker(remote, parent_remote,
            runner_args) -> None:
    parent_remote.close()
    runner = Runner(*runner_args)
    remote.send("ready")
    # id of encoder and critic from parent and from main_encoder reference is not the same, but loading state_dict from them works
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "run_episodes":
                episode_reward = runner.run_episodes(*data)
                remote.send(episode_reward)
                runner.sync_network()
            elif cmd == "run_eval_episodes":
                time_steps, episode_reward = runner.run_episodes(*data)
                remote.send((time_steps, episode_reward))
            elif cmd == "seed":
                raise
            elif cmd == "close":
                runner.close()
                remote.close()
                break
            elif cmd == "env_method":
                method = getattr(runner, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(runner, data))
            elif cmd == "set_attr":
                remote.send(setattr(runner, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class ExtendedTimeStep(NamedTuple, tuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    pose: Any
    action: Any
    t: Any = None
    excluding_seq: Any = None
    aes_obs: Any = None
    diversity_ratio: Any = None
    avg_step_size: Any = None
    smoothness_ratio: Any = None

    def first(self):
        return self.step_type == StepType.FIRST
    
    def mid(self):
        return self.step_type == StepType.MID
    
    def last(self):
        return self.step_type == StepType.LAST
    
    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)  # This fails, as getattr("name") returns idx, and getattr(idx) fails
        else:
            return tuple.__getitem__(self, attr)  # default


class SpaceMapping:
    def __init__(self, sceneName):
        boundingbox_dirs = ["denseSampling", "../../../denseSampling"]
        boundingbox_dir = None
        for d in boundingbox_dirs:
            if os.path.isdir(d):
                boundingbox_dir = d
                break
        if boundingbox_dir is None:
            raise Exception(f"Did not find boundingbox dir, cwd: {os.getcwd()}")
        left_corner_n, room_size_n, xAxis_n, yAxis_n, zAxis_n, maxmin_scores = loadBoundingBox(fpath=boundingbox_dir, sceneName=sceneName) # Peggy: new bounding box
        self.left_corner_n = left_corner_n
        self.room_size_n = room_size_n
        self.xAxis_n = xAxis_n  # (3,)
        self.yAxis_n = yAxis_n
        self.zAxis_n = zAxis_n
        self.maxmin_scores = maxmin_scores

        self.rotation_range_angle = np.array([180., 90.])
        self.absolute_range_angle = np.array([180., 90.])

        # left_corner_t, room_size_t, xAxis_t, yAxis_t, zAxis_t = torch.tensor(left_corner_n, dtype=torch.float, device=device), \
        #     torch.tensor(room_size_n, dtype=torch.float, device=device),\
        #     torch.tensor(xAxis_n, dtype=torch.float, device=device),\
        #     torch.tensor(yAxis_n, dtype=torch.float, device=device),\
        #     torch.tensor(zAxis_n, dtype=torch.float, device=device)
        # self.left_corner_t = left_corner_t
        # self.room_size_t = room_size_t
        # self.xAxis_t = xAxis_t
        # self.yAxis_t = yAxis_t
        # self.zAxis_t = zAxis_t

        self.m_n = np.stack([xAxis_n,yAxis_n,zAxis_n])  # (3,3)
        self.inv_m_n = inv(self.m_n)
        # self.m_t = torch.tensor(self.m_n, dtype=torch.float, device=device)
        # self.inv_m_t = torch.tensor(self.m_t, dtype=torch.float, device=device)

    def normalize_position(self, position):
        """ returns np array pose (3,) normalized to [-1,1]. Please note input pose is in the old world coordinate system."""
        """ First we map it to the new world coordinates. Then normalize it."""
        # print(f"in normalizing pose: before {pose}", end='')
        # ret = np.array( [np.dot( (pose - self.left_corner_n), self.xAxis_n) / self.room_size_n[0] * 2 - 1.0, \
        #     np.dot( (pose - self.left_corner_n), self.yAxis_n) / self.room_size_n[1] * 2 - 1.0, \
        #     np.dot( (pose - self.left_corner_n), self.zAxis_n) / self.room_size_n[2] * 2 - 1.0])
        
        ret_new = (np.dot((position.reshape(1, 3) - self.left_corner_n), self.m_n.T) / self.room_size_n * 2.0 - 1.0).reshape(-1)
        # _a = np.equal(ret, ret_new)
        # print(f"post pose {pose}")
        return ret_new

    def normalize_angle(self, angle):
        return angle / self.absolute_range_angle

    def normalize_rotation(self, rotation):
        return rotation / self.rotation_range_angle

    def normalize_translation(self, translation):
        """ normalize the world translation into [-1, +1]"""
        trans_new_coord = np.dot(self.m_n, translation)
        trans = np.array([trans_new_coord[0]/self.room_size_n[0] * 2, trans_new_coord[1]/self.room_size_n[1] * 2, trans_new_coord[2]/self.room_size_n[2] * 2])
        return trans

    def MapToWorldPosition(self, campos):
        """ Map a normalized pose into the new world coordinates."""
        if isinstance(campos, np.ndarray):
            if len(campos.shape) > 1:
                # result = np.stack([np.dot(self.inv_m_n, ((campos[i]+1.0)/2.0 * self.room_size_n).T) + self.left_corner_n for i in range(campos.shape[0])])
                new_result = np.dot((campos+1.0)/2.0 * self.room_size_n, self.inv_m_n.T) + np.expand_dims(self.left_corner_n, 0)
                # _a =  np.equal(result, new_result)
                result = new_result
            else:
                result = np.dot(self.inv_m_n, (campos + 1.0) / 2.0 * self.room_size_n) + self.left_corner_n
            # print("campos", campos, "m", m_n, "inverse m ", inv_m_n, "invm*pos", np.dot(inv_m_n, (campos+1.0)/2.0)," result ", result)
            return result
        result = torch.dot(self.inv_m_t, (campos+1.0)/2.0 * self.room_size_t) + self.left_corner_t
        return result

    def MapToWorldTranslation(self, norm_v):
        if isinstance(norm_v, np.ndarray):
            result = np.dot(self.inv_m_n, norm_v / 2 * self.room_size_n)
            # print("normalized v", norm_v, "m", m_n, "inverse m ", inv_m_n, "invm*normv", norm_v*room_size_n," result ", result)
            return result
        result = torch.dot(self.inv_m_t, norm_v * self.room_size_t)
        return result

    def MapToAngle(self, rotation):
        return rotation * self.absolute_range_angle # world angle is within [-45, 45]

    def MapToRotation(self, rotation):
        return rotation * self.rotation_range_angle # world angle is within [-45, 45]
    
    def normalize_score(self, score):
        return (score - self.maxmin_scores[1]) * 2. / (self.maxmin_scores[0] - self.maxmin_scores[1]) - 1.
    
    def MapToWorldScore(self, normalized_score):
        return (normalized_score + 1) / 2 * (self.maxmin_scores[0] - self.maxmin_scores[1]) + self.maxmin_scores[1]




@registry.register_move_fn(name="stay", body_action=True)
class Stay(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec) -> None:
        pass


def make_default_settings(width=128, height=128, sceneName="room_0", gpu_device_id=0, mesh_name="mesh",
                          camera_fov=90, gpu_aes_obs=True, aes_obs_width=240, aes_obs_height=240):
    # Replica should be downloaded to aestheticview/habitat_data/Replica
    # "scene": "habitat_data/Replica/room_0/habitat/mesh_semantic.ply",  # Scene path
    # scene_paths = ["habitat_data/Replica/"+sceneName+"/mesh.ply", "../../../habitat_data/Replica/"+sceneName+"/mesh.ply"]
    scene_paths = [f"habitat_data/Replica/{sceneName}/{mesh_name}.ply", f"../../../habitat_data/Replica/{sceneName}/{mesh_name}.ply",
                   f"habitat_data/Replica/{sceneName}/habitat/{mesh_name}.ply", f"../../../habitat_data/Replica/{sceneName}/habitat/{mesh_name}.ply",
                   f"habitat_data/gibson/{sceneName}.glb", f"../../../habitat_data/gibson/{sceneName}.glb"]
    scene = None
    for s in scene_paths:
        if os.path.isfile(s):
            scene = s
            break
    if scene is None:
        raise Exception(f"Did not find scene file, cwd: {os.getcwd()}, sceneName: {sceneName}, mesh_name: {mesh_name}")
    settings = {
        # changing w and h only affects aspect ratio and image resolution, does not change FOV
        "width": width,  # for RL network encoder input
        "height": height,
        "aes_obs_width": aes_obs_width,  # for aesthetic model input
        "aes_obs_height": aes_obs_height,
        "scene": scene,
        "default_agent": 0,
        "sensor_height": 0,  # Since we disable apply_filter and move in y direction.
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
        "seed": 1,
        "enable_physics": False,
        "gpu_device_id": gpu_device_id,
        "camera_fov": camera_fov,
        "gpu_aes_obs": gpu_aes_obs
    }
    return settings


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = settings["gpu_device_id"]
    if "scene_dataset" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Specify the location of the scene dataset
    if "scene_dataset_config" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config"]
    if "override_scene_light_defaults" in settings:
        sim_cfg.override_scene_light_defaults = settings[
            "override_scene_light_defaults"
        ]
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]
    
    # Note: all sensors must have the same resolution
    sensor_specs = []
    color_sensor_1st_person_spec = habitat_sim.CameraSensorSpec()
    color_sensor_1st_person_spec.uuid = "color_sensor_1st_person"
    color_sensor_1st_person_spec.gpu2gpu_transfer = False  # the 84*84 img should be a CPU np ndarray
    color_sensor_1st_person_spec.hfov = magnum.Deg(settings["camera_fov"])
    color_sensor_1st_person_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_1st_person_spec.resolution = [
        settings["height"],
        settings["width"],
    ]
    color_sensor_1st_person_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_1st_person_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    color_sensor_1st_person_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_1st_person_spec)
    
    # Used for Aesthetic Model Input, which requires 240*240
    color_sensor_1st_person_aes_spec = habitat_sim.CameraSensorSpec()
    color_sensor_1st_person_aes_spec.uuid = "color_sensor_1st_person_aes"
    color_sensor_1st_person_aes_spec.gpu2gpu_transfer = settings["gpu_aes_obs"]  # the 240*240 img should be a GPU torch tensor
    color_sensor_1st_person_aes_spec.hfov = magnum.Deg(settings["camera_fov"])
    color_sensor_1st_person_aes_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_1st_person_aes_spec.resolution = [
        settings["aes_obs_height"],
        settings["aes_obs_width"],
    ]
    color_sensor_1st_person_aes_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_1st_person_aes_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    color_sensor_1st_person_aes_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_1st_person_aes_spec)
    
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space["stay"] = ActionSpec("stay")
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_simulator_from_settings(sim_settings):
    cfg = make_cfg(sim_settings)
    # clean-up the current simulator instance if it exists
    # global obj_attr_mgr
    # global prim_attr_mgr
    # global stage_attr_mgr
    # global rigid_obj_mgr
    # global metadata_mediator

    # initialize the simulator
    sim = habitat_sim.Simulator(cfg)
    print("Habitat-sim simulator is constructed")
    # Managers of various Attributes templates
    # obj_attr_mgr = sim.get_object_template_manager()
    # prim_attr_mgr = sim.get_asset_template_manager()
    # stage_attr_mgr = sim.get_stage_template_manager()
    # # Manager providing access to rigid objects
    # rigid_obj_mgr = sim.get_rigid_object_manager()
    # # get metadata_mediator
    # metadata_mediator = sim.metadata_mediator
    return sim


def save_np_img(fname, x):
    img = Image.fromarray(x)
    img.save(fname)


def findRoomBoundaries():
    env = HabitatSimGymWrapper(1)
    corners = env.detectBoundaries()
    print(f"corners: {corners}")
    
    output_path = "save_img/"
    env.visualizeHeight(-1.3273629, 3, output_path)
    # start is #2, end is #18 (both 2 steps from out of bound view
    # step size 0.14424543. #18 is start+16*step_size = 0.98056398


class _env():
    def __init__(self):
        self.action_space = spaces.Box(-1., 1., shape=(5,))


