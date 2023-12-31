import time
import traceback
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from torch import multiprocessing
from pathlib import Path
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


import ray
import hydra
import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True, sign=' ', floatmode='fixed')
from dm_env import specs  # https://github.com/deepmind/dm_env/blob/master/dm_env/_environment.py

# local import
import drqv2_net
import drqv2.utils as drqutils
from drqv2.logger import Logger
from drqv2.replay_buffer import ReplayBufferStorage, make_replay_loader
import saver_utils
from habitat_test import Runner, AsyncRunners, MultiSceneWrapper, make_async_runners, AestheticTourDMCWrapper


def make_agent(obs_spec, pos_spec, action_spec, cfg):
    """ Created from config.yaml agent block"""
    cfg.obs_shape = obs_spec.shape
    cfg.pos_shape = pos_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg, rank=0, storage=None):
        print(cfg)
        self.max_episode = 100  # number of initial positions
        # ray
        self.rank = rank
        self.storage = storage

        self.work_dir = Path.cwd()
        self.evaluating = cfg.evaluating
        if cfg.load_snapshot:
            self.snapshot_path = Path(cfg.snapshot_dir)
            if not (cfg.evaluating or cfg.finetuning):
                # Continue training in snapshot_dir, or evaluating or finetuning in new dir created by hydra
                self.work_dir = self.snapshot_path
            assert (self.snapshot_path / 'snapshot.pt').exists(), f"The specified snapshot is not found: {self.snapshot_path}"
            print(f'Resuming. Loaded snapshot at : {self.snapshot_path / "snapshot.pt"}')
        print(f'workspace: {self.work_dir}')
        
        # read cfg and modify
        self.cfg = cfg
        self.use_context = cfg.agent.use_context
        self.use_rotation = self.cfg.use_rotation
        self.use_multiprocessing = self.cfg.use_multiprocessing
        cfg.replay_buffer_num_workers = multiprocessing.cpu_count()
        if self.evaluating and cfg.replay_buffer_num_workers > self.cfg.num_evaluation_episodes:
            # avoid empty _episode_fns in a worker
            cfg.replay_buffer_num_workers = self.cfg.num_evaluation_episodes
        # cfg.replay_buffer_num_workers = 0  # for debug
        if self.evaluating:
            # self.cfg.max_timestep = int(self.cfg.max_timestep / 3)
            cfg.evaluation_batch_size = self.cfg.num_evaluation_episodes * self.cfg.max_timestep
        drqutils.set_seed_everywhere(cfg.seed)

        # have to comment out this (disable with if doesn't work), or Workspace can't be pickled by Ray.
        # if cfg.device[:4] == "cuda" and not self.cfg.ray:
        #     torch.backends.cudnn.benchmark = True
        self.device = torch.device(cfg.device)
        self.num_scenes = self.cfg.num_scenes
        
        if self.evaluating:
            self.evaluation_setup()
        else:
            self.setup()

        if cfg.load_snapshot:
            # from snapshot load agent, time, global_step, global_episode
            self.load_snapshot()
            if self.cfg.finetuning:
                self.timer = drqutils.Timer()
                self._global_step = 0
                self._global_episode = 0
        else:
            pose_dim = 5 if self.cfg.use_rotation else 3
            observation_spec = specs.BoundedArray(self.cfg.state_dim, np.uint8, 0, 255, 'observation')
            pose_spec = specs.Array((pose_dim,), np.float32, 'pose')
            action_spec = specs.BoundedArray((pose_dim,), np.float32, -1.0, 1.0, "action")
            d_pose_shape = pose_dim
            if cfg.distance_obs:
                d_pose_shape += 1
            excluding_seq_spec = specs.BoundedArray((cfg.num_excluding_sequences, d_pose_shape,), np.float32, -1.0, 1.0, "excluding_seq")
            self.agent = make_agent(
                observation_spec,
                pose_spec,
                action_spec,
                self.cfg.agent)
            if cfg.ray:
                self.agent = self.agent.cpu()
            self.timer = drqutils.Timer()
            self._global_step = 0
            self._global_episode = 0
            
            if self.use_multiprocessing and self.cfg.async_:
                data_specs = (
                    observation_spec,
                    pose_spec,
                    action_spec,
                    excluding_seq_spec,
                    specs.Array((1,), np.float32, 'reward'),
                    specs.Array((1,), np.float32, 'discount'),
                )
                self.agent.encoder.share_memory()
                self.agent.actor.share_memory()
                #self.agent.critic.share_memory()
                #self.agent.critic_target.share_memory()
                self.train_runners = make_async_runners(self.max_episode, self.cfg.max_timestep, np.array(self.step_size), self.cfg.state_dim,
                                                pose_dim, False, self.cfg.use_rotation, self.cfg.use_context,
                                                self.cfg.context_history_length, self.cfg.uniform_sample, self.cfg.boundingBox_dir,
                                                self.cfg.num_scenes, self.cfg.sceneList, self.cfg.scene_index,
                                                self.cfg.GPU_IDs, self.work_dir, data_specs, self.cfg.mesh_name,
                                                (self.agent.encoder, self.agent.actor), self.cfg,
                                                (id(self.agent.encoder), id(self.agent.actor)), self.cfg.use_position)
                                                #self.cfg.GPU_IDs, self.log_dir, data_specs, self.cfg.mesh_name, self.agent, self.cfg, id(self.agent))
                self.eval_runners = self.train_runners

        self.trajectories = [[] for _ in range(self.num_scenes)]
        self.np_trajectories = [[] for _ in range(self.num_scenes)]
        self.eval_trajectories = [[] for _ in range(self.num_scenes)]
        self.np_eval_trajectories = [[] for _ in range(self.num_scenes)]
        self.empty = None
        self.best_eval_reward = -np.inf

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
            self.train_env = AestheticTourDMCWrapper(self.cfg)
                # self.max_episode, step_size=np.array(self.step_size), state_dim=self.cfg.state_dim,
                # pose_dim=pose_dim, use_rotation=self.cfg.use_rotation,
                # use_context=self.cfg.agent.use_context, hist_len=self.cfg.agent.context_history_length,
                # uniform_sample=False, boundingbox_dir=self.cfg.boundingBox_dir, num_scenes=self.cfg.num_scenes,
                # sceneList=self.cfg.sceneList, scene_index=self.cfg.scene_index, use_multiprocessing=self.cfg.use_multiprocessing,
                # GPU_IDs=self.cfg.GPU_IDs, log_dir=self.log_dir, data_specs=data_specs, mesh_name=self.cfg.mesh_name)
            # multiple Habitat-sim at the same time works, only for habitat-sim built from source
            self.eval_env = self.train_env  # let train and eval share env, reduce resources used
            # self.eval_env = habitat_test.HabitatSimDMCWrapper(self.max_episode, max_action=max_action, state_dim=state_dim)
            
        if not self.cfg.use_multiprocessing:
            # replay_storage will be created in each SubprocEnv
            # create replay buffer
            data_specs = [
                self.train_env.observation_spec,
                self.train_env.pose_spec,
                self.train_env.t_spec,
                self.train_env.action_spec,
                specs.Array((1,), np.float32, 'reward'),
                specs.Array((1,), np.float32, 'discount'),
            ]
            if self.cfg.diversity:
                data_specs.insert(-3, self.train_env.excluding_seq_spec)
            if self.cfg.smoothness:
                data_specs.insert(-3, self.train_env.avg_step_size_spec)

            if self.cfg.double_single_train:
                self.replay_storage = ReplayBufferStorage(tuple(data_specs),
                                                          self.work_dir / 'buffer', self.num_scenes, scene_index=0)
            elif self.cfg.ray:
                self.replay_storage = ReplayBufferStorage(tuple(data_specs),
                                                          self.work_dir / 'buffer', self.num_scenes, scene_index=self.rank)
            else:
                self.replay_storage = ReplayBufferStorage(tuple(data_specs),
                                                          self.work_dir / 'buffer', self.num_scenes)

        # if not self.cfg.ray:
        if self.cfg.double_single_train:
            replay_loader_n_scenes = 1
        elif self.cfg.ray:
            replay_loader_n_scenes = self.cfg.num_ray_gpus
        else:
            replay_loader_n_scenes = self.cfg.num_scenes
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.use_context,
            self.cfg.agent.context_history_length, replay_loader_n_scenes, self.cfg.use_position,
            self.cfg.diversity, self.cfg.smoothness, False)
        self._replay_iter = None

    def evaluation_setup(self):
        """ create logger, env, replay buffer, and video recorder"""
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        if simulation == "blendertorch-gym":
            raise
        elif simulation == "habitat-sim":
            position_dim = 5 if self.cfg.use_rotation else 3
            self.step_size = self.cfg.step_size[:position_dim]
            self.evaluation_env = MultiSceneWrapper(
                self.max_episode, max_timestep=self.cfg.max_timestep, step_size=np.array(self.step_size), state_dim=self.cfg.state_dim,
                position_dim=position_dim, use_rotation=self.cfg.use_rotation,
                use_context=self.cfg.agent.use_context, hist_len=self.cfg.agent.context_history_length,
                uniform_sample=False, boundingbox_dir=self.cfg.boundingBox_dir, sceneList=self.cfg.sceneList, scene_index=self.cfg.scene_index)
        # create replay buffer
        data_specs = (
            self.evaluation_env.observation_spec,
            self.evaluation_env.pose_spec,
            self.evaluation_env.action_spec,
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount'),
        )
    
        self.evaluation_replay_storage = ReplayBufferStorage(data_specs,
                                                             self.work_dir / 'buffer', self.num_scenes)
        self.evaluation_replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size, self.cfg.evaluation_batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.use_context, self.cfg.agent.context_history_length)
        self._evaluation_replay_iter = None
    
        self.metatrain_replay_storage = ReplayBufferStorage(data_specs,
                                                            self.snapshot_path / 'buffer', self.num_scenes)
        self.metatrain_replay_loader = make_replay_loader(
            self.snapshot_path / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size*self.cfg.main_snap_bsize_mult, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.use_context, self.cfg.agent.context_history_length)
        self._metatrain_replay_iter = None

    @property
    def global_step(self):
        return self._global_step
    
    @property
    def global_episode(self):
        return self._global_episode
    
    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def evaluation_replay_iter(self):
        if self._evaluation_replay_iter is None:
            self._evaluation_replay_iter = iter(self.evaluation_replay_loader)
        return self._evaluation_replay_iter
    
    @property
    def metatrain_replay_iter(self):
        if self._metatrain_replay_iter is None:
            self._metatrain_replay_iter = iter(self.metatrain_replay_loader)
        return self._metatrain_replay_iter
    
        
    def eval(self):
        step, episode, total_rewards = 0, 0, [0] * self.num_scenes
        eval_until_episode = drqutils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_steps, histories = self.eval_env.reset(eval_i=episode)
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
                
            self.plot_trajectory(eval_i=episode)
            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            avg_ep_rewards = []
            for i in range(self.num_scenes):
                avg_ep_rewards.append(total_rewards[i] / episode)
                log(f'episode_reward{i}', avg_ep_rewards[-1])
            ep_reward = np.average(np.array(avg_ep_rewards))
            log(f'episode_reward', ep_reward)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
        return ep_reward
    
    def train(self):
        # predicates
        train_until_step = drqutils.Until(self.cfg.num_train_frames,
                                          self.cfg.action_repeat)
        seed_until_step = drqutils.Until(self.cfg.num_seed_frames,
                                         self.cfg.action_repeat)
        eval_every_episodes = drqutils.Every(self.cfg.eval_every_episodes,
                                         self.cfg.action_repeat)
        
        episode_step, episode_rewards = 0, [0] * self.num_scenes
        last_global_step = 0
        training_start_global_step = -1
        last_model_index = 0
        last_eval_index = 0
        local_episode = 0
        training_step_counter = 0
        time_steps, histories = self.train_env.reset()
        if not self.use_multiprocessing:  # time_steps are added to storage in SubprocEnv
            self.replay_storage.add(time_steps)
        metrics = None
        while train_until_step(self.global_step):
            if time_steps[0].last():
                if self.cfg.ray:
                    local_episode += 1
                    if local_episode % self.cfg.selfplay_step_sync_local_episode == 0:
                        self.storage.incr_selfplay_counter.remote(self.cfg.max_timestep*self.cfg.selfplay_step_sync_local_episode)
                        self._global_step = ray.get(self.storage.get_selfplay_counter.remote())
                        self._global_episode = self._global_step // self.cfg.max_timestep
                        training_step_counter = ray.get(self.storage.get_training_counter.remote())
                        
                    if training_start_global_step == -1:
                        if training_step_counter > 0:
                            training_start_global_step = self.global_step
                    else:  # training started
                        # sync model weights
                        if local_episode % self.cfg.ray_model_sync_local_episode == 0:
                            weights = ray.get(self.storage.get_weights.remote())
                            self.agent.set_weights(weights)
                            self.agent.to(self.device)
                            # model.eval()

                        while (self.global_step - training_start_global_step) - self.cfg.agent.update_every_steps * training_step_counter > self.cfg.max_timestep * 100:
                            if self.rank == 0:
                                print("selfplay too fast")
                            time.sleep(1)
                            self._global_step = ray.get(self.storage.get_selfplay_counter.remote())
                            self._global_episode = self._global_step // self.cfg.max_timestep
                            training_step_counter = ray.get(self.storage.get_training_counter.remote())
                else:
                    self._global_episode += 1
                    
                # wait until all the metrics schema is populated
                if metrics is not None or \
                        (self.cfg.ray and self.rank == 0 and local_episode % self.cfg.selfplay_step_sync_local_episode == 0):  # ray, no metrics here
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        if self.cfg.ray:
                            log('fps', (self.global_step - last_global_step) / elapsed_time)
                            last_global_step = self.global_step
                        else:
                            log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        ep_rewards = []
                        for i in range(self.num_scenes):
                            ep_rewards.append(episode_rewards[i])
                            log(f'episode_reward{i}', ep_rewards[-1])
                        log(f'episode_reward', np.average(np.array(ep_rewards)))
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                        log('env_num', self.num_scenes)

                # try to evaluate
                new_eval_index = self.global_episode // self.cfg.eval_every_episodes
                if (not self.cfg.ray and eval_every_episodes(self.global_episode)) or \
                        (self.cfg.ray and new_eval_index > last_eval_index and self.rank == 0):
                    last_eval_index = new_eval_index
                    self.logger.log('eval_total_time', self.timer.total_time(),
                                    self.global_frame)
                    eval_ep_reward = self.eval()
                    
                    # try to save snapshot
                    if self.cfg.save_snapshot:
                        fn = f'snapshot_{self._global_step}.pt'
                        if eval_ep_reward > self.best_eval_reward:
                            self.best_eval_reward = eval_ep_reward
                            self.save_snapshot(fn=fn, best=True)
                        # don't save if not best
                        # else:
                        #     self.save_snapshot(fn=fn, best=False)

                # reset env
                time_steps, histories = self.train_env.reset()
                if not self.use_multiprocessing:  # time_steps are added to storage in SubprocEnv
                    self.replay_storage.add(time_steps)
                # plot previous trajectory, and add initial observation of new trajectory
                if self.rank == 0:
                    self.plot_trajectory()
                    self.append_to_trajectory(time_steps)
                episode_step = 0
                episode_rewards = [0] * self.num_scenes
            
            # sample action
            with torch.no_grad(), drqutils.eval_mode(self.agent):
                action = [self.agent.act(time_steps[i].observation,
                                         time_steps[i].pose,
                                         time_steps[i].t,
                                         time_steps[i].excluding_seq,
                                         time_steps[i].avg_step_size,
                                         self.global_step,
                                         eval_mode=False,
                                         history=histories[i]) for i in range(self.num_scenes)]

            # try to update the agent. only update here if not using ray multi-gpu
            if not seed_until_step(self.global_step) and not self.cfg.ray:
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            # take env step
            time_steps, histories = self.train_env.step(action)
            for i in range(self.num_scenes):
                episode_rewards[i] += time_steps[i].reward
            if not self.use_multiprocessing:  # time_steps are added to storage in SubprocEnv
                self.replay_storage.add(time_steps)
            episode_step += 1
            if self.cfg.ray:
                self._global_step += self.cfg.num_ray_gpus  # approximate global step
            elif self.cfg.double_single_train and self.num_scenes == 2:
                self._global_step += 2
            else:
                self._global_step += 1
            if self.rank == 0:
                self.append_to_trajectory(time_steps)
                
        if self.rank == 0:
            self.save_snapshot('final.pt')
        print("Training Done")
        
    def ray_train(self):
        try:
            # torch.backends.cudnn.benchmark = True
            del self.replay_loader
            self.agent.cuda()
            # self.train_env = AestheticTourDMCWrapper(self.cfg)
            self.train()
        except:
            traceback.print_exc()

    def async_train(self):
        train_until_step = drqutils.Until(self.cfg.num_train_frames,
                                          self.cfg.action_repeat)
        seed_until_step = drqutils.Until(self.cfg.num_seed_frames,
                                         self.cfg.action_repeat)
        eval_every_episodes = drqutils.Every(self.cfg.eval_every_episodes,
                                             self.cfg.action_repeat)

        self.train_runners.all_ready()
        self.train_runners.run_episodes(n_episodes=self.cfg.num_async_episodes)  # TODO run this async
        metrics = None
        while train_until_step(self.global_step):
            wait_start_time = time.time()
            # async simulation steps, but still syncs every x episode
            episode_rewards = self.train_runners.finish_episodes()
            
            self._global_episode += 1 * self.cfg.num_async_episodes
            #episode_step = self.cfg.max_timestep * self.cfg.num_async_episodes
            # use update times to determine global steps, affecting progression speed for total num_async_update_iterstraining frames and decay schedule
            episode_step = self.cfg.agent.update_every_steps * self.cfg.num_async_update_iters * self.cfg.num_async_episodes
            self._global_step += episode_step
            
            # wait until all the metrics schema is populated
            # log stats
            elapsed_time, total_time = self.timer.reset()
            episode_frame = episode_step * self.cfg.action_repeat
            if metrics is not None:
                with self.logger.log_and_dump_ctx(self.global_frame,
                                                  ty='train') as log:
                    log('fps', episode_frame / elapsed_time)
                    log('ups', self.cfg.num_async_update_iters * self.cfg.num_async_episodes / elapsed_time)
                    log('total_time', total_time)
                    log('parent_wait_time', time.time() - wait_start_time)
                    ep_rewards = []
                    for i in range(self.num_scenes):
                        ep_rewards.append(episode_rewards[i])
                        log(f'episode_reward{i}', ep_rewards[-1])
                    log(f'episode_reward', np.average(ep_rewards))
                    log('episode_length', episode_frame)
                    log('episode', self.global_episode)
                    # log('buffer_size', len(self.replay_storage))
                    log('step', self.global_step)
    
            # try to evaluate
            if eval_every_episodes(self.global_episode):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.async_eval()
        
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
    
            # async run next episode
            self.train_runners.run_episodes(n_episodes=self.cfg.num_async_episodes)
    
            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.num_async_update_iters * self.cfg.num_async_episodes):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
        self.save_snapshot()
        self.train_runners.close()
        
    def async_eval(self):
        self.eval_runners.run_episodes(eval_i=1, n_episodes=self.cfg.num_eval_episodes)
        time_stepss, episode_rewards = self.eval_runners.finish_episodes(eval_i=1)
        time_stepss = list(zip(*time_stepss))  # switch 2D list axes to max_timestep, num_scenes
        episode = 0
        eps_len = len(time_stepss) / self.cfg.num_eval_episodes  # deal with 30 or 31 episodes length
        for i, time_steps in enumerate(time_stepss):
            self.append_to_trajectory(time_steps, eval=True)
            if (i+1) % eps_len == 0:
                self.plot_trajectory(eval_i=episode)
                episode += 1
    
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            avg_ep_rewards = []
            for i in range(self.num_scenes):
                avg_ep_rewards.append(episode_rewards[i])
                log(f'episode_reward{i}', episode_rewards[i])
            log(f'episode_reward', np.average(np.array(avg_ep_rewards)))
            log('episode', self.global_episode)
            log('step', self.global_step)

    def evaluation(self):
        self.agent.copy_model_params()
        # first collect data from evaluation_env for adaptation update
        # "before"
        step, episode, ep_returns, ep_best_scores = 0, 0, [0] * self.num_scenes, [-5] * self.num_scenes
        list_ep_returns, list_ep_best_scores = [], []
        assert self.num_scenes == 1
        eval_until_episode = drqutils.Until(self.cfg.num_evaluation_episodes)
        
        start_time = time.time()
        while eval_until_episode(episode):
            time_steps, histories = self.evaluation_env.reset(eval_i=episode)
            self.evaluation_replay_storage.add(time_steps)
            self.append_to_trajectory(time_steps, eval=True)
            while not time_steps[0].last():
                with torch.no_grad(), drqutils.eval_mode(self.agent):
                    # since we are collecting evaluation_replay_buffer for adaptaion update, use noise
                    action = [self.agent.act(time_steps[i].observation,
                                             time_steps[i].pose,
                                             step=-1,
                                             eval_mode=False,
                                             history=histories[i]) for i in range(self.num_scenes)]
                time_steps, histories = self.evaluation_env.step(action)
                self.evaluation_replay_storage.add(time_steps)
                self.append_to_trajectory(time_steps, eval=True)
                step += 1
                for i in range(self.num_scenes):
                    ep_returns[i] += time_steps[i].reward
                    ep_best_scores[i] = max(ep_best_scores[i], time_steps[i].reward)
        
            self.plot_trajectory(eval_i=episode, evaluatation_phase="before")
            episode += 1
        
            with self.logger.log_and_dump_ctx(episode, ty='evaluation_before') as log:
                # TODO for each scene, log best aesthetic score and corresponding pose
                # TODO do this for eval and evaluation
                log(f'episode_reward', ep_returns[0])
                log(f'episode_best_score', ep_best_scores[0])
                log('episode', episode)
                list_ep_returns.append(ep_returns[0])
                list_ep_best_scores.append(ep_best_scores[0])
            ep_returns = [0] * self.num_scenes
            ep_best_scores = [-5] * self.num_scenes

        print(f"before: {time.time() - start_time}")
        with self.logger.log_and_dump_ctx(episode, ty='evaluation_before') as log:
            log(f'episode_reward_avg', np.average(list_ep_returns))
            log(f'episode_best_score_avg', np.average(list_ep_best_scores))

        # adaptation: update networks once in eval env during evaluation
        # train_cs, train_new_buffer for snap_iter_nums iterations, train_meta_buffer for main_snap_iter_nums iterations
        stats_main, stats_csv = self.agent.adapt(metatrain_replay_iter=self.metatrain_replay_iter,
                                                   eval_replay_iter=self.evaluation_replay_iter,
                                                   snap_iter_nums=self.cfg.snap_iter_nums,
                                                   main_snap_iter_nums=self.cfg.main_snap_iter_nums,
                                                   main_snap_bsize_mult=self.cfg.main_snap_bsize_mult,
                                                 evaluation_buffer_size=self.cfg.evaluation_batch_size)
        print(f"adapt: {time.time() - start_time}")
        
        # evaluate on evaluation_env after adaptation
        # "after"
        step, episode, ep_returns, ep_best_scores = 0, 0, [0] * self.num_scenes, [-5] * self.num_scenes
        list_ep_returns, list_ep_best_scores = [], []
        eval_until_episode = drqutils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_steps, histories = self.evaluation_env.reset(eval_i=episode)
            self.append_to_trajectory(time_steps, eval=True)
            while not time_steps[0].last():
                with torch.no_grad(), drqutils.eval_mode(self.agent):
                    action = [self.agent.act(time_steps[i].observation,
                                             time_steps[i].pose,
                                             step=-1,
                                             eval_mode=True,
                                             history=histories[i]) for i in range(self.num_scenes)]
                time_steps, histories = self.evaluation_env.step(action)
                self.append_to_trajectory(time_steps, eval=True)
                step += 1
                for i in range(self.num_scenes):
                    ep_returns[i] += time_steps[i].reward
                    ep_best_scores[i] = max(ep_best_scores[i], time_steps[i].reward)

            self.plot_trajectory(eval_i=episode, evaluatation_phase="after")
            episode += 1
    
            with self.logger.log_and_dump_ctx(episode, ty='evaluation_after') as log:
                assert self.num_scenes == 1
                log(f'episode_reward', ep_returns[0])
                log(f'episode_best_score', ep_best_scores[0])
                log('episode', episode)
                list_ep_returns.append(ep_returns[0])
                list_ep_best_scores.append(ep_best_scores[0])
            ep_returns = [0] * self.num_scenes
            ep_best_scores = [-5] * self.num_scenes

        print(f"after: {time.time() - start_time}")
        with self.logger.log_and_dump_ctx(episode, ty='evaluation_after') as log:
            log(f'episode_reward_avg', np.average(list_ep_returns))
            log(f'episode_best_score_avg', np.average(list_ep_best_scores))

    def save_snapshot(self, fn='snapshot.pt', best=False):
        keys_to_save = ['agent', 'cfg', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        if best:
            best = self.work_dir / 'best.pt'
            with best.open('wb') as f:
                torch.save(payload, f)
        else:
            snapshot = self.work_dir / fn
            with snapshot.open('wb') as f:
                torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.snapshot_path / 'snapshot.pt'
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
    from rlcam_drqv2_mql import Workspace as W
    if cfg.ray:
        ray.init(num_gpus=8)
        ray_main(cfg)
    else:
        workspace = W(cfg)
        
        if cfg.evaluating:
            workspace.evaluation()
        elif cfg.use_multiprocessing and cfg.async_:
            workspace.async_train()
        else:
            workspace.train()

def ray_main(cfg):
    from rlcam_drqv2_mql import Workspace as W
    try:
        num_cpus = multiprocessing.cpu_count()
        storage_cpus = 2
        cfg.replay_buffer_num_workers = num_cpus - storage_cpus - cfg.num_ray_gpus
        # cfg.replay_buffer_num_workers = 0
        pose_dim = 5
        observation_spec = specs.BoundedArray(cfg.state_dim, np.uint8, 0, 255, 'observation')
        pose_spec = specs.Array((pose_dim,), np.float32, 'pose')
        action_spec = specs.BoundedArray((pose_dim,), np.float32, -1.0, 1.0, "action")
        d_pose_shape = pose_dim
        if cfg.distance_obs:
            d_pose_shape += 1
        agent = make_agent(
            observation_spec,
            pose_spec,
            action_spec,
            cfg.agent)
        agent = agent.to('cpu')
        storage_o = SharedStorage.options(num_cpus=storage_cpus)
        storage = storage_o.remote(agent, None)

        W_remote = ray.remote(W)
        workspace_o = W_remote.options(num_cpus=1, num_gpus=1)
        data_workers = [workspace_o.remote(cfg, rank, storage) for rank in range(cfg.num_ray_gpus)]
        workers = [worker.ray_train.remote() for worker in data_workers]

        update_o = update_agent.options(num_cpus=num_cpus - storage_cpus - cfg.num_ray_gpus, num_gpus=1)
        workers += [update_o.remote(cfg, storage, agent)]
        ray.wait(workers)
    except:
        traceback.print_exc()


# TODO run update in main thread, so don't have to specify CPU number. let update run on GPU 0, Data workers on 1-7
@ray.remote
def update_agent(cfg, storage, agent):
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=10322, stdoutToServer=True, stderrToServer=True)
    try:
        agent.cuda()

        work_dir = Path.cwd()
        logger = Logger(work_dir, use_tb=cfg.use_tb)
        replay_loader_n_scenes = cfg.num_ray_gpus
        replay_loader = make_replay_loader(
            work_dir / 'buffer', cfg.replay_buffer_size,
            cfg.batch_size, cfg.replay_buffer_num_workers,
            cfg.save_snapshot, cfg.nstep, cfg.discount, cfg.use_context,
            cfg.agent.context_history_length, replay_loader_n_scenes, cfg.use_position,
            cfg.diversity, cfg.smoothness, False)
        replay_iter = iter(replay_loader)
        
        last_model_index = 0
        update_step = 0
        update_begin_global_step = -1
        train_not_done = True
        global_step = ray.get(storage.get_selfplay_counter.remote())
        # wait util num seed frames
        while global_step < cfg.num_seed_frames:
            time.sleep(1)
            global_step = ray.get(storage.get_selfplay_counter.remote())
        
        # start training
        while train_not_done:
            try:
                metrics = agent.update(replay_iter, global_step)
                logger.log_metrics(metrics, global_step, ty='train')
                update_step += 1  # accurate
                if update_step % 5000 == 0:
                    print(f"update step: {update_step}")
                    
                if update_begin_global_step == -1:  # mark update beginning
                    global_step = ray.get(storage.get_selfplay_counter.remote())
                    update_begin_global_step = global_step
                    print("update begins")
                    
                else:  # update has begun
                    if update_step % cfg.ray_step_sync_interval_update == 0:  # sync global step, update and step
                        global_step = ray.get(storage.get_selfplay_counter.remote())
                        train_not_done = global_step < cfg.num_train_frames
                        storage.incr_training_counter.remote(cfg.ray_step_sync_interval_update)
    
                        if update_step * 2 - (global_step - update_begin_global_step) > cfg.max_timestep * 100:
                            print(f"update too fast, update steps: {update_step}")
                            time.sleep(1)
                            continue
                    else:
                        global_step += 2  # approximate global step

            except IndexError as e:
                if str(e) == 'Cannot choose from an empty sequence':
                    print("One of replay loader can't load any episode, sleeping 10")
                    time.sleep(10)
                    continue
                else:
                    traceback.print_exc()
            
            new_model_index = global_step // cfg.ray_model_sync_interval_update
            if new_model_index > last_model_index:
                last_model_index = new_model_index
                storage.set_weights.remote(agent.get_weights())  # upload updated weights
        print("Update Done")
    except:
        traceback.print_exc()

if __name__ == '__main__':
    main()
