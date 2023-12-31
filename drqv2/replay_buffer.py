# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate  # pytorch 1.10
# from torch.utils.data import default_collate  # pytorch 1.11


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    """ requires each item in **episode to be an numpy ndarray"""
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

# based on https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/14
# pytorch default_collate: https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
# 3 solutions to variable length in a batch: Padding, PackedSequence, Bucketing: https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/15
# Visual explanation of PackedSequqnce: https://stackoverflow.com/a/56211056
def collate_pack_history(batch):
    '''
    Pads history batch of variable lengths

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    batch = list(zip(*batch))  # reorder 2D list from (batch_size, 9) to (9, batch_size)
    ret = []
    for x in batch[-4:]:
        ret.append(torch.nn.utils.rnn.pad_sequence(x, batch_first=True).contiguous())
    
    # remove hitory_len, history from batch
    batch_no_history = list(zip(*batch[:-4]))
    return (*default_collate(batch_no_history), *ret)

class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir, num_scenes, scene_index=None):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._num_scenes = num_scenes
        self._scene_idx = scene_index
        self._current_episodes = [defaultdict(list)] * num_scenes
        self._preload()  # load existing episode files

    def __len__(self):
        return self._num_transitions

    def add(self, time_steps):
        """ save each timestep in each _current_episodes[s_idx]"""
        for s_idx, time_step in enumerate(time_steps):
            # for spec in self._data_specs:
            for i, spec in enumerate(self._data_specs):
                value = time_step[spec.name]
                # value = time_step[i]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype, f"spec: {spec.name}, " \
                                                                                f"expected: {spec.shape}, {spec.dtype} \n" \
                                                                                f"got {value.shape}, {value.dtype}"
                self._current_episodes[s_idx][spec.name].append(value)
            if time_step.last():
                episode = dict()
                # for spec in self._data_specs:
                for i, spec in enumerate(self._data_specs):
                    # value = self._current_episode[i]
                    value = self._current_episodes[s_idx][spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
                self._current_episodes[s_idx] = defaultdict(list)
                if self._scene_idx is None:
                    self._store_episode(episode, s_idx)
                else:
                    # this storage only stores for a single scene
                    self._store_episode(episode, self._scene_idx)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len, _ = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode, scene_idx):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}_{scene_idx}.npz'  # this should already support multi-scene
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(torch.utils.data.IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot, use_context, context_history_length,
                 num_scenes, use_position, diversity, smoothness, curl_torch_random_crop):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []  # list of file names
        self._episodes = dict()  # dictionary {fn: episode_dict}
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._use_context = use_context
        self._context_history_length = context_history_length
        self._num_scenes = num_scenes
        self._use_position = use_position
        self._diversity = diversity
        self._smoothness = smoothness
        self._curl_torch_random_crop = curl_torch_random_crop
        self._replay_dir.mkdir(parents=True, exist_ok=True)

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        """ Load episode file and store in memory"""
        try:
            # load episode file into a dictionary
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            # expel earlier episodes in replay buffer if we are exceeding max replay buffer size
            # expelled episode is comepletely removed from replay buffer memory and replay buffer storage
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            # early_eps_fn.unlink(missing_ok=True)  # supported in Python 3.8+
            try:
                early_eps_fn.unlink()
            except FileNotFoundError:
                pass
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            # keep the episode file if save_snapshot
            # eps_fn.unlink(missing_ok=True)  # supported in Python 3.8+
            try:
                eps_fn.unlink()
            except FileNotFoundError:
                pass
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        # with open(self._replay_dir / "buffer_debug.txt", "a") as f:
        # f.write(f"eps_fns len: {len(eps_fns)}, worker_id: {worker_id}\n")
        has_eps = False
        while not has_eps:
            fetched_size = 0
            for eps_fn in eps_fns:
                eps_idx, eps_len, scene_idx = [int(x) for x in eps_fn.stem.split('_')[1:]]
                eps_fn_id = (eps_idx * self._num_scenes + scene_idx)
                # with open(self._replay_dir / "buffer_debug.txt", "a") as f:
                #     f.write(f"fn id: {eps_fn_id}, num_workers: {self._num_workers}, remainder: {eps_fn_id % self._num_workers}, worker_id: {worker_id}\n")
                if eps_fn_id % self._num_workers != worker_id:
                    continue
                if eps_fn in self._episodes.keys():
                    break
                if fetched_size + eps_len > self._max_size:
                    break
                fetched_size += eps_len
                if not self._store_episode(eps_fn): # stop if exception in _store_episode
                    break
            has_eps = len(self._episode_fns) > 0
            if not has_eps:
                time.sleep(2)
                eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)

    def _sample(self):
        try:
            self._try_fetch()  # fetch episode from file to memory
        except:
            traceback.print_exc()
            with open(self._replay_dir / "buffer_debug.txt", "a") as f:
                f.write(traceback.format_exc())
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()  # a single episode dict
        if self._use_context:
            # avoid terminal state as next_state, since no action at terminal state, action history has one less entry
            idx = np.random.randint(0, episode_len(episode) - self._nstep) + 1
        else:
            # add +1 for the first dummy transition, i.e. reset with obs, action=0, reward=0
            idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        pose = episode['pose'][idx - 1]
        t = episode['t'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]  # n-step next state here
        next_pose = episode['pose'][idx + self._nstep - 1]
        next_t = episode['t'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        if self._diversity:
            excluding_seq = episode['excluding_seq'][idx - 1]
            next_excluding_seq = episode['excluding_seq'][idx + self._nstep - 1]
        if self._smoothness:
            avg_step_size = episode['avg_step_size'][idx - 1]
            next_avg_step_size = episode['avg_step_size'][idx + self._nstep - 1]
        for i in range(self._nstep):  # n-step return computed here
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
            
        if self._use_context:
            end = idx + self._nstep - 1
            start = np.clip(end - self._context_history_length, a_min=0, a_max=None)
            next_history_len = end - start
            if self._use_position:
                next_history_others = np.concatenate([episode['reward'][start:end], episode['action'][start+1:end+1], episode['pose'][start:end]], axis=-1)
            else:
                next_history_others = np.concatenate([episode['reward'][start:end],
                                                      episode['action'][start + 1:end + 1]], axis=-1)
            next_history_obs = episode['observation'][start:end]
            # pad from (next_history_len, 7) to (10, 7)
            if next_history_len < self._context_history_length:
                pad = ((0,self._context_history_length - next_history_len), (0,0))
                next_history_others = np.pad(next_history_others, pad)
                pad = ((0,self._context_history_length - next_history_len), (0,0), (0,0), (0,0))
                next_history_obs = np.pad(next_history_obs, pad)
            
            # +1 at the end to include action,reward,pos,obs of current step in history
            end = idx - 1
            start = np.clip(end - self._context_history_length, a_min=0, a_max=None)
            history_len = end - start
            if history_len == 0:
                # same padded shape, all zeros, and make history_len=1 and let GRU take one timestep of zeros
                history_len = 1
                history_others = np.zeros_like(next_history_others)
                history_obs = np.zeros_like(next_history_obs)
            else:
                if self._use_position:
                    history_others = np.concatenate([episode['reward'][start:end], episode['action'][start+1:end+1], episode['pose'][start:end]], axis=-1)
                else:
                    history_others = np.concatenate([episode['reward'][start:end],
                                                     episode['action'][start + 1:end + 1]], axis=-1)
                history_obs = episode['observation'][start:end]
                if history_len < self._context_history_length:
                    pad = ((0,self._context_history_length - history_len), (0,0))
                    history_others = np.pad(history_others, pad)
                    pad = ((0,self._context_history_length - history_len), (0,0), (0,0), (0,0))
                    history_obs = np.pad(history_obs, pad)
            
        ret = [obs, pose, t, action, reward, discount, next_obs, next_pose, next_t]
        if self._use_context:
            ret.extend([history_len, history_others, history_obs, next_history_len, next_history_others, next_history_obs])
        if self._diversity:
            ret.extend([excluding_seq, next_excluding_seq])
        if self._smoothness:
            ret.extend([avg_step_size, next_avg_step_size])
        if self._curl_torch_random_crop:
            ret.extend([obs.copy()])
        return tuple(ret)
        # return obs, pos, action, reward, discount, next_obs, next_pos, history_len, history_others, history_obs, next_history_len, next_history_others, next_history_obs, excluding_seq, next_excluding_seq, avg_step_size, next_avg_step_size
        # return obs, pos, action, reward, discount, next_obs, next_pos, excluding_seq, next_excluding_seq, avg_step_size, next_avg_step_size

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

# 1000000, 256, 8, 
def make_replay_loader(replay_dir, max_size, batch_size, num_workers, save_snapshot, nstep, discount,
                       use_context, context_history_length, num_scenes, use_position, diversity, smoothness,
                       curl_torch_random_crop):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot,
                            use_context=use_context,
                            context_history_length=context_history_length,
                            num_scenes=num_scenes,
                            use_position=use_position,
                            diversity=diversity,
                            smoothness=smoothness,
                            curl_torch_random_crop=curl_torch_random_crop)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
                                         # collate_fn=collate_pack_history)
    return loader
