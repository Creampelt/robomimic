"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import re
import sys
import time
import datetime
import threading
import contextlib
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils

from robomimic.utils.dataset import SequenceDataset, MetaDataset
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy


def get_exp_dir(config, auto_remove_exp_dir=False, resume=False):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.

    Args:
        auto_remove_exp_dir (bool): if True, automatically remove the existing experiment
            folder if it exists at the same path.
        resume (bool): if True, resume an existing training run instead of creating a 
            new experiment directory
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y-%m-%d_%H-%M-%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expanduser(config.train.output_dir)
    if not os.path.isabs(base_output_dir):
        # relative paths are specified relative to robomimic module location
        base_output_dir = os.path.join(robomimic.__path__[0], base_output_dir)
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)
    if resume:
        assert os.path.exists(base_output_dir), "Resuming training run, but output dir {} does not exist".format(base_output_dir)
        subdir_lst = os.listdir(base_output_dir)
        time_str = sorted(subdir_lst)[-1]  # get the most recent subdirectory
        assert os.path.isdir(os.path.join(base_output_dir, time_str)), "Found item {} that is not a subdirectory in {}".format(time_str, base_output_dir)
    # Each run gets its own timestamped subdirectory under base_output_dir, so
    # it's safe (and desirable) to keep previous runs around for comparison.

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir, exist_ok=resume)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir, exist_ok=resume)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir, exist_ok=resume)

    time_dir = os.path.join(base_output_dir, time_str)
    
    return log_dir, output_dir, video_dir, time_dir


def load_data_for_training(config, obs_keys):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.train.hdf5_filter_key
    valid_filter_by_attribute = config.train.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.experiment.validate, "specified validation filter key {}, but config.experiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.experiment.validate:
        # assert not config.train.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.train.hdf5_filter_key and config.train.hdf5_validation_filter_key"
        assert isinstance(config.train.data, list), "config.train.data should be a list of datasets, not a single dataset"
        for dataset_cfg in config.train.data:
            train_demo_keys = FileUtils.get_demos_for_filter_key(
                hdf5_path=os.path.expanduser(dataset_cfg["path"]),
                filter_key=train_filter_by_attribute,
            )
            valid_demo_keys = FileUtils.get_demos_for_filter_key(
                hdf5_path=os.path.expanduser(dataset_cfg["path"]),
                filter_key=valid_filter_by_attribute,
            )
            assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
                "validation demonstrations!"
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = dataset_factory(config, obs_keys, filter_by_attribute=valid_filter_by_attribute)
    else:
        train_dataset = dataset_factory(config, obs_keys, filter_by_attribute=train_filter_by_attribute)
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.train.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.train.data

    # NOTE: currently supporting fixed language embedding per dataset
    ## that is fetched from dataset config and not from file
    if LangUtils.LANG_EMB_OBS_KEY in obs_keys:
        obs_keys.remove(LangUtils.LANG_EMB_OBS_KEY)
        ds_langs = [ds_cfg.get("lang", "dummy") for ds_cfg in config.train.data]
    else:
        ds_langs = [None for _ in config.train.data]

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.train.action_keys,
        dataset_keys=config.train.dataset_keys,
        action_config=config.train.action_config,
        load_next_obs=config.train.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
    )

    ds_kwargs["hdf5_path"] = [ds_cfg["path"] for ds_cfg in config.train.data]
    ds_kwargs["filter_by_attribute"] = [ds_cfg.get("filter_key", filter_by_attribute) for ds_cfg in config.train.data]
    ds_kwargs["demo_limit"] = [ds_cfg.get("demo_limit", None) for ds_cfg in config.train.data]
    ds_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in config.train.data]

    meta_ds_kwargs = dict()

    dataset = get_dataset(
        ds_class=SequenceDataset,
        ds_kwargs=ds_kwargs,
        ds_weights=ds_weights,
        ds_langs=ds_langs,
        normalize_weights_by_ds_size=config.train.normalize_weights_by_ds_size,
        meta_ds_class=MetaDataset,
        meta_ds_kwargs=meta_ds_kwargs,
    )

    return dataset


def get_dataset(
    ds_class,
    ds_kwargs,
    ds_weights,
    ds_langs,
    normalize_weights_by_ds_size,
    meta_ds_class=MetaDataset,
    meta_ds_kwargs=None,
):
    """
    Create a dataset object from the provided class and parameters.

    Args:
        ds_class (class): class of the dataset to create (e.g. SequenceDataset)
        ds_kwargs (dict): keyword arguments to pass to the dataset class constructor
        ds_weights (list): list of weights for each dataset instance, used in MetaDataset
        ds_langs (list): list of language embeddings for each dataset instance
        normalize_weights_by_ds_size (bool): if True, normalize dataset weights by the size of each dataset
        meta_ds_class (class): class of the meta dataset to create (e.g. MetaDataset)
        meta_ds_kwargs (dict): keyword arguments to pass to the meta dataset class constructor
    
    Returns:
        ds (SequenceDataset or MetaDataset instance): dataset object created from the provided class and parameters
    """
    ds_list = []
    for i in range(len(ds_weights)):
        
        ds_kwargs_copy = deepcopy(ds_kwargs)

        keys = ["hdf5_path", "filter_by_attribute", "demo_limit"]

        for k in keys:
            ds_kwargs_copy[k] = ds_kwargs[k][i]

        ds_kwargs_copy["lang"] = ds_langs[i]
        
        ds_list.append(ds_class(**ds_kwargs_copy))
    
    if len(ds_weights) == 1:
        ds = ds_list[0]
    else:
        if meta_ds_kwargs is None:
            meta_ds_kwargs = dict()
        ds = meta_ds_class(
            datasets=ds_list,
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=normalize_weights_by_ds_size,
            **meta_ds_kwargs
        )

    return ds


def batchify_obs(obs_list):
    """
    Converts a list of observation dictionaries into a single dictionary.
    """
    keys = list(obs_list[0].keys())
    obs = {
        k: np.stack([obs_list[i][k] for i in range(len(obs_list))]) for k in keys
    }
    
    return obs


def run_rollout(
        policy, 
        env, 
        horizon,
        use_goals=False,
        render=False,
        video_writer=None,
        video_skip=5,
        terminate_on_success=False,
    ):
    """
    Runs a rollout in an environment with the current network parameters.

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        env (EnvBase instance): environment to use for rollouts.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        render (bool): if True, render the rollout to the screen

        video_writer (imageio Writer instance): if not None, use video writer object to append frames at 
            rate given by @video_skip

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

    Returns:
        results (dict): dictionary containing return, success rate, etc.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        # retrieve goal from the environment
        goal_dict = env.get_goal()

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = None # success metrics

    end_step = None

    video_frames = []
    
    try:
        for step_i in range(horizon):
            # get action from policy
            policy_ob = ob_dict
            ac = policy(ob=policy_ob, goal=goal_dict)

            # play action
            ob_dict, r, done, _ = env.step(ac)

            # render to screen
            if render:
                env.render(mode="human")

            # compute reward
            rews.append(r)

            cur_success_metrics = env.is_success()

            if success is None:
                success = deepcopy(cur_success_metrics)
            else:
                for k in success:
                    success[k] = success[k] | cur_success_metrics[k]

            # visualization
            if video_writer is not None:
                if video_count % video_skip == 0:
                    frame = env.render(mode="rgb_array", height=512, width=512)
                    video_frames.append(frame)

                video_count += 1

            # break if done
            if done or (terminate_on_success and success["task"]):
                end_step = step_i
                break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))


    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    end_step = end_step or step_i
    total_reward = np.sum(rews[:end_step + 1])
    
    results["Return"] = total_reward
    results["Horizon"] = end_step + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            results["{}_Success_Rate".format(k)] = float(success[k])

    return results


# Counters incremented by the physics-explosion guard in run_warp_rollout.
# Useful for stability benchmarking — reset to 0 before a rollout and read after.
WARP_EXPLOSION_STEPS = 0   # number of rollout steps that tripped the NaN/inf guard
WARP_EXPLOSION_ENVS = 0    # cumulative (env, step) pairs with NaN/inf observations


# mujoco-warp kernel-side warnings we don't want flooding stdout during rollouts.
# These come from `wp.printf` inside Warp CUDA kernels, so Python-level stdout
# redirection doesn't catch them — filtering happens at the OS file-descriptor
# level in `_suppress_warp_kernel_warnings`.
_WARP_KERNEL_WARNING_PATTERNS = (
    re.compile(r"Warning: opt\.ccd_iterations, currently set to \d+, needs to be increased\."),
    re.compile(r"broadphase overflow - please increase nconmax"),
    re.compile(r"narrowphase overflow - please increase nconmax"),
    re.compile(r"nefc overflow - please increase njmax"),
    re.compile(r"CCD overflow - please increase naccdmax"),
    re.compile(r"contact match overflow: please increase"),
)


@contextlib.contextmanager
def _suppress_warp_kernel_warnings():
    """
    Redirect stdout at the FD level and drop lines matching known mujoco-warp
    kernel-printf warnings. Other stdout content is passed through untouched.
    Falls through as a no-op if the redirect can't be set up (non-POSIX, etc.).
    """
    try:
        saved_fd = os.dup(1)                     # backup of original stdout
        pump_fd = os.dup(saved_fd)               # independent FD for the pump thread
        r_fd, w_fd = os.pipe()
    except (OSError, AttributeError):
        yield
        return

    def _pump():
        # tqdm uses '\r' to redraw progress bars, so we split on both '\r' and
        # '\n' and filter each chunk independently; otherwise a progress bar
        # chunk can carry a trailing kernel warning past the filter.
        with os.fdopen(r_fd, "rb", buffering=0) as reader, \
                os.fdopen(pump_fd, "wb", buffering=0) as writer:
            buf = b""
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(b"([\r\n])", buf)
                buf = parts.pop() if parts and parts[-1] not in (b"\r", b"\n") else b""
                out = bytearray()
                segment = b""
                for part in parts:
                    if part in (b"\r", b"\n"):
                        text = segment.decode("utf-8", errors="replace")
                        if not any(p.search(text) for p in _WARP_KERNEL_WARNING_PATTERNS):
                            out += segment + part
                        segment = b""
                    else:
                        segment = part
                if out:
                    writer.write(bytes(out))
            # flush any residual trailing text without a terminator
            if buf:
                text = buf.decode("utf-8", errors="replace")
                if not any(p.search(text) for p in _WARP_KERNEL_WARNING_PATTERNS):
                    writer.write(buf)

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    try:
        sys.stdout.flush()
        os.dup2(w_fd, 1)                         # fd 1 now points at the pipe
        os.close(w_fd)
        yield
    finally:
        try:
            sys.stdout.flush()
        except (OSError, ValueError):
            pass
        # Restore original stdout. Closing the last write-side of the pipe
        # (the one at fd 1) sends EOF to the pump's reader so the thread exits.
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        thread.join(timeout=2.0)


def run_warp_rollout(
        policy,
        env,
        horizon,
        use_goals=False,
        terminate_on_success=False,
        video_writer=None,
        video_skip=5,
    ):
    """
    Runs a single vectorized rollout across all parallel MuJoCo Warp environments in @env.
    Instead of running @num_episodes sequential rollouts, one batched rollout of @horizon
    steps is executed, with observations and actions handled as batched CUDA tensors.

    Args:
        policy (RolloutPolicy instance): policy with use_warp=True.

        env (EnvBase instance): warp-enabled environment with env.env.num_envs parallel worlds.

        horizon (int): maximum number of steps per rollout.

        use_goals (bool): if True, retrieve goal observations from env.

        terminate_on_success (bool): if True, end the rollout early once all envs succeed.

        video_writer (imageio Writer): if not None, record frames from env 0 at this writer.

        video_skip (int): write a frame every this many steps.

    Returns:
        results (list[dict]): one result dict per parallel environment, each containing
            Return, Horizon, Success_Rate, and any additional success metrics.
    """
    assert isinstance(policy, RolloutPolicy)
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)

    policy.start_episode()

    ob_dict = env.reset()
    goal_dict = None
    if use_goals:
        goal_dict = env.get_goal()

    num_envs = env.env.num_envs  # type: ignore[union-attr]

    rews = []
    success = None
    end_step = None
    step_i = 0
    video_count = 0
    video_frames = []

    def _scalar(v, i):
        """Extract a Python float for env index i from a batched value."""
        try:
            return float(v[i])
        except (TypeError, IndexError):
            return float(v)

    # Track first explosion step for the post-rollout summary.
    first_explosion_step = None
    global WARP_EXPLOSION_STEPS, WARP_EXPLOSION_ENVS
    start_steps = WARP_EXPLOSION_STEPS
    start_envs = WARP_EXPLOSION_ENVS

    try:
        with _suppress_warp_kernel_warnings():
            for step_i in LogUtils.custom_tqdm(range(horizon), desc="warp rollout", total=horizon):
                # Physics-explosion guard: detect NaN/inf in any observation with a
                # single GPU→CPU sync (cat all obs tensors, one .all() check).  Only do
                # the per-key slow path when something is actually broken.
                # Filter to tensors that share the same leading (batch) dimension to
                # exclude RNN hidden states or other non-obs tensors with different shapes.
                _obs_raw = [v for v in ob_dict.values() if isinstance(v, torch.Tensor) and v.dim() >= 1]
                if _obs_raw:
                    _B = _obs_raw[0].shape[0]
                    _obs_tensors = [v.reshape(_B, -1) for v in _obs_raw if v.shape[0] == _B]
                else:
                    _obs_tensors = []
                if _obs_tensors and not torch.cat(_obs_tensors, dim=1).isfinite().all():
                    WARP_EXPLOSION_STEPS += 1
                    if first_explosion_step is None:
                        first_explosion_step = step_i
                    for k, v in ob_dict.items():
                        if isinstance(v, torch.Tensor) and not v.isfinite().all():
                            bad = (~v.isfinite().all(dim=-1) if v.dim() > 1 else ~v.isfinite()).nonzero(as_tuple=False).squeeze(-1)
                            WARP_EXPLOSION_ENVS += bad.numel()
                            ob_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                ac = policy(ob=ob_dict, goal=goal_dict, batched_ob=True)
                ob_dict, r, done, _ = env.step(ac)
                rews.append(r)

                cur_success_metrics = env.is_success()
                if success is None:
                    success = deepcopy(cur_success_metrics)
                else:
                    for k in success:
                        success[k] = success[k] | cur_success_metrics[k]

                # Capture a frame from env 0 for the video.
                if video_writer is not None and video_count % video_skip == 0:
                    frame = env.render(mode="rgb_array", height=512, width=512)
                    video_frames.append(frame)
                video_count += 1

                task_success = success["task"]
                all_done = task_success.all() if isinstance(task_success, torch.Tensor) else np.all(task_success)
                if terminate_on_success and all_done:
                    end_step = step_i
                    break

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    # One-line summary instead of per-step spam.
    delta_steps = WARP_EXPLOSION_STEPS - start_steps
    delta_envs = WARP_EXPLOSION_ENVS - start_envs
    if delta_steps > 0:
        LogUtils.log_warning(
            f"warp rollout: physics explosions detected on {delta_steps} steps "
            f"(first at step {first_explosion_step}, {delta_envs} env-steps zeroed)"
        )

    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    if end_step is None:
        end_step = step_i

    results_list = []
    for i in range(num_envs):
        total_reward = sum(_scalar(r, i) for r in rews[:end_step + 1])
        result = dict(
            Return=total_reward,
            Horizon=end_step + 1,
            Success_Rate=_scalar(success["task"], i),
        )
        for k in success:
            if k != "task":
                result["{}_Success_Rate".format(k)] = _scalar(success[k], i)
        results_list.append(result)

    return results_list


def rollout_with_stats(
        policy,
        envs,
        horizon,
        use_goals=False,
        num_episodes=None,
        render=False,
        video_dir=None,
        video_path=None,
        epoch=None,
        video_skip=5,
        terminate_on_success=False,
        verbose=False,
    ):
    """
    A helper function used in the train loop to conduct evaluation rollouts per environment
    and summarize the results.

    Can specify @video_dir (to dump a video per environment) or @video_path (to dump a single video
    for all environments).

    Args:
        policy (RolloutPolicy instance): policy to use for rollouts.

        envs (dict): dictionary that maps env_name (str) to EnvBase instance. The policy will
            be rolled out in each env.

        horizon (int): maximum number of steps to roll the agent out for

        use_goals (bool): if True, agent is goal-conditioned, so provide goal observations from env

        num_episodes (int): number of rollout episodes per environment

        render (bool): if True, render the rollout to the screen

        video_dir (str): if not None, dump rollout videos to this directory (one per environment)

        video_path (str): if not None, dump a single rollout video for all environments

        epoch (int): epoch number (used for video naming)

        video_skip (int): how often to write video frame

        terminate_on_success (bool): if True, terminate episode early as soon as a success is encountered

        verbose (bool): if True, print results of each rollout
    
    Returns:
        all_rollout_logs (dict): dictionary of rollout statistics (e.g. return, success rate, ...) 
            averaged across all rollouts 

        video_paths (dict): path to rollout videos for each environment
    """
    assert isinstance(policy, RolloutPolicy)

    all_rollout_logs = OrderedDict()

    # handle paths and create writers for video writing
    assert (video_path is None) or (video_dir is None), "rollout_with_stats: can't specify both video path and dir"
    write_video = (video_path is not None) or (video_dir is not None)
    video_paths = OrderedDict()
    video_writers = OrderedDict()
    if video_path is not None:
        # a single video is written for all envs
        video_paths = { k : video_path for k in envs }
        video_writer = imageio.get_writer(video_path, fps=20)
        video_writers = { k : video_writer for k in envs }
    if video_dir is not None:
        # video is written per env
        video_str = "_epoch_{}.mp4".format(epoch) if epoch is not None else ".mp4" 
        video_paths = { k : os.path.join(video_dir, "{}{}".format(k, video_str)) for k in envs }
        video_writers = { k : imageio.get_writer(video_paths[k], fps=20) for k in envs }

    for env_key, env in envs.items():
        env_video_writer = None
        if write_video:
            print("video writes to " + video_paths[env_key])
            env_video_writer = video_writers[env_key]

        env_name = env.name

        print("rollout: env={}, horizon={}, use_goals={}, num_episodes={}".format(
            env_name, horizon, use_goals, num_episodes,
        ))

        if getattr(policy, "use_warp", False):
            # Single parallelized rollout: one step sequence across num_envs simultaneous worlds.
            # Video is captured from env 0 only.
            rollout_timestamp = time.time()
            rollout_logs = run_warp_rollout(
                policy=policy,
                env=env,
                horizon=horizon,
                use_goals=use_goals,
                terminate_on_success=terminate_on_success,
                video_writer=env_video_writer,
                video_skip=video_skip,
            )
            elapsed = time.time() - rollout_timestamp
            per_env_time = elapsed / max(len(rollout_logs), 1)
            for log in rollout_logs:
                log["time"] = per_env_time
        else:
            rollout_logs = []
            iterator = range(num_episodes)
            if not verbose:
                iterator = LogUtils.custom_tqdm(iterator, total=num_episodes)

            num_success = 0
            for ep_i in iterator:
                rollout_timestamp = time.time()
                rollout_info = run_rollout(
                    policy=policy,
                    env=env,
                    horizon=horizon,
                    render=render,
                    use_goals=use_goals,
                    video_writer=env_video_writer,
                    video_skip=video_skip,
                    terminate_on_success=terminate_on_success,
                )
                rollout_info["time"] = time.time() - rollout_timestamp

                rollout_logs.append(rollout_info)
                num_success += rollout_info["Success_Rate"]

                if verbose:
                    print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
                    print(json.dumps(rollout_info, sort_keys=True, indent=4))

        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            env_video_writer.close()

        # average metric across all episodes
        rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
        rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
        rollout_logs_mean["Time_Episode"] = float(np.sum(rollout_logs["time"]))  # total rollout wall time (s)
        all_rollout_logs[env_key] = rollout_logs_mean

    if video_path is not None:
        # close video writer that was used for all envs
        video_writer.close()
    elif video_dir is not None:
        # close per-env writers
        for vw in video_writers.values():
            vw.close()

    return all_rollout_logs, video_paths


def should_save_from_rollout_logs(
        all_rollout_logs,
        best_return,
        best_success_rate,
        epoch_ckpt_name,
        save_on_best_rollout_return,
        save_on_best_rollout_success_rate,
    ):
    """
    Helper function used during training to determine whether checkpoints and videos
    should be saved. It will modify input attributes appropriately (such as updating
    the best returns and success rates seen and modifying the epoch ckpt name), and
    returns a dict with the updated statistics.

    Args:
        all_rollout_logs (dict): dictionary of rollout results that should be consistent
            with the output of @rollout_with_stats

        best_return (dict): dictionary that stores the best average rollout return seen so far
            during training, for each environment

        best_success_rate (dict): dictionary that stores the best average success rate seen so far
            during training, for each environment

        epoch_ckpt_name (str): what to name the checkpoint file - this name might be modified
            by this function

        save_on_best_rollout_return (bool): if True, should save checkpoints that achieve a 
            new best rollout return

        save_on_best_rollout_success_rate (bool): if True, should save checkpoints that achieve a 
            new best rollout success rate

    Returns:
        save_info (dict): dictionary that contains updated input attributes @best_return,
            @best_success_rate, @epoch_ckpt_name, along with two additional attributes
            @should_save_ckpt (True if should save this checkpoint), and @ckpt_reason
            (string that contains the reason for saving the checkpoint)
    """
    should_save_ckpt = False
    ckpt_reason = None
    for env_name in all_rollout_logs:
        rollout_logs = all_rollout_logs[env_name]

        if rollout_logs["Return"] > best_return[env_name]:
            best_return[env_name] = rollout_logs["Return"]
            if save_on_best_rollout_return:
                # save checkpoint if achieve new best return
                should_save_ckpt = True
                ckpt_reason = "return"

        if rollout_logs["Success_Rate"] > best_success_rate[env_name]:
            best_success_rate[env_name] = rollout_logs["Success_Rate"]
            if save_on_best_rollout_success_rate:
                # save checkpoint if achieve new best success rate
                should_save_ckpt = True
                ckpt_reason = "success"

    # return the modified input attributes
    return dict(
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        should_save_ckpt=should_save_ckpt,
        ckpt_reason=ckpt_reason,
    )


def save_model(model, config, env_meta, shape_meta, ckpt_path, variable_state=None, obs_normalization_stats=None, action_normalization_stats=None):
    """
    Save model to a torch pth file.

    Args:
        model (Algo instance): model to save

        config (BaseConfig instance): config to save

        env_meta (dict): env metadata for this training run

        shape_meta (dict): shape metdata for this training run

        ckpt_path (str): writes model checkpoint to this path

        variable_state (dict): internal variable state in main train loop, used for restoring training process
            from ckpt

        obs_normalization_stats (dict): optionally pass a dictionary for observation
            normalization. This should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

        action_normalization_stats (dict): optionally pass a dictionary for action
            normalization. This should map action keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the action.
    """
    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model=model.serialize(),
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
        variable_state=variable_state,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        obs_normalization_stats = deepcopy(obs_normalization_stats)
        params["obs_normalization_stats"] = TensorUtils.to_list(obs_normalization_stats)
    if action_normalization_stats is not None:
        action_normalization_stats = deepcopy(action_normalization_stats)
        params["action_normalization_stats"] = TensorUtils.to_list(action_normalization_stats)
    torch.save(params, ckpt_path)
    print("save checkpoint to {}".format(ckpt_path))


def run_epoch(model, data_loader, epoch, validate=False, num_steps=None, obs_normalization_stats=None):
    """
    Run an epoch of training or validation.

    Args:
        model (Algo instance): model to train

        data_loader (DataLoader instance): data loader that will be used to serve batches of data
            to the model

        epoch (int): epoch number

        validate (bool): whether this is a training epoch or validation epoch. This tells the model
            whether to do gradient steps or purely do forward passes.

        num_steps (int): if provided, this epoch lasts for a fixed number of batches (gradient steps),
            otherwise the epoch is a complete pass through the training dataset

        obs_normalization_stats (dict or None): if provided, this should map observation keys to dicts
            with a "mean" and "std" of shape (1, ...) where ... is the default
            shape for the observation.

    Returns:
        step_log_all (dict): dictionary of logged training metrics averaged across all batches
    """
    epoch_timestamp = time.time()
    if validate:
        model.set_eval()
    else:
        model.set_train()
    if num_steps is None:
        num_steps = len(data_loader)

    step_log_all = []
    timing_stats = dict(Data_Loading=[], Process_Batch=[], Train_Batch=[], Log_Info=[])
    start_time = time.time()

    data_loader_iter = iter(data_loader)
    for _ in LogUtils.custom_tqdm(range(num_steps)):

        # load next batch from data loader
        try:
            t = time.time()
            batch = next(data_loader_iter)
        except StopIteration:
            # reset for next dataset pass
            data_loader_iter = iter(data_loader)
            t = time.time()
            batch = next(data_loader_iter)
        timing_stats["Data_Loading"].append(time.time() - t)

        # process batch for training
        t = time.time()
        input_batch = model.process_batch_for_training(batch)
        input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=obs_normalization_stats)
        timing_stats["Process_Batch"].append(time.time() - t)

        # forward and backward pass
        t = time.time()
        info = model.train_on_batch(input_batch, epoch, validate=validate)
        timing_stats["Train_Batch"].append(time.time() - t)
        model.on_gradient_step()

        # tensorboard logging
        t = time.time()
        step_log = model.log_info(info)
        step_log_all.append(step_log)
        timing_stats["Log_Info"].append(time.time() - t)

    # flatten and take the mean of the metrics
    step_log_dict = {}
    for i in range(len(step_log_all)):
        for k in step_log_all[i]:
            if k not in step_log_dict:
                step_log_dict[k] = []
            step_log_dict[k].append(step_log_all[i][k])
    step_log_all = dict((k, float(np.mean(v))) for k, v in step_log_dict.items())

    # add in timing stats (seconds)
    for k in timing_stats:
        step_log_all["Time_{}".format(k)] = float(np.sum(timing_stats[k]))
    step_log_all["Time_Epoch"] = time.time() - epoch_timestamp

    return step_log_all


def is_every_n_steps(interval, current_step, skip_zero=False):
    """
    Convenient function to check whether current_step is at the interval. 
    Returns True if current_step % interval == 0 and asserts a few corner cases (e.g., interval <= 0)
    
    Args:
        interval (int): target interval
        current_step (int): current step
        skip_zero (bool): whether to skip 0 (return False at 0)

    Returns:
        is_at_interval (bool): whether current_step is at the interval
    """
    if interval is None:
        return False
    assert isinstance(interval, int) and interval > 0
    assert isinstance(current_step, int) and current_step >= 0
    if skip_zero and current_step == 0:
        return False
    return current_step % interval == 0
