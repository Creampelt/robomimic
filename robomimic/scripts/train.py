"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict

import numpy as np
import psutil
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
import torch
from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
from robomimic.utils.async_rollout import AsyncRolloutManager
from robomimic.utils.log_utils import DataLogger, PrintLogger, flush_warnings, log_warning
from torch.utils.data import DataLoader


def _perf_key(k, stage):
    """Map a step_log ``Time_{X}`` key to ``Perf/{name}_time`` (or ``Perf/valid_{name}_time``).

    Train stage keys are flat under Perf/; valid stage keys are prefixed with
    ``valid_`` to keep them distinct. Rollout timing is logged as
    ``Perf/rollout_time`` from a dedicated code path, not via this helper.
    """
    base = k[len("Time_"):].lower()
    if base == "train_batch":
        base = "batch"   # avoid redundant "train_batch" under the Train stage
    if stage == "valid":
        return "Perf/valid_{}_time".format(base)
    return "Perf/{}_time".format(base)


def _metric_key(k, stage):
    """Normalize a step_log metric key to ``{stage}/{name}``."""
    if k.startswith("Optimizer/") and k.endswith("_lr"):
        tail = k[len("Optimizer/"):-len("_lr")]
        if tail in ("", "policy0"):
            return "{}/learning_rate".format(stage)
        return "{}/learning_rate/{}".format(stage, tail.lower())
    renames = {
        "policy_grad_norms": "policy_grad_norm",
    }
    name = renames.get(k.lower(), k.lower())
    return "{}/{}".format(stage, name)


def _consume_async_rollout_result(
    *,
    result,
    data_logger,
    config,
    ckpt_dir: str,
    env_meta_list,
    shape_meta_list,
    best_valid_loss,
    best_return,
    best_success_rate,
    obs_normalization_stats,
    action_normalization_stats,
):
    """Log one drained rollout and, if it's a new best, save the exact
    snapshot weights — not the current live model, which may have moved
    on since the rollout was submitted. ``best_return`` / ``best_success_rate``
    are dicts mutated in place by the underlying tracker.
    """
    if result.error is not None:
        log_warning(
            "async rollout for epoch {} failed: {}".format(result.epoch, result.error)
        )
        return

    # Use the rollout's epoch in the filename so the saved checkpoint is
    # named after the weights it contains, not the training epoch we
    # happen to be on when the result lands.
    per_result_ckpt_name = "model_epoch_{}".format(result.epoch)

    updated_stats = _log_rollout_result(
        data_logger=data_logger,
        config=config,
        epoch_for_log=result.epoch,
        all_rollout_logs=result.all_rollout_logs,
        video_paths=result.video_paths,
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=per_result_ckpt_name,
    )

    save_rollout_best = (
        config.experiment.save.enabled
        and updated_stats["should_save_ckpt"]
        and updated_stats["ckpt_reason"] in ("return", "success")
        and result.nets_state_dict is not None
    )
    if not save_rollout_best:
        return

    snapshot_ckpt_path = os.path.join(ckpt_dir, updated_stats["epoch_ckpt_name"] + ".pth")
    variable_state_snapshot = dict(
        epoch=result.epoch,
        best_valid_loss=best_valid_loss,
        best_return=best_return,
        best_success_rate=best_success_rate,
    )
    _save_model_from_state_dict(
        nets_state_dict=result.nets_state_dict,
        config=config,
        env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
        shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
        ckpt_path=snapshot_ckpt_path,
        variable_state=variable_state_snapshot,
        obs_normalization_stats=obs_normalization_stats,
        action_normalization_stats=action_normalization_stats,
    )
    if config.experiment.logging.log_wandb:
        data_logger.log_checkpoint(snapshot_ckpt_path)


def _save_model_from_state_dict(
    *,
    nets_state_dict: dict,
    config,
    env_meta,
    shape_meta,
    ckpt_path: str,
    variable_state: dict,
    obs_normalization_stats=None,
    action_normalization_stats=None,
):
    """Write an Algo-compatible checkpoint from a raw ``nets`` state_dict.

    Used by the async rollout path to save the exact weights that scored
    a new best, even after the main model has continued training.
    Optimizer and lr_scheduler state are left empty — these checkpoints
    are for publishing eval weights, not resuming training.
    """
    from copy import deepcopy

    import robomimic.utils.tensor_utils as TensorUtils

    env_meta = deepcopy(env_meta)
    shape_meta = deepcopy(shape_meta)
    params = dict(
        model={"nets": nets_state_dict, "optimizers": {}, "lr_schedulers": {}},
        config=config.dump(),
        algo_name=config.algo_name,
        env_metadata=env_meta,
        shape_metadata=shape_meta,
        variable_state=variable_state,
    )
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        params["obs_normalization_stats"] = TensorUtils.to_list(deepcopy(obs_normalization_stats))
    if action_normalization_stats is not None:
        params["action_normalization_stats"] = TensorUtils.to_list(deepcopy(action_normalization_stats))
    torch.save(params, ckpt_path)
    print("save (snapshot) checkpoint to {}".format(ckpt_path))


def _log_rollout_result(
    *,
    data_logger,
    config,
    epoch_for_log: int,
    all_rollout_logs: dict,
    video_paths: dict | None,
    best_return,
    best_success_rate,
    epoch_ckpt_name: str,
):
    """Consume one rollout result (sync or async) and log it to wandb/tb.

    Mirrors the per-epoch logging block that used to live inline in the
    train loop. Returns ``TrainUtils.should_save_from_rollout_logs``'s
    updated stats so the caller can decide whether to checkpoint.

    ``epoch_for_log`` is the epoch the *rollout* was for (may lag the
    current training epoch when running async). Metrics and video are
    recorded against this value — with ``use_local_step`` enabled, wandb
    plots them at the correct x position even when the training side
    has already logged later epochs.
    """
    rollout_key_map = {
        "Return": "Rollout/mean_reward",
        "Success_Rate": "Rollout/success_rate",
        "time": "Rollout/mean_time",
    }
    for env_name, rollout_logs in all_rollout_logs.items():
        for k, v in rollout_logs.items():
            if k == "Horizon":
                continue
            if k in rollout_key_map:
                data_logger.record(rollout_key_map[k], v, epoch_for_log)
            elif k.endswith("_Success_Rate"):
                data_logger.record("Rollout/{}".format(k.lower()), v, epoch_for_log)
            elif k == "Time_Episode":
                data_logger.record("Perf/rollout_time", v, epoch_for_log)

        print("\nEpoch {} Rollouts took {}s (avg) with results:".format(
            epoch_for_log, rollout_logs["time"]))
        print("Env: {}".format(env_name))
        print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    if (
        video_paths is not None
        and config.experiment.logging.log_wandb
        and config.experiment.logging.get("log_rollout_videos", True)
    ):
        for _, video_path in video_paths.items():
            data_logger.log_video("video", video_path, epoch_for_log)

    return TrainUtils.should_save_from_rollout_logs(
        all_rollout_logs=all_rollout_logs,
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
        save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
    )


def _fetch_wandb_checkpoint(run_ref, model_name, download_dir):
    """Download ``model_name`` from the wandb run at ``run_ref`` (entity/project/id).

    Returns the local path of the downloaded file.
    """
    import wandb
    api = wandb.Api()
    run = api.run(run_ref)
    os.makedirs(download_dir, exist_ok=True)
    file_obj = run.file(model_name)
    downloaded = file_obj.download(root=download_dir, replace=True)
    # wandb returns an open file handle whose .name is the local path
    return downloaded.name


def train(config, device, resume=False, wandb_run=None, wandb_model=None):
    """
    Train a model using the algorithm.

    Args:
        config: robomimic Config
        device: torch device
        resume: if True, reload the latest checkpoint from the existing exp dir
        wandb_run: optional ``entity/project/run_id`` to fetch a checkpoint from
        wandb_model: optional checkpoint filename within that wandb run (e.g. ``model_50.pth``);
            required when ``wandb_run`` is set. Starts training in a fresh timestamp dir
            but loads weights + optimizer state + epoch counter from the downloaded file.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, time_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=True, resume=resume)

    # path for latest model and backup (to support @resume functionality)
    latest_model_path = os.path.join(time_dir, "last.pth")
    latest_model_backup_path = os.path.join(time_dir, "last_bak.pth")

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    if isinstance(config.train.data, str):
        # if only a single dataset is provided, convert to list
        with config.values_unlocked():
            config.train.data = [{"path": config.train.data}]
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

        # populate language instruction for env in env_meta
        env_meta["lang"] = dataset_cfg.get("lang", "dummy")

        # update env meta if applicable
        from robomimic.utils.python_utils import deep_update

        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config=dataset_cfg,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            verbose=True,
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        # if an environment name is specified, just use this env using the first dataset's metadata
        # and ignore envs from all datasets
        env_meta = env_meta_list[0].copy()
        env_meta["env_name"] = config.experiment.env
        env_meta_list = [env_meta]
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    _use_warp = config.experiment.rollout.use_warp
    _num_envs = config.experiment.rollout.n if _use_warp else 1
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        for env_i in range(len(env_meta_list)):
            # check if this env should be evaluated
            dataset_cfg = config.train.data[env_i]
            do_eval = dataset_cfg.get("eval", True)
            if not do_eval:
                continue

            env_meta = env_meta_list[env_i]
            shape_meta = shape_meta_list[env_i]

            env_names = [env_meta["env_name"]]
            if (env_i == 0) and (config.experiment.additional_envs is not None):
                # if additional environments are specified, add them to the list
                # all additional environments use env_meta from the first dataset
                for name in config.experiment.additional_envs:
                    env_names.append(name)

            # create environment for each env_name
            def create_env(env_name):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"] or shape_meta["use_depths"],
                    use_warp=_use_warp,
                    num_envs=_num_envs,
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable
                return env

            for env_name in env_names:
                env = create_env(env_name)
                env_key = (
                    os.path.splitext(os.path.basename(dataset_cfg["path"]))[0]
                    if not dataset_cfg.get("key", None)
                    else dataset_cfg["key"]
                )
                envs[env_key] = env
                print(env)

    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True,
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # add info to optim_params
    with config.values_unlocked():
        if "optim_params" in config.algo:
            # add info to optim_params of each net
            for k in config.algo.optim_params:
                config.algo.optim_params[k]["num_train_batches"] = (
                    len(trainset) if train_num_steps is None else train_num_steps
                )
                config.algo.optim_params[k]["num_epochs"] = config.train.num_epochs
        # handling for "hbc" and "iris" algorithms
        if config.algo_name == "hbc":
            for sub_algo in ["planner", "actor"]:
                # add info to optim_params of each net
                for k in config.algo[sub_algo].optim_params:
                    config.algo[sub_algo].optim_params[k]["num_train_batches"] = (
                        len(trainset) if train_num_steps is None else train_num_steps
                    )
                    config.algo[sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs
        if config.algo_name == "iris":
            for sub_algo in ["planner", "value"]:
                # add info to optim_params of each net
                for k in config.algo["value_planner"][sub_algo].optim_params:
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_train_batches"] = (
                        len(trainset) if train_num_steps is None else train_num_steps
                    )
                    config.algo["value_planner"][sub_algo].optim_params[k]["num_epochs"] = config.train.num_epochs

    # setup for a new training run
    # Async rollouts log past-epoch results, which fails against wandb's
    # monotonic implicit step — auto-enable local_step before DataLogger
    # does its one-shot wandb.define_metric.
    if (
        getattr(config.experiment.rollout, "async_enabled", False)
        and config.experiment.logging.log_wandb
        and not getattr(config.experiment.logging, "use_local_step", False)
    ):
        with config.values_unlocked():
            config.experiment.logging.use_local_step = True
        print("async rollouts enabled: auto-enabling logging.use_local_step for wandb")
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )

    if wandb_run is not None:
        assert wandb_model is not None, (
            "--wandb_model is required when --wandb_run is given "
            "(pick a checkpoint name like model_50.pth from the wandb run's files)"
        )
        assert not resume, "--resume and --wandb_run are mutually exclusive"
        print("*" * 50)
        print("fetching ckpt '{}' from wandb run {}".format(wandb_model, wandb_run))
        wandb_ckpt_path = _fetch_wandb_checkpoint(wandb_run, wandb_model, time_dir)
        print("downloaded to {}".format(wandb_ckpt_path))
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=wandb_ckpt_path)
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        resume = True  # reuse the existing resume variable_state handling below
        print("*" * 50)
    elif resume:
        # load ckpt dict
        print("*" * 50)
        print("resuming from ckpt at {}".format(latest_model_path))
        try:
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_path)
        except Exception as e:
            print("got error: {} when loading from {}".format(e, latest_model_path))
            print("trying backup path {}".format(latest_model_backup_path))
            ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path=latest_model_backup_path)
        # load model weights and optimizer state
        model.deserialize(ckpt_dict["model"], load_optimizers=True)
        print("*" * 50)

    # if checkpoint is specified, load in model weights;
    # will not use ckpt_path if resuming training
    ckpt_path = config.experiment.ckpt_path
    if (ckpt_path is not None) and (not resume):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint

        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    # save the config as a json file
    with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # print all warnings before training begins
    print("*" * 50)
    print(
        "Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully."
    )
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1.0 for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()
    # accumulated wall time for the current stretch of training epochs; reset
    # after each rollout so Perf/learn_time reports per-eval-interval cost.
    learn_start_time = time.time()

    # Async rollout manager (optional). When enabled, the env pool is
    # handed off to a background thread so training keeps making progress
    # during evaluation rollouts. The main thread no longer calls
    # ``rollout_with_stats`` directly; it submits snapshots and drains
    # completed results.
    async_rollout_enabled = bool(
        config.experiment.rollout.enabled
        and getattr(config.experiment.rollout, "async_enabled", False)
    )
    async_rollouts = None
    if async_rollout_enabled:
        async_rollouts = AsyncRolloutManager(
            envs=envs,
            model=model,
            horizon=config.experiment.rollout.horizon,
            num_episodes=config.experiment.rollout.n,
            use_warp=config.experiment.rollout.use_warp,
            use_goals=config.use_goals,
            video_dir=video_dir if config.experiment.render_video else None,
            video_skip=config.experiment.get("video_skip", 5),
            terminate_on_success=config.experiment.rollout.terminate_on_success,
            queue_size=getattr(config.experiment.rollout, "async_queue_size", 2),
        )

    start_epoch = 1  # epoch numbers start at 1
    if resume:
        # load variable state needed for train loop
        variable_state = ckpt_dict["variable_state"]
        start_epoch = (
            variable_state["epoch"] + 1
        )  # start at next epoch, since this recorded the last epoch of training completed
        best_valid_loss = variable_state["best_valid_loss"]
        best_return = variable_state["best_return"]
        best_success_rate = variable_state["best_success_rate"]
        print("*" * 50)
        print("resuming training from epoch {}".format(start_epoch))
        print("*" * 50)

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record(_perf_key(k, "train"), v, epoch)
            else:
                data_logger.record(_metric_key(k, "Train"), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    epoch=epoch,
                    validate=True,
                    num_steps=valid_num_steps,
                    obs_normalization_stats=obs_normalization_stats,
                )
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(_perf_key(k, "valid"), v, epoch)
                else:
                    data_logger.record(_metric_key(k, "Valid"), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        fire_rollout = (
            config.experiment.rollout.enabled
            and (epoch > config.experiment.rollout.warmstart)
            and rollout_check
        )

        if fire_rollout:
            # log cumulative learn wall time since last rollout (seconds)
            data_logger.record("Perf/learn_time", time.time() - learn_start_time, epoch)

        if fire_rollout and async_rollouts is None:
            # Synchronous path (original behavior).
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                use_warp=config.experiment.rollout.use_warp,
            )
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=config.experiment.rollout.n,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )
            updated_stats = _log_rollout_result(
                data_logger=data_logger,
                config=config,
                epoch_for_log=epoch,
                all_rollout_logs=all_rollout_logs,
                video_paths=video_paths,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (
                config.experiment.save.enabled and updated_stats["should_save_ckpt"]
            ) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]
            learn_start_time = time.time()
        elif fire_rollout and async_rollouts is not None:
            # Async path: queue a snapshot, don't block. Results come
            # back on the next few iterations via drain() below.
            async_rollouts.submit(
                epoch=epoch,
                model=model,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )
            learn_start_time = time.time()

        if async_rollouts is not None:
            # Drain any rollouts that completed during training. Each
            # result is logged at its own epoch via local_step so wandb
            # plots them at the correct x-axis position. Rollout-best
            # saves are handled inline with exact-weight snapshots —
            # they do NOT propagate ``should_save_ckpt`` out to the
            # outer save block (which would save the current, newer
            # model under a past-epoch's best-score filename).
            for result in async_rollouts.drain():
                _consume_async_rollout_result(
                    result=result,
                    data_logger=data_logger,
                    config=config,
                    ckpt_dir=ckpt_dir,
                    env_meta_list=env_meta_list,
                    shape_meta_list=shape_meta_list,
                    best_valid_loss=best_valid_loss,
                    best_return=best_return,
                    best_success_rate=best_success_rate,
                    obs_normalization_stats=obs_normalization_stats,
                    action_normalization_stats=action_normalization_stats,
                )

        # get variable state for saving model
        variable_state = dict(
            epoch=epoch,
            best_valid_loss=best_valid_loss,
            best_return=best_return,
            best_success_rate=best_success_rate,
        )

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            epoch_ckpt_path = os.path.join(ckpt_dir, epoch_ckpt_name + ".pth")
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
                shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
                variable_state=variable_state,
                ckpt_path=epoch_ckpt_path,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )
            if config.experiment.logging.log_wandb:
                data_logger.log_checkpoint(epoch_ckpt_path)

        # always save latest model for resume functionality
        print("\nsaving latest model at {}...\n".format(latest_model_path))
        TrainUtils.save_model(
            model=model,
            config=config,
            env_meta=env_meta_list[0] if len(env_meta_list) == 1 else env_meta_list,
            shape_meta=shape_meta_list[0] if len(shape_meta_list) == 1 else shape_meta_list,
            variable_state=variable_state,
            ckpt_path=latest_model_path,
            obs_normalization_stats=obs_normalization_stats,
            action_normalization_stats=action_normalization_stats,
        )

        # keep a backup model in case last.pth is malformed (e.g. job died last time during saving)
        shutil.copyfile(latest_model_path, latest_model_backup_path)
        print("\nsaved backup of latest model at {}\n".format(latest_model_backup_path))

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # Wait for any in-flight async rollouts to finish and log their
    # results before wandb closes. Each result may still trigger a
    # best-score snapshot checkpoint.
    if async_rollouts is not None:
        try:
            for result in async_rollouts.drain_blocking(timeout=None):
                _consume_async_rollout_result(
                    result=result,
                    data_logger=data_logger,
                    config=config,
                    ckpt_dir=ckpt_dir,
                    env_meta_list=env_meta_list,
                    shape_meta_list=shape_meta_list,
                    best_valid_loss=best_valid_loss,
                    best_return=best_return,
                    best_success_rate=best_success_rate,
                    obs_normalization_stats=obs_normalization_stats,
                    action_normalization_stats=action_normalization_stats,
                )
        finally:
            async_rollouts.close()

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, "r"))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = [{"path": args.dataset}]

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(
            config,
            device=device,
            resume=args.resume,
            wandb_run=args.wandb_run,
            wandb_model=args.wandb_model,
        )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug", action="store_true", help="set this flag to run a quick training run for debugging purposes"
    )

    # resume training from latest checkpoint
    parser.add_argument(
        "--resume",
        action="store_true",
        help="set this flag to resume training from latest checkpoint",
    )

    # resume training from a wandb-logged checkpoint
    parser.add_argument(
        "--wandb_run",
        type=str,
        default=None,
        help="(optional) wandb run reference 'entity/project/run_id' to fetch a checkpoint from. "
             "Requires --wandb_model. Training continues in a fresh timestamp dir, but weights, "
             "optimizer state, and epoch counter are loaded from the downloaded checkpoint.",
    )
    parser.add_argument(
        "--wandb_model",
        type=str,
        default=None,
        help="(optional) checkpoint filename within the --wandb_run to download (e.g. 'model_50.pth').",
    )

    args = parser.parse_args()
    main(args)
