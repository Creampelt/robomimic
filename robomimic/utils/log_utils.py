"""
This file contains utility classes and functions for logging to stdout, stderr,
and to tensorboard.
"""
import os
import sys
import numpy as np
from datetime import datetime
from contextlib import contextmanager
import textwrap
import time
from tqdm import tqdm
from termcolor import colored

import robomimic

# global list of warning messages can be populated with @log_warning and flushed with @flush_warnings
WARNINGS_BUFFER = []


class PrintLogger(object):
    """
    This class redirects print statements to both console and a file.
    """
    def __init__(self, log_file):
        self.terminal = sys.stdout
        print('STDOUT will be forked to %s' % log_file)
        self.log_file = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        # ensure stdout gets flushed
        self.terminal.flush()


class DataLogger(object):
    """
    Logging class to log metrics to tensorboard and/or retrieve running statistics about logged data.
    """
    def __init__(self, log_dir, config, log_tb=True, log_wandb=False):
        """
        Args:
            log_dir (str): base path to store logs
            log_tb (bool): whether to use tensorboard logging
        """
        self._tb_logger = None
        self._wandb_logger = None
        self._data = dict() # store all the scalar data logged so far
        # When True, use a user-logged ``local_step`` key as the wandb x-axis
        # (via ``define_metric``) instead of wandb's implicit monotonic step.
        # Required by the async rollout path, which needs to log rollout
        # results at past epochs after more training has already logged.
        self._use_local_step = bool(
            getattr(config.experiment.logging, "use_local_step", False)
        )

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir, 'tb'))

        if log_wandb:
            import wandb
            import robomimic.macros as Macros

            # set up wandb api key if specified in macros (env var takes precedence)
            if Macros.WANDB_API_KEY is not None and "WANDB_API_KEY" not in os.environ:
                os.environ["WANDB_API_KEY"] = Macros.WANDB_API_KEY

            # resolve entity: env var (WANDB_ENTITY) > robomimic macro
            wandb_entity = os.environ.get("WANDB_ENTITY") or Macros.WANDB_ENTITY
            assert wandb_entity is not None, (
                "wandb entity is not set. Either export WANDB_ENTITY (e.g. via .env.wandb) "
                "or set WANDB_ENTITY in {base_path}/macros_private.py (run "
                "python {base_path}/scripts/setup_macros.py to create it).".format(
                    base_path=robomimic.__path__[0]
                )
            )
            
            # attempt to set up wandb 10 times. If unsuccessful after these trials, don't use wandb
            num_attempts = 10
            for attempt in range(num_attempts):
                try:
                    # set up wandb
                    self._wandb_logger = wandb

                    # optional wandb group (e.g. dataset variant like "d0")
                    wandb_group = getattr(config.experiment.logging, "wandb_group", None)

                    # run name is the date+time of the run (e.g. 2026-04-15_22-40-51)
                    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    self._wandb_logger.init(
                        entity=wandb_entity,
                        project=config.experiment.logging.wandb_proj_name,
                        name=run_name,
                        group=wandb_group,
                        dir=log_dir,
                        mode=("offline" if attempt == num_attempts - 1 else "online"),
                    )

                    # set up info for identifying experiment
                    wandb_config = {k: v for (k, v) in config.meta.items() if k not in ["hp_keys", "hp_values"]}
                    for (k, v) in zip(config.meta["hp_keys"], config.meta["hp_values"]):
                        wandb_config[k] = v
                    if "algo" not in wandb_config:
                        wandb_config["algo"] = config.algo_name
                    # Include the full training config so every hyperparameter
                    # (algo, train, experiment, observation, …) shows up in the
                    # wandb run's Config panel.
                    try:
                        wandb_config["config"] = config.to_dict()
                    except Exception as e:
                        log_warning("failed to serialize config for wandb: {}".format(e))
                    self._wandb_logger.config.update(wandb_config)

                    if self._use_local_step:
                        # Make every metric plot against ``local_step`` in
                        # the wandb UI. Wandb's internal step keeps
                        # incrementing monotonically on every ``log`` call
                        # (we stop passing ``step=``), but plots read
                        # ``local_step`` which may go backwards — needed
                        # for async rollout results logged after later
                        # training epochs.
                        self._wandb_logger.define_metric("local_step")
                        self._wandb_logger.define_metric("*", step_metric="local_step")

                    break
                except Exception as e:
                    log_warning("wandb initialization error (attempt #{}): {}".format(attempt + 1, e))
                    self._wandb_logger = None
                    time.sleep(30)

    def record(self, k, v, epoch, data_type='scalar', log_stats=False):
        """
        Record data with logger.
        Args:
            k (str): key string
            v (float or image): value to store
            epoch: current epoch number
            data_type (str): the type of data. either 'scalar' or 'image'
            log_stats (bool): whether to store the mean/max/min/std for all data logged so far with key k
        """

        assert data_type in ['scalar', 'image']

        if data_type == 'scalar':
            # maybe update internal cache if logging stats for this key
            if log_stats or k in self._data: # any key that we're logging or previously logged
                if k not in self._data:
                    self._data[k] = []
                self._data[k].append(v)

        # maybe log to tensorboard
        if self._tb_logger is not None:
            if data_type == 'scalar':
                self._tb_logger.add_scalar(k, v, epoch)
                if log_stats:
                    stats = self.get_stats(k)
                    for (stat_k, stat_v) in stats.items():
                        stat_k_name = '{}-{}'.format(k, stat_k)
                        self._tb_logger.add_scalar(stat_k_name, stat_v, epoch)
            elif data_type == 'image':
                if len(v.shape) == 3:
                    v = v[None, ...]
                self._tb_logger.add_images(k, img_tensor=v, global_step=epoch, dataformats="NHWC")

        if self._wandb_logger is not None:
            try:
                if data_type == 'scalar':
                    self._wandb_log({k: v}, epoch)
                    if log_stats:
                        stats = self.get_stats(k)
                        for (stat_k, stat_v) in stats.items():
                            self._wandb_log({"{}/{}".format(k, stat_k): stat_v}, epoch)
                elif data_type == 'image':
                    import wandb
                    self._wandb_log({k: wandb.Image(v)}, epoch)
            except Exception as e:
                log_warning("wandb logging: {}".format(e))

    def log_checkpoint(self, ckpt_path):
        """
        Upload a checkpoint file to wandb as a run artifact (no-op if wandb is
        disabled or the file is missing).

        Args:
            ckpt_path (str): path to checkpoint file on disk
        """
        if self._wandb_logger is None:
            return
        if not ckpt_path or not os.path.isfile(ckpt_path):
            return
        try:
            self._wandb_logger.save(ckpt_path, base_path=os.path.dirname(ckpt_path), policy="now")
        except Exception as e:
            log_warning("wandb checkpoint logging: {}".format(e))

    def log_video(self, k, video_path, epoch, fps=20):
        """
        Upload a video file to wandb (no-op if wandb is disabled or the file is missing).

        Args:
            k (str): logging key (e.g. "Rollout/video/env_name")
            video_path (str): path to video file on disk
            epoch (int): step for wandb logging
            fps (int): video frame rate for wandb player
        """
        if self._wandb_logger is None:
            return
        if not video_path or not os.path.isfile(video_path):
            return
        try:
            import wandb
            self._wandb_log({k: wandb.Video(video_path, fps=fps, format="mp4")}, epoch)
        except Exception as e:
            log_warning("wandb video logging: {}".format(e))

    def _wandb_log(self, payload: dict, epoch: int) -> None:
        """Wandb ``log`` helper that either uses wandb's implicit step or
        a user-provided ``local_step`` key, depending on config.

        In local-step mode, ``epoch`` may be less than a previously
        logged one (e.g. a rollout for epoch 50 arriving while training
        is on epoch 80). Wandb disallows going backwards in its internal
        step, so we log without ``step=`` and let wandb auto-increment,
        then rely on ``define_metric(step_metric='local_step')`` set at
        init to get the correct plot x-axis.
        """
        if self._use_local_step:
            payload = dict(payload)
            payload["local_step"] = epoch
            self._wandb_logger.log(payload)
        else:
            self._wandb_logger.log(payload, step=epoch)

    def get_stats(self, k):
        """
        Computes running statistics for a particular key.
        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats['mean'] = np.mean(self._data[k])
        stats['std'] = np.std(self._data[k])
        stats['min'] = np.min(self._data[k])
        stats['max'] = np.max(self._data[k])
        return stats

    def close(self):
        """
        Run before terminating to make sure all logs are flushed
        """
        if self._tb_logger is not None:
            self._tb_logger.close()

        if self._wandb_logger is not None:
            self._wandb_logger.finish()


class custom_tqdm(tqdm):
    """
    Small extension to tqdm to make a few changes from default behavior.
    By default tqdm writes to stderr. Instead, we change it to write
    to stdout.
    """
    def __init__(self, *args, **kwargs):
        assert "file" not in kwargs
        super(custom_tqdm, self).__init__(*args, file=sys.stdout, **kwargs)


@contextmanager
def silence_stdout():
    """
    This contextmanager will redirect stdout so that nothing is printed
    to the terminal. Taken from the link below:

    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    """
    old_target = sys.stdout
    try:
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            yield new_target
    finally:
        sys.stdout = old_target


def log_warning(message, color="yellow", print_now=True):
    """
    This function logs a warning message by recording it in a global warning buffer.
    The global registry will be maintained until @flush_warnings is called, at
    which point the warnings will get printed to the terminal.

    Args:
        message (str): warning message to display
        color (str): color of message - defaults to "yellow"
        print_now (bool): if True (default), will print to terminal immediately, in
            addition to adding it to the global warning buffer
    """
    global WARNINGS_BUFFER
    buffer_message = colored("ROBOMIMIC WARNING(\n{}\n)".format(textwrap.indent(message, "    ")), color)
    WARNINGS_BUFFER.append(buffer_message)
    if print_now:
        print(buffer_message)


def flush_warnings():
    """
    This function flushes all warnings from the global warning buffer to the terminal and
    clears the global registry.
    """
    global WARNINGS_BUFFER
    for msg in WARNINGS_BUFFER:
        print(msg)
    WARNINGS_BUFFER = []
