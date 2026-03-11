"""
The main entry point for training policies from pre-collected data.

This script loads dataset(s), creates a model based on the algorithm specified,
and trains the model. It supports training on various environments with multiple
algorithms from robomimic.

Args:
    algo: Name of the algorithm to run.
    task: Name of the environment.
    name: If provided, override the experiment name defined in the config.
    dataset: If provided, override the dataset path defined in the config.
    log_dir: Directory to save logs.
    normalize_training_actions: Whether to normalize actions in the training data.

This file has been modified from the original robomimic version to integrate with IsaacLab.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app

# suppress logs
kit_args = " --/log/level=error  --/log/fileLogLevel=error --/log/outputStreamLevel=error"

app_launcher = AppLauncher(headless=True, enable_cameras=True, kit_args=kit_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401 E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401 E402
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401 E402

import robomimic.tasks.lift  # noqa: F401 E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
from robomimic.config import config_factory  # noqa: E402
from robomimic.scripts.train import train  # noqa: E402


def main(args: argparse.Namespace):
    """Train a model on a task using a specified algorithm.

    Args:
        args: Command line arguments.
    """
    # load config
    if args.task is not None:
        # obtain the configuration entry point
        cfg_entry_point_key = f"robomimic_{args.algo}_cfg_entry_point"
        task_name = args.task.split(":")[-1]

        print(f"Loading configuration for task: {task_name}")
        print(" ")
        cfg_entry_point_file = gym.spec(task_name).kwargs.pop(cfg_entry_point_key)
        # check if entry point exists
        if cfg_entry_point_file is None:
            raise ValueError(
                f"Could not find configuration for the environment: '{task_name}'."
                f" Please check that the gym registry has the entry point: '{cfg_entry_point_key}'."
            )

        # resolve module path if needed
        if ":" in cfg_entry_point_file:
            mod_name, file_name = cfg_entry_point_file.split(":")
            mod = importlib.import_module(mod_name)
            if mod.__file__ is None:
                raise ValueError(f"Could not find module file for: '{mod_name}'")
            mod_path = os.path.dirname(mod.__file__)
            config_file = os.path.join(mod_path, file_name)
        else:
            config_file = cfg_entry_point_file

        with open(config_file) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        raise ValueError("Please provide a task name through CLI arguments.")

    if args.dataset is not None:
        config.train.data = args.dataset
    if args.name is not None:
        config.experiment.name = args.name
    if args.epochs is not None:
        config.train.num_epochs = args.epochs

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = f"run failed with error:\n{e}\n\n{traceback.format_exc()}"
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--algo", type=str, default=None, help="Name of the algorithm.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional: Number of training epochs. If specified, overrides the number of epochs from the JSON training"
            " config."
        ),
    )

    args = parser.parse_args()

    # run training
    main(args)
    # close sim app
    simulation_app.close()
