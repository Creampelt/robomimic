"""
This file contains the gym environment wrapper that is used
to provide a standardized environment API for training policies and interacting
with metadata present in datasets.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class EnvIsaacLab(EB.EnvBase):
    """Wrapper class for gym"""

    def __init__(
        self,
        env_name: str,
        render: bool = False,
        render_offscreen: bool = False,
        use_image_obs: bool = False,
        use_depth_obs: bool = False,
        num_envs: int | None = None,
        seed: int | None = None,
        episode_length: int | None = None,
        **kwargs,
    ):
        """
        Args:
            env_name (str): name of environment. Only needs to be provided if making a different
                environment from the one in @env_meta.

            render (bool): ignored - gym envs always support on-screen rendering

            render_offscreen (bool): ignored - gym envs always support off-screen rendering

            use_image_obs (bool): ignored - gym envs don't typically use images
        """
        from isaaclab_tasks.utils import load_cfg_from_registry

        self._init_kwargs = deepcopy(kwargs)
        self._env_name = env_name
        self._current_obs = None
        self._current_reward = None
        self._current_done = None
        self._done = None
        # load env config
        env_cfg = load_cfg_from_registry(env_name, "env_cfg_entry_point")
        # reduce num_envs to necessary # of rollouts, otherwise keep env_cfg.num_envs
        if num_envs and num_envs < env_cfg.scene.num_envs:
            env_cfg.scene.num_envs = num_envs
        if seed is not None:
            env_cfg.seed = seed
        if episode_length:
            env_cfg.episode_length_s = episode_length / (env_cfg.sim.dt * env_cfg.decimation)
        self.env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array" if render_offscreen else None, **kwargs)

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

        env = self.env.unwrapped
        assert isinstance(env, ManagerBasedRLEnv) or isinstance(env, DirectRLEnv)
        return env

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """
        Step in the environment with an action.

        Args:
            action (torch.Tensor): action to take

        Returns:
            observation (dict): new observation dictionary
            reward (torch.Tensor): environment rewards for this step
            done (torch.Tensor): whether the task is done per environment
            info (dict): extra information
        """
        obs_dict, rew, terminated, truncated, extras = self.env.step(action)
        # compute dones
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self.get_observation(obs_dict), rew, dones, extras

    def reset(self) -> dict[str, torch.Tensor]:
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """
        obs_dict, extras = self.env.reset()
        return self.get_observation(obs_dict)

    def render(
        self,
        mode: str = "human",
        height: int | None = None,
        width: int | None = None,
        camera_name=None,
        **kwargs,
    ):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        return self.env.render()
        # if mode == "human":
        #     return self.env.render(mode=mode, **kwargs)
        # if mode == "rgb_array":
        #     return self.env.render(mode="rgb_array", height=height, width=width)
        # else:
        #     raise NotImplementedError("mode={} is not implemented".format(mode))

    def reset_to(self, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains:
                - states (np.ndarray): initial state of the mujoco environment

        Returns:
            observation (dict): observation dictionary after setting the simulator state
        """
        raise NotImplementedError

    def get_observation(self, obs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current flat observation vector to wrap and provide as a dictionary.
                If not provided, uses self._current_obs.
        """
        if obs is None:
            if hasattr(self.env, "observation_manager"):
                obs, _ = self.unwrapped.observation_manager.compute()  # type: ignore
            else:
                obs, _ = self.unwrapped._get_observations()  # type: ignore
        assert isinstance(obs, dict) and "imitation" in obs and isinstance(obs["imitation"], dict)
        return obs["imitation"]

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        raise NotImplementedError

    def get_reward(self):
        """
        Get current reward.
        """
        raise NotImplementedError

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        raise NotImplementedError

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """
        if hasattr(self.unwrapped, "_check_success"):
            return self.unwrapped._check_success()

        # gym envs generally don't check task success - we only compare returns
        return {"task": False}

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        if hasattr(self.unwrapped, "action_manager"):
            return self.unwrapped.action_manager.total_action_dim
        else:
            return gym.spaces.flatdim(self.unwrapped.single_action_space)

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        return self._env_name

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.ISAACLAB_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        render=None,
        render_offscreen=None,
        use_image_obs=None,
        use_depth_obs=None,
        **kwargs,
    ):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvGym instance)
        """

        # make sure to initialize obs utils so it knows which modalities are image modalities.
        # For currently supported gym tasks, there are no image observations.
        obs_modality_specs = {
            "obs": {
                "low_dim": ["flat"],
                "rgb": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        return cls(env_name=env_name, **kwargs)

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        return self.unwrapped

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)
