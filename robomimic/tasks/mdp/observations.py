import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def body_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_pos_b, _ = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w.unsqueeze(1),
        asset.data.root_quat_w.unsqueeze(1),
        body_pos_w,
    )
    return body_pos_b.view(-1, 3)


def body_quat_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    _, body_quat_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w.unsqueeze(1),
        asset.data.root_quat_w.unsqueeze(1),
        q02=body_quat_w,
    )
    # robomimic uses xyzw quats
    body_quat_b = math_utils.convert_quat(body_quat_b, to="xyzw")
    # necessary bc isaac lab typing is bad
    assert isinstance(body_quat_b, torch.Tensor)
    return body_quat_b.view(-1, 4)


def object_obs(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position, rotation, and relative position of the object in end effectors' root frame."""
    robot: Articulation = env.scene[eef_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pose_w = object.data.root_pose_w
    eef_pose = robot.data.body_pose_w[:, eef_cfg.body_ids]
    object_pos_to_eefs, _ = math_utils.subtract_frame_transforms(
        eef_pose[..., :3],
        eef_pose[..., 3:],
        object_pose_w[:, :3].unsqueeze(1),
    )
    full_obs = torch.cat([object_pose_w, object_pos_to_eefs.view(-1, 3)], dim=-1)
    return full_obs
