# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_rot, quat_conjugate
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .wbr_rl_env_cfg import WbrRlEnvCfg


class WbrRlEnv(DirectRLEnv):
    cfg: WbrRlEnvCfg

    def __init__(self, cfg: WbrRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        
        # Find only the actuated joints defined in the configuration
        joint_names = [
            "Left_front_joint", "Left_rear_joint", "Right_front_joint", 
            "Right_rear_joint", "Left_Wheel_joint", "Right_Wheel_joint"
        ]
        self._joint_dof_idx = []
        for joint_name in joint_names:
            idx, _ = self.robot.find_joints(joint_name)
            self._joint_dof_idx.append(idx[0])
        self._joint_dof_idx = torch.tensor(self._joint_dof_idx, dtype=torch.long, device=self.sim.device)
        
        # Initialize rotation tracking
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _get_observations(self) -> dict:
        # Get robot state
        torso_rotation = self.robot.data.root_quat_w
        ang_velocity = self.robot.data.root_ang_vel_w
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        
        # Compute pitch angle and angular velocity
        torso_quat = torch_utils.quat_mul(torso_rotation, self.inv_start_rot)
        _, _, _, pitch, _, _ = compute_rot(
            torso_quat, 
            torch.zeros_like(ang_velocity),  # velocity (not used)
            ang_velocity,
            torch.zeros_like(self.robot.data.root_pos_w),  # targets (not used)
            self.robot.data.root_pos_w
        )
        
        # Simplified observation: pitch, pitch_velocity, joint positions, joint velocities
        obs = torch.cat(
            (
                pitch.unsqueeze(-1),  # pitch angle
                ang_velocity[:, 1].unsqueeze(-1),  # pitch angular velocity (y-axis)
                dof_pos,  # 6 joint positions
                dof_vel * self.cfg.dof_vel_scale,  # 6 joint velocities
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Get pitch angle
        torso_rotation = self.robot.data.root_quat_w
        torso_quat = torch_utils.quat_mul(torso_rotation, self.inv_start_rot)
        _, _, _, pitch, _, _ = compute_rot(
            torso_quat,
            torch.zeros_like(self.robot.data.root_lin_vel_w),
            self.robot.data.root_ang_vel_w,
            torch.zeros_like(self.robot.data.root_pos_w),
            self.robot.data.root_pos_w
        )
        
        # Reward for keeping pitch close to 0
        pitch_reward = torch.exp(-torch.abs(pitch) * 10.0) * self.cfg.pitch_reward_scale
        
        # Penalty for large actions
        actions_cost = torch.sum(self.actions ** 2, dim=-1) * self.cfg.actions_cost_scale
        
        total_reward = pitch_reward - actions_cost
        
        # Apply death cost for terminated agents
        total_reward = torch.where(
            self.reset_terminated, 
            torch.ones_like(total_reward) * self.cfg.death_cost, 
            total_reward
        )
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.robot.data.root_pos_w[:, 2] < self.cfg.termination_height
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
