# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_rot, quat_conjugate, quat_rotate_inverse
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .wbr_rl_env_cfg import WbrRlEnvCfg

class WbrRlEnv(DirectRLEnv):
    cfg: WbrRlEnvCfg

    def __init__(self, cfg: WbrRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        
        # ----------------------------------------------------------------------
        # 1. Joint & Body Indexing
        # ----------------------------------------------------------------------
        self.wheel_joint_names = ["Left_Wheel_joint", "Right_Wheel_joint"]
        self.leg_joint_names = ["Left_front_joint", "Left_rear_joint", "Right_front_joint", "Right_rear_joint"]
        
        # Get Global Robot DOF Indices (returns python list[int])
        wheel_indices_list, _ = self.robot.find_joints(self.wheel_joint_names)
        leg_indices_list, _ = self.robot.find_joints(self.leg_joint_names)
        
        # Convert to Tensors for torch operations
        self.wheel_dof_indices = torch.tensor(wheel_indices_list, device=self.device)
        self.leg_dof_indices = torch.tensor(leg_indices_list, device=self.device)
        
        # Combined indices for mapping Actions -> Robot Joints
        self._joint_dof_idx = torch.cat((self.leg_dof_indices, self.wheel_dof_indices))

        # === Action Vector Indices ===
        # Since we concatenated (Legs, Wheels) in _joint_dof_idx, the action vector 
        # follows the same order: [Leg1, Leg2, Leg3, Leg4, Wheel1, Wheel2]
        num_legs = len(self.leg_dof_indices)
        num_wheels = len(self.wheel_dof_indices)
        
        # Indices to slice the 'self.actions' tensor
        self.leg_action_indices = torch.arange(num_legs, device=self.device)
        self.wheel_action_indices = torch.arange(num_wheels, device=self.device) + num_legs
        
        # Find indices for Feet/Wheels bodies (using correct case "link")
        self.left_foot_body_idx, _ = self.robot.find_bodies("Left_Wheel_link") 
        self.right_foot_body_idx, _ = self.robot.find_bodies("Right_Wheel_link")
        
        # ----------------------------------------------------------------------
        # 2. Buffers for History & Commands
        # ----------------------------------------------------------------------
        # Commands: [v_x, v_y, omega_z, target_height]
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.command_timer = torch.zeros(self.num_envs, device=self.device)
        
        # History buffers
        # FIX: last_actions must match the size of self.actions (actuated joints only), NOT joint_pos
        self.last_actions = torch.zeros(self.num_envs, len(self._joint_dof_idx), device=self.device)
        
        # last_dof_vel matches full robot joint state (for calculating accelerations)
        self.last_dof_vel = torch.zeros_like(self.robot.data.joint_vel)
        
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Link state buffers
        self.left_foot_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_foot_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # ----------------------------------------------------------------------
        # 3. Config Parsing
        # ----------------------------------------------------------------------
        self.reward_cfg = getattr(self.cfg, "reward_params", {
            "tracking_linx_sigma": 0.25,
            "tracking_liny_sigma": 0.25,
            "tracking_ang_sigma": 0.25,
            "tracking_height_sigma": 0.1,
            "feet_distance": [0.2, 0.5], # [min, max]
            "tracking_gravity_sigma": 0.1
        })
        self.command_cfg = getattr(self.cfg, "command_params", {
            "zero_stable": True,
            "resampling_time": 4.0
        })

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # filter collisions
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # Resample commands if needed
        self._resample_commands()

    def _apply_action(self):
        # Store history before step
        self.last_dof_vel[:] = self.robot.data.joint_vel[:]
        self.last_actions[:] = self.actions[:]
        
        # Apply Torques
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _resample_commands(self):
        """Randomly sample velocity and height commands."""
        dt = self.cfg.sim.dt * self.cfg.decimation
        self.command_timer += dt
        
        # Mask for envs that need new commands
        resample_mask = self.command_timer >= self.command_cfg["resampling_time"]
        resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
        
        if len(resample_ids) > 0:
            # Vx (Forward)
            self.commands[resample_ids, 0] = sample_uniform(-1.0, 1.0, (len(resample_ids),), device=self.device)
            # Vy (Lateral)
            self.commands[resample_ids, 1] = sample_uniform(-0.5, 0.5, (len(resample_ids),), device=self.device)
            # Omega Z (Yaw)
            self.commands[resample_ids, 2] = sample_uniform(-1.5, 1.5, (len(resample_ids),), device=self.device)
            # Height (Z) - Adjust range based on your robot size
            self.commands[resample_ids, 3] = sample_uniform(0.1, 0.4, (len(resample_ids),), device=self.device)
            
            if self.command_cfg["zero_stable"]:
                zero_mask = torch.rand(len(resample_ids), device=self.device) < 0.2
                self.commands[resample_ids[zero_mask], :3] = 0.0

            self.command_timer[resample_ids] = 0.0

    def _get_observations(self) -> dict: 
        # 1. Base State
        root_quat = self.robot.data.root_quat_w
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_w = self.robot.data.root_ang_vel_w
        
        # Convert World Velocity to Body Frame
        self.base_lin_vel = quat_rotate_inverse(root_quat, root_lin_vel_w)
        self.base_ang_vel = quat_rotate_inverse(root_quat, root_ang_vel_w)
        
        # Projected Gravity (Global [0,0,-1] in Body Frame)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.projected_gravity = quat_rotate_inverse(root_quat, gravity_vec)

        # 2. Joint State
        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel
        
        # 3. Feet Positions (for rewards)
        if len(self.left_foot_body_idx) > 0:
             self.left_foot_pos = self.robot.data.body_pos_w[:, self.left_foot_body_idx[0], :]
             self.right_foot_pos = self.robot.data.body_pos_w[:, self.right_foot_body_idx[0], :]

        # 4. Construct Observation Vector
        obs = torch.cat(
            (
                self.base_lin_vel * 2.0,      # Scale for network stability
                self.base_ang_vel * 0.25,
                self.projected_gravity,
                self.commands * 2.0,          # [vx, vy, wz, h]
                (self.dof_pos - self.robot.data.default_joint_pos) * 1.0,
                torch.clamp(self.dof_vel * 0.05, -5.0, 5.0),  # Clamp to prevent explosion
                self.actions                  # Last actions
            ),
            dim=-1,
        )
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1. Tracking
        rew_lin_x = self._reward_tracking_lin_x_vel()
        rew_lin_y = self._reward_tracking_lin_y_vel()
        rew_ang_z = self._reward_tracking_ang_vel()
        rew_height = self._reward_tracking_leg_length()
        
        # 2. Stability / Penalties
        rew_lin_z = self._reward_lin_vel_z()
        rew_ang_xy = self._reward_ang_vel_xy()
        rew_projected_gravity = self._reward_projected_gravity()
        
        # 3. Action / Smoothness
        rew_joint_acc = self._reward_joint_action_rate() 
        rew_wheel_acc = self._reward_reward_wheel_action_rate()
        rew_dof_acc = self._reward_dof_acc()
        rew_dof_vel = self._reward_joint_vel()
        
        # 4. Constraints
        rew_collision = self._reward_collision()
        rew_feet_dist = self._reward_feet_distance()
        rew_calf_sym = self._reward_similar_calf()

        # Summation (Rewards rescaled to keep total per-step reward around -1 to +1)
        total_reward = (
            rew_lin_x * 0.15 +
            rew_lin_y * 0.10 + 
            rew_ang_z * 0.08 + 
            rew_height * 0.12 +
            # Penalties (Subtracted)
            rew_projected_gravity * -0.15 +
            rew_lin_z * -0.25 +
            rew_ang_xy * -0.05 +
            rew_joint_acc * -0.005 +
            rew_wheel_acc * -0.005 +
            rew_dof_acc * -1.0e-7 + 
            rew_dof_vel * -0.0005 +
            rew_collision * -0.1 +
            rew_calf_sym * -0.08 +
            rew_feet_dist * -0.1
        )
        
        # Apply death cost
        total_reward = torch.where(
            self.reset_terminated, 
            torch.ones_like(total_reward) * self.cfg.death_cost, 
            total_reward
        )
        
        return total_reward

    # ------------ Custom Reward Implementations ----------------

    def _reward_tracking_lin_x_vel(self):
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        lin_vel_reward = torch.exp(-lin_vel_error / self.reward_cfg["tracking_linx_sigma"])
        if self.command_cfg["zero_stable"]:
            near_zero_mask = (torch.abs(self.commands[:, 0]) <= 0.01)
            if torch.any(near_zero_mask):
                second_error = torch.square(self.base_lin_vel[near_zero_mask, 0])
                second_reward = torch.exp(-second_error / self.reward_cfg["tracking_linx_sigma"])
                lin_vel_reward[near_zero_mask] += second_reward
        return lin_vel_reward

    def _reward_tracking_lin_y_vel(self):
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_liny_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / self.reward_cfg["tracking_ang_sigma"])
        return ang_vel_reward

    def _reward_tracking_leg_length(self):
        # Uses leg joint indices (Indices relative to full robot state)
        # Note: adjust index offset [1]/[3] if your 'Left_rear_joint' etc order is different
        knee_error = torch.square(self.dof_pos[:, self.leg_dof_indices[1]] - self.commands[:, 3])
        knee_error += torch.square(self.dof_pos[:, self.leg_dof_indices[3]] - self.commands[:, 3])
        return torch.exp(-knee_error / self.reward_cfg["tracking_height_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_joint_action_rate(self):
        # FIX: Use 'leg_action_indices' to slice 'actions' (size 6)
        # NOT 'leg_dof_indices' which are global joint indices (e.g. 5,6,7,8)
        joint_action_rate = self.last_actions[:, self.leg_action_indices] - self.actions[:, self.leg_action_indices]
        return torch.sum(torch.square(joint_action_rate), dim=1)

    def _reward_reward_wheel_action_rate(self):
        # FIX: Use 'wheel_action_indices'
        wheel_action_rate = self.last_actions[:, self.wheel_action_indices] - self.actions[:, self.wheel_action_indices]
        return torch.sum(torch.square(wheel_action_rate), dim=1)

    def _reward_projected_gravity(self):
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward
    
    def _reward_similar_calf(self):
        # Uses global joint positions, so 'leg_dof_indices' is correct here
        rew = torch.square(self.dof_pos[:, self.leg_dof_indices[0]] - self.dof_pos[:, self.leg_dof_indices[2]])
        return rew

    def _reward_joint_vel(self):
        # Uses global joint velocities, so 'leg_dof_indices' is correct here
        return torch.sum(torch.square(self.dof_vel[:, self.leg_dof_indices]), dim=1)

    def _reward_dof_acc(self):
        dt = self.cfg.sim.dt * self.cfg.decimation
        dof_acc = (self.dof_vel - self.last_dof_vel) / dt
        # Clamp acceleration to prevent explosion
        dof_acc = torch.clamp(dof_acc, -100.0, 100.0)
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_collision(self):
        data = self.scene.articulations["robot"].data
        # 向后兼容 IsaacLab 各版本的接触力字段
        contact_forces = getattr(data, "net_contact_forces", None)
        if contact_forces is None:
            contact_forces = getattr(data, "contact_forces", None)
        if contact_forces is None:
            return torch.zeros(self.num_envs, device=self.device)
        force_norm = torch.norm(contact_forces, dim=-1)
        # Clamp force to prevent explosion
        force_norm = torch.clamp(force_norm, 0.0, 1000.0)
        return torch.sum(force_norm.square(), dim=1)
    
    def _reward_feet_distance(self):
        feet_distance = torch.norm(self.left_foot_pos - self.right_foot_pos, dim=-1)
        reward = torch.clamp(self.reward_cfg["feet_distance"][0] - feet_distance, min=0.0, max=1.0) + \
                 torch.clamp(feet_distance - self.reward_cfg["feet_distance"][1], min=0.0, max=1.0)
        return reward
    
    def _reward_survive(self):
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

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
        
        # Reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.command_timer[env_ids] = 0.0
        self.commands[env_ids, 0] = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
        self.commands[env_ids, 1] = sample_uniform(-0.5, 0.5, (len(env_ids),), device=self.device)
        self.commands[env_ids, 2] = sample_uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
        self.commands[env_ids, 3] = sample_uniform(0.1, 0.4, (len(env_ids),), device=self.device)