# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import numpy as np
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_rot, quat_conjugate
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .wbr_rl_env_cfg import WbrRlEnvCfg
from .vmc import vmcsolver

def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack((roll, pitch, yaw), dim=-1)

class WbrRlEnv(DirectRLEnv):
    cfg: WbrRlEnvCfg

    def __init__(self, cfg: WbrRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        
        # Find only the actuated joints defined in the configuration
        self.rjoint4_idx, _ = self.robot.find_joints("Left_front_joint")
        self.rjoint1_idx, _ = self.robot.find_joints("Left_rear_joint")
        self.ljoint4_idx, _ = self.robot.find_joints("Right_front_joint")
        self.ljoint1_idx, _ = self.robot.find_joints("Right_rear_joint")
        self.rwheel_idx, _ = self.robot.find_joints("Left_Wheel_joint")
        self.lwheel_idx, _ = self.robot.find_joints("Right_Wheel_joint")
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

        # Initialize vmc solver
        self.lsolver = [vmcsolver() for _ in range(self.num_envs)]
        self.rsolver = [vmcsolver() for _ in range(self.num_envs)]

        self.roll = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.pitch = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.yaw = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.gyro_r = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.gyro_p = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.gyro_y = torch.zeros(self.num_envs, device=self.sim.device, dtype=torch.float32)
        self.vmc_state = torch.zeros((self.num_envs, 4), device=self.sim.device, dtype=torch.float32)

        self.dof_pos = self.robot.data.joint_pos
        self.dof_vel = self.robot.data.joint_vel

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
        quat_copy = self.robot.data.root_quat_w  # 世界坐标系四元数
        rpy = quat_to_euler(quat_copy)
        # 角速度 (3) - 相当于陀螺仪
        ang_vel = self.robot.data.root_ang_vel_b  # body frame角速度
        
        # 保存RPY和角速度供reward函数使用
        self.roll = rpy[:, 0]      # [num_envs]
        self.pitch = rpy[:, 1]     # [num_envs]
        self.yaw = rpy[:, 2]       # [num_envs]
        self.gyro_r = ang_vel[:, 0]  # [num_envs]
        self.gyro_p = ang_vel[:, 1]  # [num_envs]
        self.gyro_y = ang_vel[:, 2]  # [num_envs]

        # print ("Pitch angles:", self.pitch)

        vmc_data = []

        if torch.all(self.dof_pos == 0):
            print("Warning: Observation called before environment reset. Returning zero observations.")
            self.vmc_state = torch.zeros((self.num_envs, 4), device=self.device)  
        else:
            # VMC kinematics
            for i in range(self.num_envs):
                ljoint1_pos = self.dof_pos[i, self.ljoint1_idx].item()
                ljoint4_pos = self.dof_pos[i, self.ljoint4_idx].item()
                rjoint1_pos = self.dof_pos[i, self.rjoint1_idx].item()
                rjoint4_pos = self.dof_pos[i, self.rjoint4_idx].item()

                ljoint1_vel = self.dof_vel[i, self.ljoint1_idx].item()
                ljoint4_vel = self.dof_vel[i, self.ljoint4_idx].item()
                rjoint1_vel = self.dof_vel[i, self.rjoint1_idx].item()
                rjoint4_vel = self.dof_vel[i, self.rjoint4_idx].item()
                
                self.lsolver[i].Resolve(math.pi+ljoint1_pos, -ljoint4_pos)
                self.rsolver[i].Resolve(math.pi-rjoint1_pos,  rjoint4_pos)

                lleg_dot, lphi_dot = self.lsolver[i].VMCVelCal(np.array([ljoint1_vel, -ljoint4_vel]))
                rleg_dot, rphi_dot = self.rsolver[i].VMCVelCal(np.array([rjoint1_vel, -rjoint4_vel]))

                lphi = self.lsolver[i].GetPendulumRadian()
                rphi = self.rsolver[i].GetPendulumRadian()
                llen = self.lsolver[i].GetPendulumLen()
                rlen = self.rsolver[i].GetPendulumLen()

                lalpha = lphi-0.5*math.pi+self.pitch[i].item()
                ralpha = rphi-0.5*math.pi+self.pitch[i].item()

                lalpha_dot = lphi_dot + self.gyro_p[i].item()
                ralpha_dot = rphi_dot + self.gyro_p[i].item()

                vmc_data.append([lalpha, ralpha, lalpha_dot, ralpha_dot])
            
            self.vmc_state = torch.tensor(vmc_data, device=self.device, dtype=torch.float32)

        # Simplified observation: pitch, pitch_velocity, joint positions, joint velocities
        obs = torch.cat(
            (
                self.pitch.unsqueeze(-1),  # pitch angle
                self.gyro_p.unsqueeze(-1),  # pitch angular velocity (y-axis)
                self.dof_pos,  # 6 joint positions
                self.dof_vel * self.cfg.dof_vel_scale,  # 6 joint velocities
                self.vmc_state # VMC data: lalpha, ralpha, lalpha_dot, ralpha_dot
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        lalpha = self.vmc_state[:, 0]
        ralpha = self.vmc_state[:, 1]
        lalpha_dot = self.vmc_state[:, 2]
        ralpha_dot = self.vmc_state[:, 3]

        lalpha_reward = torch.exp(-torch.abs(lalpha)*5.0) * self.cfg.alpha_reward_scale
        ralpha_reward = torch.exp(-torch.abs(ralpha)*5.0) * self.cfg.alpha_reward_scale
        # lalpha_dot_penalty = torch.exp(-torch.abs(lalpha_dot) * 0.1) * self.cfg.alpha_dot_scale
        # ralpha_dot_penalty = torch.exp(-torch.abs(ralpha_dot) * 0.1) * self.cfg.alpha_dot_scale
        
        # Reward for keeping pitch close to 0
        pitch_reward = torch.exp(-torch.abs(self.pitch)*20.0) * self.cfg.pitch_reward_scale
        
        # Penalty for large actions
        actions_cost = torch.sum(self.actions**2, dim=-1) * self.cfg.actions_cost_scale
        
        # print ("obs:", lalpha.mean().item(), ralpha.mean().item(), lalpha_dot.mean().item(), ralpha_dot.mean().item())
        # print ("reward components:", pitch_reward.mean().item(), lalpha_reward.mean().item(), ralpha_reward.mean().item(), actions_cost.mean().item())
        # print ("penalty components:", lalpha_dot_penalty.mean().item(), ralpha_dot_penalty.mean().item())
        total_reward = pitch_reward+lalpha_reward+ralpha_reward-actions_cost #-lalpha_dot_penalty-ralpha_dot_penalty
        
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
