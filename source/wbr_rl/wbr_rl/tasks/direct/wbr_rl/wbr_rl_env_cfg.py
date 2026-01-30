# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .assets.cod import COD_CFG

@configclass
class WbrRlEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 6  # 6 joints
    observation_space = 31  # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + commands(4) + dof_pos(6) + dof_vel(6) + actions(6)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = COD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [
        10.0,  # Left_front_joint
        10.0,  # Left_rear_joint
        10.0,  # Right_front_joint
        10.0,  # Right_rear_joint
        5.0,   # Left_Wheel_joint
        5.0,   # Right_Wheel_joint
    ]

    # Reward parameters - only for pitch stabilization
    alpha_reward_scale: float = 1.2
    pitch_reward_scale: float = 2.0
    actions_cost_scale: float = 0.01
    dof_vel_scale: float = 0.05
    alpha_dot_scale: float = 0.05
    
    death_cost: float = -1.0
    termination_height: float = 0.1
