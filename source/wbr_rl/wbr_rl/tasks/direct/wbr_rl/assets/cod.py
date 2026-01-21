# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom bipedal robot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get current directory
_current_dir = os.path.dirname(os.path.realpath(__file__))

##
# Configuration
##

COD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_current_dir, "Robot.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=10.0,  # Increased to help resolve penetration faster
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,  # Increased for better collision resolution
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),  # Initial position - raised to prevent wheels sinking
        joint_pos={
            "Left_front_joint": 0.0,
            "Left_rear_joint": 0.0,
            "Right_front_joint": 0.0,
            "Right_rear_joint": 0.0,
            "Left_Wheel_joint": 0.0,
            "Right_Wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Left_front_joint", "Left_rear_joint", "Right_front_joint", "Right_rear_joint"],
            effort_limit=100,
            stiffness=50.0,
            damping=5.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["Left_Wheel_joint", "Right_Wheel_joint"],
            effort_limit=50,
            stiffness=20.0,
            damping=2.0,
        ),
    },
)
