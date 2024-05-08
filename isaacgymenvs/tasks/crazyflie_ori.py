# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# CTBR added 

import math
import numpy as np
import os
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
from CTBRcontroller import CTRBctrl
import torch
import pandas as pd


class Crazyflie(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        
        num_observations = 18
        num_actions = 4
        
        bodies_per_env = 1
        
        self.cfg["env"]["numObservations"] = num_observations
        self.cfg["env"]["numActions"] = num_actions
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
                
        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs, 13)
        
        self.root_states = vec_root_tensor
        self.root_positions = self.root_states[:, 0:3]
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.initial_root_states = self.root_states.clone()
        
        # set thrust limits
        max_thrust = 2
        self.thrust_lower_limits = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device, dtype=torch.float32)

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        
        # CTBR added
        self.controller = CTRBctrl(self.num_envs, device=self.device)
        self.friction = torch.zeros((self.num_envs, bodies_per_env, 3), device=self.device, dtype=torch.float32)

        # trajectory
        # trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/line_x.csv')
        # trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xy.csv')
        # trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/line_xyz.csv')
        self.trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')
        # trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/d_circle.csv')
        # trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/d_circle_plus.csv')
        # trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/helix.csv')
        
        self.x = self.trajectory.iloc[:, 0]
        self.y = self.trajectory.iloc[:, 1]
        self.z = self.trajectory.iloc[:, 2]
        
        self.len_of_traj = len(self.trajectory)
                
        # taret index for envs
        self.target_index = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int32)
        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:,2] = 1
        self.reset_target = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]
           
       
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))  

    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/crazyflie2.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi # 4
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        default_pose = gymapi.Transform()
        default_pose.p.x = 0
        default_pose.p.y = 0
        default_pose.p.z = 1 # set initial height to 0.5

        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env, asset, default_pose, "crazyflie", i, 1, 1)            
            self.envs.append(env)
        
        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 4, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z
        
    def reset_idx(self, env_ids):
        
        # print("env_ids: ", env_ids)
        num_resets = len(env_ids)
        actor_indices = self.all_actor_indices[env_ids].flatten()
        
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(-0, 0, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 1] += torch_rand_float(-0, 0, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] += torch_rand_float(-0, 0, (num_resets, 1), self.device).flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.target_index[env_ids] = -1
        self.set_targets(env_ids)
        
    def set_targets(self,env_ids):
        
        self.target_index[env_ids] += 1
        idx = self.target_index.cpu().numpy()

        for i in env_ids:
            if self.target_index[i] >= (self.len_of_traj-1):
                self.target_root_positions[i,0] = torch.from_numpy(self.x[self.len_of_traj-1].values).float().to(self.target_root_positions.device)
                self.target_root_positions[i,1] = torch.from_numpy(self.y[self.len_of_traj-1].values).float().to(self.target_root_positions.device)
                self.target_root_positions[i,2] = torch.from_numpy(self.z[self.len_of_traj-1].values).float().to(self.target_root_positions.device)

            else:
                self.target_root_positions[i,0] = torch.from_numpy(self.x[idx[i]].values).float().to(self.target_root_positions.device)
                self.target_root_positions[i,1] = torch.from_numpy(self.y[idx[i]].values).float().to(self.target_root_positions.device)
                self.target_root_positions[i,2] = torch.from_numpy(self.z[idx[i]].values).float().to(self.target_root_positions.device)

        self.reset_target[env_ids] = 0
        
    def pre_physics_step(self, _actions):

        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)  
            
        # set targets
        reset_env_ids_target = self.reset_target.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids_target) > 0:
            self.set_targets(reset_env_ids_target)
        
        actions = _actions.to(self.device)
        total_torque, common_thrust = self.controller.update(actions, 
                                                        self.root_quats, 
                                                        self.root_linvels, 
                                                        self.root_angvels)
        self.friction[:, 0, :] = -0.02*torch.sign(self.controller.body_drone_linvels)*self.controller.body_drone_linvels**2       
        self.forces[:,0,2] = common_thrust
        self.forces[:,0,:] += self.friction[:,0,:]

        # clear actions for reset envs
        self.forces[reset_env_ids] = 0.0
        total_torque[reset_env_ids] = 0.0
        
        # Apply forces and torques to the drone
        self.gym.apply_rigid_body_force_tensors( self.sim, 
                                            gymtorch.unwrap_tensor(self.forces), 
                                            gymtorch.unwrap_tensor(total_torque),
                                            gymapi.LOCAL_SPACE)
        

    def post_physics_step(self):
        
        
        self.progress_buf += 1
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.compute_observations()
        self.compute_reward()
        
        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

        
    def compute_observations(self):
        target_x = 0.0
        target_y = 0.0
        target_z = 1.0
        self.obs_buf[..., 0] = (target_x - self.root_positions[..., 0]) / 3
        self.obs_buf[..., 1] = (target_y - self.root_positions[..., 1]) / 3
        self.obs_buf[..., 2] = (target_z - self.root_positions[..., 2]) / 3
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        return self.obs_buf
    
    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.reset_target[:] = compute_crazyflie_reward(
            self.root_positions,
            self.target_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length,
            self.reset_target,
        )
        
  
        
#####################################################################
###=========================jit functions=========================###
 
@torch.jit.script
def compute_crazyflie_reward(root_positions, target_root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length,reset_target):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor]

    # distance to target
    # target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (2 - root_positions[..., 2]) * (2 - root_positions[..., 2]))    
    
    pos_reward = 1 / (1 + target_dist * target_dist)

    # print("target_root_positions: ", target_root_positions)
    # print("root_positions: ", root_positions)
    # print("target_dist: ", target_dist)
    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 0.5, ones, die)
    die = torch.where(root_positions[..., 2] < 0.5, ones, die)
    
    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    # reset = torch.where(target_index-(len_of_traj-1) >= 0, ones, die)
    
    # reset target
    one = torch.ones_like(reset_target)
    next = torch.zeros_like(reset_target)
    next = torch.where(target_dist < 0.2, one, next)

    
    return reward, reset, next
