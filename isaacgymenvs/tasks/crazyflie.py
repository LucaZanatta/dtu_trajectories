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
import csv
from isaacgymenvs.utils.torch_jit_utils import copysign, get_euler_xyz
# from skrl.utils import isaacgym_utils

class Crazyflie(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        num_observations = 13
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

        # control tensors
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        
        # CTBR added
        self.controller = CTRBctrl(self.num_envs, device=self.device)
        self.friction = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device)
        # trajectory
        self.trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/line_x.csv')
        # self.trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/circle.csv')
        # self.trajectory = pd.read_csv('isaacgymenvs/tasks/trajectory/ouroboros.csv')
        self.trajectory_len = torch.tensor(len(self.trajectory), dtype=torch.int32, device=self.device)
        self.tra_index = torch.arange(0, self.trajectory_len, 1, dtype=torch.int32, device=self.device)
        self.tra_index = self.tra_index.unsqueeze(0).expand(self.num_envs, -1)
        
        coordinates = self.trajectory[['X', 'Y', 'Z']].values
        self.trajectory = torch.tensor(coordinates, dtype=torch.float32, device=self.device)
        self.x = self.trajectory[:, 0]
        self.y = self.trajectory[:, 1]
        self.z = self.trajectory[:, 2]
        self.len_of_traj = len(self.trajectory)
                
        # taret index for envs
        # initial position of target is [1] point in the trajectory
        # initial position of drone  is [0] point in the trajectory
        # so here I set, when starting, last targt is [0], current target is [1], next target is [2], next next target is [3]
        
        # last target
        self.target_index_last = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.target_pos_last = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_pos_last[:,0] = self.x[0]
        self.target_pos_last[:,1] = self.y[0]
        self.target_pos_last[:,2] = self.z[0]
        
        # current target
        self.target_index = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_pos[:,0] = self.x[1]
        self.target_pos[:,1] = self.y[1]
        self.target_pos[:,2] = self.z[1]
        
        # next target
        self.target_index_next = 2*torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self.target_pos_next = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_pos_next[:,0] = self.x[2]
        self.target_pos_next[:,1] = self.y[2]
        self.target_pos_next[:,2] = self.z[2]
        
        # next next target
        self.target_index_next_next = 3*torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self.target_pos_next_next = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_pos_next_next[:,0] = self.x[3]
        self.target_pos_next_next[:,1] = self.y[3]
        self.target_pos_next_next[:,2] = self.z[3]
        
        # record the last position of the drone, to chek if it is moving towards the target
        self.root_pos_last = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        
        # reset target buffers
        self.reset_target_buf = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        
        # set camera position
        if self.viewer:
            cam_pos = gymapi.Vec3(-1, 0, 3)
            cam_target = gymapi.Vec3(1, 0, 1)
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
        # create a web viewer instance
        # self.web_viewer = isaacgym_utils.WebViewer()
        
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
        default_pose.p.z = 1 # set initial height to 1

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

        num_resets = len(env_ids)
        actor_indices = self.all_actor_indices[env_ids].flatten()
        
        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(-0, 0, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 1] += torch_rand_float(-0, 0, (num_resets, 1), self.device).flatten()
        self.root_states[env_ids, 2] += torch_rand_float(-0, 0, (num_resets, 1), self.device).flatten()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # reset target index, index will be +1 in self.set_targets
        self.target_index_last[env_ids] = -1
        self.target_index[env_ids] = 0
        self.target_index_next[env_ids] = 1
        self.target_index_next_next[env_ids] = 2
        
        self.set_targets(env_ids)
        
    def set_targets(self,env_ids):
        # env_ids is the indices of the envs that need to be reset for target

        self.target_index_last[env_ids] += 1
        self.target_index_last[self.target_index_last > (self.len_of_traj-1)] = self.len_of_traj-1 # if need drone stay at last traj, change 0 to self.len_of_traj-1
        self.target_pos_last[env_ids,0] = self.x[self.target_index_last[env_ids]]
        self.target_pos_last[env_ids,1] = self.y[self.target_index_last[env_ids]]
        self.target_pos_last[env_ids,2] = self.z[self.target_index_last[env_ids]]
        
        self.target_index[env_ids] += 1
        self.target_index[self.target_index > (self.len_of_traj-1)] = self.len_of_traj-1 # if need drone stay at last traj, change 0 to self.len_of_traj-1
        self.target_pos[env_ids,0] = self.x[self.target_index[env_ids]]
        self.target_pos[env_ids,1] = self.y[self.target_index[env_ids]]
        self.target_pos[env_ids,2] = self.z[self.target_index[env_ids]]
        
        self.target_index_next[env_ids] += 1
        self.target_index_next[self.target_index_next > (self.len_of_traj-1)] = self.len_of_traj-1 # if need drone stay at last traj, change 0 to self.len_of_traj-1
        self.target_pos_next[env_ids,0] = self.x[self.target_index_next[env_ids]]
        self.target_pos_next[env_ids,1] = self.y[self.target_index_next[env_ids]]
        self.target_pos_next[env_ids,2] = self.z[self.target_index_next[env_ids]]
        
        self.target_index_next_next[env_ids] += 1
        self.target_index_next_next[self.target_index_next_next > (self.len_of_traj-1)] = self.len_of_traj-1 # if need drone stay at last traj, change 0 to self.len_of_traj-1
        self.target_pos_next_next[env_ids,0] = self.x[self.target_index_next_next[env_ids]]
        self.target_pos_next_next[env_ids,1] = self.y[self.target_index_next_next[env_ids]]
        self.target_pos_next_next[env_ids,2] = self.z[self.target_index_next_next[env_ids]]
        
        self.reset_target_buf[env_ids] = 0
        
    def pre_physics_step(self, _actions):
        
        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
    
        # set targets
        reset_env_ids_target = self.reset_target_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids_target) > 0:
            self.set_targets(reset_env_ids_target)
            
        actions = _actions.to(self.device)

        ######
        # NN #
        ######
        total_torque = torch.clamp(actions[:, 0:3], -0.001, 0.001) # 0.0004
        common_thrust = torch.clamp(actions[:, 3], 0, 1.5) # 1.5


        ###############
        # NN and CTBR #
        ###############
        # total_torque, common_thrust = self.controller.update(actions, 
        #                                                 self.root_quats, 
        #                                                 self.root_linvels, 
        #                                                 self.root_angvels)
        # # self.forces[:, 0, 2] = common_thrust
        # print("common_thrust: ", common_thrust)
        # print("total_torque: ", total_torque)
        # self.friction[:, 0, :] = 0.002*torch.sign(self.controller.body_drone_linvels)*self.controller.body_drone_linvels**2
        # self.friction[:, 0, :] = -0.005*torch.sign(self.root_linvels)*self.root_linvels**2
        # self.friction = torch.clamp(self.friction, -0.01, 0.01)
        # roll, pitch, yaw = get_euler_xyz(self.root_quats)
        self.forces[:, 0, 2] = common_thrust
        # print("roll: ", roll)
        # print("pitch: ", pitch)
        # print("yaw: ", yaw)

        # force_x = common_thrust * (torch.sin(yaw) * torch.sin(pitch) + torch.cos(yaw) * torch.sin(roll) * torch.cos(pitch))
        # force_y = common_thrust * (torch.sin(yaw) * torch.sin(roll) * torch.cos(pitch) - torch.cos(yaw) * torch.sin(pitch))
        # force_z = common_thrust * torch.cos(pitch) * torch.cos(roll)

        # self.forces[:, 0, 0] = force_x
        # self.forces[:, 0, 1] = force_y
        # self.forces[:, 0, 2] = force_z
        # print("self.root_linvels: ", self.root_linvels)
        # print("self.controller.body_drone_linvels: ", self.controller.body_drone_linvels)
        # print("sellf.friction: ", self.friction)
        # print("self.forces: ", self.forces)
        # print("total_torque: ", total_torque)
        
        # Log the value of self.root_linvels in a CSV file
        # velocity = torch.norm(self.root_linvels, dim=-1)
        # with open('log/root_linvels_log.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(self.num_envs):
        #         writer.writerow([self.root_linvels[i, 0].item(), self.root_linvels[i, 1].item(), self.root_linvels[i, 2].item()])
        
        # # Log the value of self.controller.body_drone_linvels in a CSV file
        # with open('log/body_drone_linvels_log.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(self.num_envs):
        #         writer.writerow([self.controller.body_drone_linvels[i, 0].item(), self.controller.body_drone_linvels[i, 1].item(), self.controller.body_drone_linvels[i, 2].item()])
        
        # # Log the value of self.friction in a CSV file
        # with open('log/friction_log.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(self.num_envs):
        #         writer.writerow([self.friction[i, 0, 0].item(), self.friction[i, 0, 1].item(), self.friction[i, 0, 2].item()])
            
        # # Log the value of self.forces in a CSV file
        # with open('log/forces_log.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(self.num_envs):
        #         writer.writerow([self.forces[i, 0, 0].item(), self.forces[i, 0, 1].item(), self.forces[i, 0, 2].item()])
                
        # # Log the value of self.forces in a CSV file
        # with open('log/torque_log.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(self.num_envs):
        #         writer.writerow([total_torque[i, 0].item(), total_torque[i, 1].item(), total_torque[i, 2].item()])
            
        # # Log the value of velocity in a CSV file
        # with open('log/velocity_log.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     for i in range(self.num_envs):
        #         writer.writerow([velocity[i].item()])

        

        # clear actions for reset envs
        self.forces[reset_env_ids] = 0.0
        total_torque[reset_env_ids] = 0.0

        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

        # Apply forces and torques to the drone
        self.gym.apply_rigid_body_force_tensors(self.sim, 
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
        self.rew_buf[:], self.reset_buf[:], self.reset_target_buf[:], self.root_pos_last[:] = compute_crazyflie_reward(
            self.tra_index,
            self.trajectory,
            self.trajectory_len, 
            self.target_index,
            self.root_pos_last,
            self.root_positions,
            self.target_pos_last,
            self.target_pos,
            self.target_pos_next,
            self.target_pos_next_next,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length,
            self.reset_target_buf,
        )
        
def write_to_csv(data, filename):
    filepath = 'log/'+filename+'.csv'
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open('log/'+filename+'.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            writer.writerow([data[i].item()])
    
        
#####################################################################
###=========================jit functions=========================###
 
# @torch.jit.script
def compute_crazyflie_reward(tra_index, trajectory, trajectory_len ,target_index, root_pos_last,root_positions, target_pos_last, target_pos, target_pos_next, target_pos_next_next ,root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length,reset_target_buf):
    # type: (Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    velocity = torch.norm(root_linvels, dim=-1)
    
    ######################
    # distance to target #
    ######################
    target_dist =  torch.sqrt(torch.square(target_pos - root_positions).sum(-1))
    target_dist_next = torch.sqrt(torch.square(target_pos_next - root_positions).sum(-1))
    target_dist_next_next = torch.sqrt(torch.square(target_pos_next_next - root_positions).sum(-1))
    
    ###################
    # calculate error #
    ###################
    # line_vector = target_pos_last - target_pos
    # line_direction = line_vector / torch.norm(line_vector, dim=-1, keepdim=True)
    # root_to_line_vector = root_positions - target_pos
    # perpendicular_distance = torch.norm(torch.cross(root_to_line_vector, line_direction), dim=-1)
    
    #############################################################################################
    # TODO: Here I calculate distance to all points in trajectory. To be modified, lack of test #
    #############################################################################################
    # target_dist_all = torch.norm(root_positions.unsqueeze(1) - trajectory, dim=-1)
    # zero_index = target_dist<0.05 # if drone gets pretty close to target, set the weight to 0
    # true_indices = torch.nonzero(zero_index == True).squeeze()

    # if true_indices.numel() > 0:
    #     tra_index[true_indices, target_index[true_indices]] = 0

    # pos_reward = tra_index**2 / (target_dist_all*10 + 1)
    # pos_reward = 0.005*torch.sum(pos_reward, dim=1)/trajectory_len
    
    #####################################
    # TODO: Weights need to be modified #
    #####################################
    
    # w_1 = target_dist*velocity
    # w_2 = target_dist_next*velocity
    # w_3 = target_dist_next_next*velocity
    
    w_1 = 1
    w_2 = 3
    w_3 = 6

    pos_reward_0 = w_1/(1 + target_dist*10) + w_2/(1 + target_dist_next*10) + w_3/(1 + target_dist_next_next*10)
    pos_reward_0 = pos_reward_0/3

    access = target_dist.clone()
    access[target_dist>0.05] = 0
    access[target_dist<=0.05] = 10
    
    pos_reward_1 = pos_reward_0.clone()
    pos_reward_1 = 0*pos_reward_0
    last_target_dist = torch.sqrt(torch.square(target_pos - root_pos_last).sum(-1))
    pos_reward_1[(last_target_dist-target_dist)>0] = 1.5 * pos_reward_0[(last_target_dist-target_dist)>0]
    
    pos_reward_2 = pos_reward_0.clone()
    pos_reward_2 = 0*pos_reward_0
    target_dist_xyz = torch.square(target_pos - root_positions)
    target_dist_last_xyz = torch.square(target_pos - root_pos_last)
    index_f = (target_dist_last_xyz - target_dist_xyz) >= 0
    index_f = torch.all(index_f,dim=1)
    indices = torch.nonzero(index_f).squeeze()
    pos_reward_2[indices] = 3 * pos_reward_0[indices]


    ####################################################################################
    # velocity reward, if need drone stay at last point of trajectory, use this reward #
    ####################################################################################
    velocity = torch.norm(root_linvels, dim=-1)
    velocity_reward = velocity.clone()
    velocity_reward[target_index < trajectory_len*4/5] = torch.tanh(velocity[target_index < trajectory_len*4/5])
    velocity_reward[target_index >= trajectory_len*4/5] = 1/(1+velocity[target_index >= trajectory_len*4/5]*10)
    
    #############################################################
    # velocity reward, if need loop trajectory, use this reward #
    #############################################################
    # velocity = torch.norm(root_linvels, dim=-1)
    # velocity_reward = velocity.clone()
    
    
    #########################
    # total reward function #
    #########################
    reward = pos_reward_0*0.1 + pos_reward_1 + pos_reward_2 + access + 0.1*velocity_reward*(0.1*pos_reward_0 + pos_reward_1 + pos_reward_2) # fine
    # reward = pos_reward_0*(0.1 + pos_reward_1 + pos_reward_2) + access + velocity_reward*pos_reward_0*(0.1 + pos_reward_1 + pos_reward_2) # testing


    # record the last position of the drone
    root_pos_last = root_positions.clone()
    
    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 0.5, ones, die) # if drone gets too far from target, reset envs
    die = torch.where(root_positions[..., 2] < 0.8, ones, die) # if drone gets too low, reset envs
    
    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die) # if episode length is reached, reset envs
    
    # reset target
    one = torch.ones_like(reset_target_buf)
    next = torch.zeros_like(reset_target_buf)
    next = torch.where(target_dist < 0.05, one, next) # if drone gets pretty close to target, rest target

    return reward, reset, next, root_pos_last
