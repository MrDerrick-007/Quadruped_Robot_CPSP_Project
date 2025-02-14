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

import numpy as np
import os


import wandb


from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
import time
import torch

class Quadruped(VecTask):  #framework di isaacgym per fare ambienti paralleli

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
    #questi parametri sono specificati in https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/framework.md
        self.cfg = cfg
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        ##############################################
        self.rew_scales["joint_acc"] = -0.0005 
        self.rew_scales["stumble"] = -2.0 #
        self.rew_scales["action_rate"] = -0.015
        self.rew_scales["hip"] = 0 
        self.rew_scales["air_time"] = 1.0 
        self.rew_scales["orient"] = -1.3 
        self.rew_scales["symmetric"] = -6 
        ##################################################
        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"] #rotazione attorno asse z

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state
        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 48 #numero di input che forniamo alla rete, nel nostro caso sarà 6, 3 assi per acc e 3 per il gyro
        self.cfg["env"]["numActions"] = 12 #numero di output, nel nostro caso deve essere 12 perchè abbiamo 12 attuatori

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        #file:///home/mrderrick/Desktop/isaacgym/docs/programming/tuning.html#tuning-sim-params
        #dt - Passo temporale della simulazione, il valore predefinito è 1/60 s
        ##########################DERRICK
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"] #lunghezza episodi in secondi
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None: 
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        ####################################################------------------------------
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        ####################################################------------------------------
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        #----------------------------------------------------------------------
        # wandb.login()
        # wandb.init(
        #     # Set the project where this run will be logged
        #     project="mytest_rosso",
        #     # Track hyperparameters and run metadata
        #     config={
        #         "learning_rate": 3e-4,
        #         "epochs": 5,
        #     },
        # )

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        #L' oggetto sim contiene contesti fisici e grafici che ti permetteranno di caricare risorse, creare ambienti e interagire con la simulazione.
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params) 
        ## Nel file config.yaml abbiamo fra le prime righe questa definizione della physics_engine: Device config 'physx' or 'flex'
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

#Create GROUND PLANE-------------------------------------------------------------------------------------
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)
#--------------------------------------------------------------------------------------------------------
    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/anymal_c/urdf/quadruped.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE #gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False #questo è quello che abbiamo modificato per far in modo che sia tutto assemblato correttamente
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01 #prima : 1
        asset_options.disable_gravity = False
        #righe di codice che ci consentono di non far penetrarare zampe sul terreno
        asset_options.override_com = True
        asset_options.override_inertia = True
        #--------------------------------------------------------------------------------------------------
        # asset_options.vhacd_enabled = True

        # asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64
        #--------------------------------------------------------------------------------------------------
        anymal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(anymal_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(anymal_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        start_pose.r = gymapi.Quat(0.707107, 0.0, 0.707107, 0.0)


        body_names = self.gym.get_asset_rigid_body_names(anymal_asset)
        self.dof_names = self.gym.get_asset_dof_names(anymal_asset)
        extremity_name = "foot"#"SHANK" if asset_options.collapse_fixed_joints else "FOOT"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "THIGH" in s]
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        dof_props = self.gym.get_asset_dof_properties(anymal_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.anymal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            anymal_handle = self.gym.create_actor(env_ptr, anymal_asset, start_pose, "quadruped", i, 1, 0) #il gruppo di collisione assegnato a ciascun attore corrisponde all'indice dell'ambiente, il che significa che gli attori di ambienti diversi non interagiranno fisicamente tra loro.
            # self.gym.set_actor_rigid_body_states(self.sim, anymal_handle, pose={"position": [0,0,0], "rotation": [1,0,0,1]})
            self.gym.set_actor_dof_properties(env_ptr, anymal_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, anymal_handle)
            self.envs.append(env_ptr)
            self.anymal_handles.append(anymal_handle)

        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], knee_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.anymal_handles[0], "base")

    def pre_physics_step(self, actions): #apply action, mettiamo i movimenti da fare. penso che la NN ci fornisca actions
         # make sure actions buffer is on the same device as the simulation
        self.actions = actions.clone().to(self.device)
        targets = self.action_scale * self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        a = {f"a/action_{i}": a for i, a in enumerate(self.actions[0])}
        #wandb.log(a)
        #b = {f"a/targets_{i}": b for i, b in enumerate(targets[0])}
        #wandb.log(b)


       # wandb.log({"reward": self.rew_buf[0]})

    def post_physics_step(self):
        self.progress_buf += 1
        #time.sleep(300)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)


        if len(env_ids) > 0:
           self.reset_idx(env_ids)
        
        self.compute_observations()
        self.compute_reward(self.actions)

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_actions[:] = self.actions[:]
        

    def compute_reward(self, actions): 

        self.rew_buf[:], self.reset_buf[:] = self.compute_anymal_reward()

        #roba nostra:
        #wandb.log({"reward": self.rew_buf[0]})

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step, per ricaricare sul tensore il valore corrente dello stat
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_anymal_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_pos_scale,
                                                        self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device) #viene data pos random iniziale in modo da allenarla a tutte le varie situazioni
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset #position offest è l'offset random dalla pos di default
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        self.progress_buf[env_ids] = 0 
        self.reset_buf[env_ids] = 1 
        #############################################
        self.last_dof_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        #############################################

    def compute_anymal_reward(
        self
    ):
        # (reward, reset, feet_in air, feet_air_time, episode sums)
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int) -> Tuple[Tensor, Tensor]

        # prepare quantities (TODO: return from obs ?)
        base_quat = self.root_states[:, 3:7]
        base_pos = self.root_states[:, 0:3]
        dof_position_sx_front = self.dof_pos[:, 1] ###  #sono i dof dei leg 
        dof_position_dx_front = self.dof_pos[:, 4] ###
        dof_position_sx_rear = self.dof_pos[:, 7] ###
        dof_position_dx_rear = self.dof_pos[:, 10] ###
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])

        rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * self.rew_scales["ang_vel_z"]

        #our penality (quella di fargli fare movimenti alle gambe per bene)
        dof_symmetric_sx_front = abs(dof_position_sx_front - dof_position_dx_rear)######
        dof_symmetric_dx_front = abs(dof_position_dx_front - dof_position_sx_rear) ######

        rew_dof_symmetric_sx_front = torch.exp(-dof_symmetric_sx_front/0.25) * 0######## 
        rew_dof_symmetric_dx_front = torch.exp(-dof_symmetric_dx_front/0.25) * 0########

        #orientation penalty
        rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]
        #print('\n\norient:', rew_orient)###

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        
        #joint acc
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        # stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"] 

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        #print('\n\shape 1:', self.last_actions.shape, 'shape 2:', rew_action_rate.shape)

        # cosmetic penalty for hip motion, per tenere il leg del cane attorno all'angolo iniziale
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [1, 4, 7, 10]] - self.default_dof_pos[:, [1, 4, 7, 10]]), dim=1)* self.rew_scales["hip"]
        #print('\n\n', rew_hip)

        # air time reward
        # contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=2) > 1.
        penality_air_time_threshold = 0.15 #limite oltre la quale applicare penalità se sta troppo in aria
        penality_factor = -2
        excessive_air_time_penality = (self.feet_air_time > penality_air_time_threshold)*penality_factor
        # if self.feet_air_time > penality_air_time_threshold :
        # print('\n\nfeet_air_time:', self.feet_air_time)###
        #print('\n\n', excessive_air_time_penality)

        contact = self.contact_forces[:, self.feet_indices, 2] > 1. #array boolean che indica se ciascun piede è a contatto con il suolo con una forza sufficiente
        #print('\n\nshape:', contact.shape)###
        first_contact = (self.feet_air_time > 0.) * contact #array bollean che indica i piedi che hanno appena toccato il suolo dopo essere stati in aria
        self.feet_air_time += self.dt #array che tiene traccia del tempo che ciascun piede ha trascorso in aria
        #print('\n\nair time:', self.feet_air_time)###
        #riga sotto calcola quanto il tempo in aria si discosta da 0.25 unitÀ di tempo, se È maggiore È positivo, minore È negativo
        rew_airTime = torch.sum((self.feet_air_time - 0.15) * first_contact + excessive_air_time_penality, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command (se reward in modulo è minore di 0.1 allora viene azzerata)
        self.feet_air_time *= ~contact
        #print('\n\nair time rew: ', rew_airTime)###

        #symmetric penalty
        feet_air_time_FSx = self.feet_air_time[:,0] 
        feet_air_time_FDx = self.feet_air_time[:,1]
        feet_air_time_RSx = self.feet_air_time[:,2] 
        feet_air_time_RDx = self.feet_air_time[:,3]
        rew_symmetric_st = torch.square(feet_air_time_FSx - feet_air_time_RDx) * self.rew_scales["symmetric"]
        rew_symmetric_nd = torch.square(feet_air_time_RSx - feet_air_time_FDx) * self.rew_scales["symmetric"]
        #print('\n\nsym:', rew_symmetric_st, '\nshape:', rew_symmetric_st.shape)

        #prova : misura altezza piedi da terra
        # feet_positions_z = [] #128 x 4
        # for i in range(129) :
        #     rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.envs[i], self.anymal_handles[i], gymapi.STATE_POS)
        #     if len(rigid_body_states) == 0:
        #         print('\n\nerro:', i)
        #     for index in self.feet_indices :
        #         pos_z = rigid_body_states[index]['pos'][2]
        #         feet_positions_z[i].append(pos_z)
        # print('\n\nshape:', feet_positions_z.shape)###


        total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_joint_acc + rew_airTime + rew_stumble + rew_orient + rew_symmetric_st + rew_symmetric_nd + rew_action_rate #+ rew_hip # stampa total-reward così capimao quando satura e fai video al volo, sono senza cell
        total_reward = torch.clip(total_reward, 0., None)
        print(sum(total_reward)/self.num_envs)
    
        # reset agents
        reset = torch.zeros(128, dtype=torch.long, device= 'cuda:0')

        #reset for time out
        time_out = self.progress_buf >= self.max_episode_length - 1  # no terminal reward for time-outs

        #wandb.log({"r/reward": total_reward[0],"r/lin_vel":rew_lin_vel_xy[0],"r/ang_val":rew_ang_vel_z[0], "r/torque":rew_torque[0] })

        #reset for body too low in axis 'z'
        p_x = base_pos[:,0]
        p_y = base_pos[:,1]
        p_z = base_pos[:,2]

        reset_pos = torch.zeros(128, dtype=torch.long, device= 'cuda:0')
        reset_pos[p_z < 0.18] = 1 #quelli che hanno il corpo troppo in basso si resettano

        reset = reset | reset_pos | time_out

        #reset for pitch and roll
        q_x, q_y, q_z, q_w= base_quat[:,0], base_quat[:,1], base_quat[:,2], base_quat[:,3]
        roll = torch.arctan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2))
        pitch = torch.arcsin(2 * (q_w * q_y - q_z * q_x))

        reset_r = torch.zeros(128, dtype=torch.long, device= 'cuda:0')
        reset_r[torch.abs(roll) > 0.4 ] = 1
        reset = reset | reset_r

        reset_p = torch.zeros(128, dtype=torch.long, device= 'cuda:0')
        reset_p[torch.abs(pitch) > 0.4 ] = 1
        reset = reset | reset_p

        # total_reward.detach() : Staccare il tensore dal grafo computazionale
        return total_reward.detach(), reset


#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script



@torch.jit.script
def compute_anymal_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_pos_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    #pog_quat = root_states[:, 0:3]

    #print('eccolo', pog_quat)

    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs
