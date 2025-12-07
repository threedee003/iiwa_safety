import os
import argparse
import math
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
import random
from pprint import pprint
from scipy.spatial.transform import Rotation as R
from configs.workspace import iiwaSceneCfg


axes_geom = gymutil.AxesGeometry(0.1)



class iiwaScene:
      def __init__(self,
                   control_type: str,
                   spawn_cube: bool,
                   spawn_table: bool,
                   cfg:  iiwaSceneCfg = iiwaSceneCfg()
                   ):
            self.gym = gymapi.acquire_gym()
            self.sim_params = gymapi.SimParams()
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
            sim_type = gymapi.SIM_PHYSX

            self.control_type = control_type
            available_control_types = ['position', 'velocity']
            assert control_type in available_control_types, f"only available control modes are : {available_control_types}"
            
            self.sim_params.dt = 1 / cfg.env.freq
            if sim_type == gymapi.SIM_FLEX:
                  self.sim_params.substeps = 4
                  self.sim_params.flex.solver_type = 5
                  self.sim_params.flex.num_outer_iterations = 4
                  self.sim_params.flex.num_inner_iterations = 20
                  self.sim_params.flex.relaxation = 0.75
                  self.sim_params.flex.warm_start = 0.8
            elif sim_type == gymapi.SIM_PHYSX:
                  self.sim_params.substeps = cfg.env.substeps
                  self.sim_params.physx.solver_type = cfg.env.solver_type
                  self.sim_params.physx.num_position_iterations = cfg.env.num_position_iterations
                  self.sim_params.physx.num_velocity_iterations = cfg.env.num_velocity_iterations
                  self.sim_params.physx.num_threads = cfg.env.num_threads
                  self.sim_params.physx.use_gpu = cfg.env.use_gpu
            compute_device_id = cfg.env.compute_device_id
            graphics_device_id = cfg.env.graphics_device_id
            self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, sim_type, self.sim_params)
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            plane_params.distance = cfg.env.distance
            plane_params.static_friction = cfg.env.static_friction
            plane_params.dynamic_friction = cfg.env.dynamic_friction
            plane_params.restitution = cfg.env.restitution
            self.gym.add_ground(self.sim, plane_params)



            iiwa_urdf = cfg.robot.iiwa_urdf

            self.asset_options = gymapi.AssetOptions()
            self.asset_options.fix_base_link = False
            self.asset_options.thickness = 0.01
            self.asset_options.disable_gravity = True

            self.asset_options.fix_base_link = True
            self.asset_options.disable_gravity = True
            robot_ = self.gym.load_asset(self.sim, cfg.robot.ROOT, iiwa_urdf, self.asset_options)
            robot_props = self.gym.get_asset_dof_properties(robot_)
            robot_props["driveMode"][7:].fill(cfg.gripper.driveMode)
            robot_props["stiffness"][7:].fill(cfg.gripper.stiffness)
            robot_props["damping"][7:].fill(cfg.gripper.damping)
            robot_props['friction'][7:].fill(cfg.gripper.friction)  



            # 1100, 300
            if self.control_type == 'position':
                  robot_props["driveMode"][:7].fill(cfg.position_control.driveMode)
                  robot_props["stiffness"][:7].fill(cfg.position_control.stiffness)
                  robot_props["damping"][:7].fill(cfg.position_control.damping)
                  # robot_props['stiffness'][:7].fill(0.)
                  # robot_props['damping'][:7].fill(400.)
            elif self.control_type == 'velocity':
                  robot_props["driveMode"][:7].fill(cfg.velocity_control.driveMode)
                  robot_props["stiffness"][:7].fill(0.)
                  robot_props["damping"][:7].fill(cfg.velocity_control.damping)
                  robot_props['friction'][:7].fill(cfg.velocity_control.friction)





            # creating viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

            
            spacing = cfg.env.spacing
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, spacing)
            self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1) 

            self.gym.viewer_camera_look_at(self.viewer,
                                           self.env,
                                           cfg.viewerCamera.cam_loc,
                                           cfg.viewerCamera.look_at 
                                          )



            pose = gymapi.Transform()
            pose.p = cfg.robot.iiwa_pos

            self.robot_handle = self.gym.create_actor(self.env, robot_, pose, "robot", 0, 0)
            if spawn_table:
                  table_ = self._create_table_()
                  self.table_handle = self.gym.create_actor(self.env, table_, self.table_pose, "table", 0, 0)
                  table_props = self.gym.get_actor_rigid_shape_properties(self.env, self.table_handle)
                  table_props[0].friction = 0.3
                  table_props[0].rolling_friction = 0.3
                  table_props[0].torsion_friction = 0.3
                  self.gym.set_actor_rigid_shape_properties(self.env, self.table_handle, table_props)

            


            
            self.gym.set_actor_dof_properties(self.env, self.robot_handle, robot_props)
            self.robot_origin = cfg.robot.iiwa_pos_vector
            self.robot_quat = cfg.robot.iiwa_quat_vector


      def compute_reward(self):
            raise NotImplementedError("Modified in subclass")
      
      def calculate_vel_cmd(self, jt: np.ndarray):
            kp = 2.
            kd = 1.
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            joint_vel = [dof_state['vel'] for dof_state in dof_states]  
            jt_error = jt-np.array(joint_pos[:7])
            vel_cmd = (kp*jt_error - kd*np.array(joint_vel[:7]))
            # print(vel_cmd)
            return vel_cmd
      

      def set_camera(self):
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 1080
            camera_props.height = 900
            self.camera_handle = self.gym.create_camera_sensor(self.env, camera_props)






      
      def reach_jt_position(self, desired_jt: np.ndarray, hack: np.ndarray = None):
            
            
            kp = 4.
            kd = 1.
            assert self.control_type == 'velocity', f"only for velocity control"
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            joint_vel = [dof_state['vel'] for dof_state in dof_states]
            gripper_jts = np.array([0., 0.])
            desired_jt = np.concatenate((desired_jt, gripper_jts), axis = 0)
            jt_error = desired_jt-np.array(joint_pos)
            vel_cmd = (kp*jt_error - kd*np.array(joint_vel))
            if hack is not None:
                  vel_cmd = vel_cmd + hack
            vel_cmd = vel_cmd.tolist()
            # self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, vel_cmd)
            return np.array(vel_cmd[:7])


      def reached_jt(self, desired_jt: np.ndarray, eps: float = None):
            if eps is None:
                  eps = 0.009
            dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
            joint_pos = [dof_state['pos'] for dof_state in dof_states]
            joint_pos.pop()
            joint_pos.pop()
            joint_pos = np.array(joint_pos)
            dist = np.linalg.norm(joint_pos-desired_jt)
            if dist >= eps:
                  return False
            else:
                  return True

  



      def apply_arm_action(self, action: list):
            if self.control_type == 'position':
                  action = np.array(action)
                  dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_ALL)
                  joint_pos = [dof_state['pos'] for dof_state in dof_states]
                  gripper_jts = np.array([joint_pos[7], joint_pos[8]])
                  action = np.concatenate((action, gripper_jts), axis = 0).tolist()
                  self.gym.set_actor_dof_position_targets(self.env, self.robot_handle, action)
            elif self.control_type == 'velocity':
                  # if len(action) == 7:
                  #       act = np.array(action)
                  #       grippers = np.array([0., 0.])
                  #       action = np.concatenate((act, grippers), axis = 0).tolist()
                  self.gym.set_actor_dof_velocity_targets(self.env, self.robot_handle, action)
            else:
                  raise NotImplementedError("I will not implement it.")


      def get_state(self, handle):
            dof_states = self.gym.get_actor_rigid_body_states(self.env, handle, gymapi.STATE_ALL)
            pos = np.array([dof_states['pose']['p']['x'][0], dof_states['pose']['p']['y'][0], dof_states['pose']['p']['z'][0]])
            orien = np.array([dof_states['pose']['r']['x'][0], dof_states['pose']['r']['y'][0], dof_states['pose']['r']['z'][0], dof_states['pose']['r']['w'][0]])
            return pos.astype('float64'), orien.astype('float64')
      
      def to_robot_frame(self, pos, quat):
            robot_pos_world = np.array([12.0, 5.0, 0.7])
            robot_quat_world = np.array([0.0, 0.0, 0.707, 0.707]) 
            robot_rot_world = R.from_quat(robot_quat_world)
            T_robot_world = np.eye(4)
            T_robot_world[:3, :3] = robot_rot_world.as_matrix()
            T_robot_world[:3, 3] = robot_pos_world
            T_world_robot = np.linalg.inv(T_robot_world)
            point_world_hom = np.ones(4)
            point_world_hom[:3] = pos
            point_robot_hom = T_world_robot @ point_world_hom
            point_pos_robot = point_robot_hom[:3]
            if quat is None:
                  quat = np.array([1., 0., 0., 0.])            
            elif np.linalg.norm(quat) == 0.:
                  quat = np.array([1., 0., 0., 0.])
            point_rot_world = R.from_quat(quat)
            point_rot_robot = robot_rot_world.inv() * point_rot_world
            point_quat_robot = point_rot_robot.as_quat()  
            return point_pos_robot, point_quat_robot


      def __del__(self):
            print("scene deleted")

      def orientation_error(self, desired, current):
            cc = quat_conjugate(current)
            q_r = quat_mul(desired, cc)
            return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


      
      def get_viewer(self):
            return self.viewer
      
      def get_sim(self):
            return self.sim
      
      def get_gym(self):
            return self.gym
      
      def get_env(self):
            return self.env
      
      def get_robot_handle(self):
            return self.robot_handle
      

      def step(self):
            t = self.gym.get_sim_time(self.sim)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
            return t


      def get_joint_pos_vel(self, handle) -> tuple:
            dof_states = self.gym.get_actor_dof_states(self.env, handle, gymapi.STATE_ALL)
            joint_pos = [float(dof_state['pos']) for dof_state in dof_states]
            joint_vel = [float(dof_state['vel']) for dof_state in dof_states]
            return joint_pos, joint_vel
      

      def post_physics_step(self):
            raise NotImplementedError("In child class")

      def pre_physics_step(self):
            raise NotImplementedError("In child class")
      

      def viewer_running(self):
            return not self.gym.query_viewer_has_closed(self.viewer)



def deg2rad(values):
      const = math.pi/ 180.
      return const*values











