from isaacgym import gymapi
import numpy as np
import open3d.visualization
from configs.asset_config import AssetConfig
from configs.camera_configs import CameraConfig
import open3d 
from workspace.iiwa_ws import iiwaScene
from scipy.spatial.transform import Rotation as R







class iiwaTestBed(iiwaScene):
      '''
      We integrate RGBD cameras for SDF generation in this class.
      
      '''
      def __init__(self):
            super().__init__(control_type="velocity")
            self.camera_cfg = CameraConfig()
            # 0: left_cam, 1: right_cam
            self.camera_handles = []
            self.camera_transforms = []
            self.camera_setup()

            # fx,  fy, cx, cy
            self.intrinsic_params = self.compute_camera_intrinsics_params()
            self.init_tsdf()


      def camera_setup(self):
            #left_cam, right_cam properties
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.camera_cfg.left_cam.resolution[0]
            camera_props.height = self.camera_cfg.left_cam.resolution[1]
            camera_props.horizontal_fov = self.camera_cfg.left_cam.fov

            left_cam_handle = self.gym.create_camera_sensor(self.env, camera_props)
            right_cam_handle = self.gym.create_camera_sensor(self.env, camera_props)
            left_cam_pos = gymapi.Vec3(
                  self.camera_cfg.left_cam.position[0],
                  self.camera_cfg.left_cam.position[1],
                  self.camera_cfg.left_cam.position[2]
            )
            right_cam_pos = gymapi.Vec3(
                  self.camera_cfg.right_cam.position[0],
                  self.camera_cfg.right_cam.position[1],
                  self.camera_cfg.right_cam.position[2]
            )
            look_at = gymapi.Vec3(
                  self.camera_cfg.left_cam.lookat[0],
                  self.camera_cfg.left_cam.lookat[1],
                  self.camera_cfg.left_cam.lookat[2]
            )
            self.gym.set_camera_location(left_cam_handle, self.env, left_cam_pos, look_at)
            self.gym.set_camera_location(right_cam_handle, self.env, right_cam_pos, look_at)
            self.camera_handles.append(left_cam_handle)
            self.camera_handles.append(right_cam_handle)

            left_cam_transform = self.gym.get_camera_transform(self.sim, self.env, self.camera_handles[0]) # dtype: gymapi.Transform
            right_cam_transform = self.gym.get_camera_transform(self.sim, self.env, self.camera_handles[1])
            self.camera_transforms.append(self.generate_homo_transform(left_cam_transform))
            self.camera_transforms.append(self.generate_homo_transform(right_cam_transform))






      def get_rgbd(self, cam_handle):
            color_image = self.gym.get_camera_image(self.sim, self.env, cam_handle, gymapi.IMAGE_COLOR)
            depth = self.gym.get_camera_image(self.sim, self.env, cam_handle, gymapi.IMAGE_DEPTH)
            rgba = np.reshape(color_image, (self.camera_cfg.left_cam.resolution[1], self.camera_cfg.left_cam.resolution[0], 4))
            rgb = rgba[:,:,:3].astype(np.uint8)
            depth = -depth
            depth = np.clip(depth, 0, self.camera_cfg.depth_process.clip_dist)
     
            return rgb, depth
      

      def get_cam_rot(self, tf: gymapi.Transform):
            q = np.array([tf.r.x, tf.r.y, tf.r.z, tf.r.w], dtype=np.float32)
            rot = R.from_quat(q).as_matrix().astype(np.float)


      def generate_homo_transform(self, tf: gymapi.Transform):
            p = np.array([tf.p.x, tf.p.y, tf.p.z], dtype=np.float32)
            q = np.array([tf.r.x, tf.r.y, tf.r.z, tf.r.w], dtype=np.float32)
            rot = R.from_quat(q).as_matrix().astype(np.float32)

            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot
            T[:3, 3] = p
            return T
      


      # to calculate camera intrinsics
      def compute_camera_intrinsics_params(self):
            hfov_deg = self.camera_cfg.right_cam.fov
            height = self.camera_cfg.right_cam.resolution[1]
            width = self.camera_cfg.right_cam.resolution[0]
            hfov = np.deg2rad(hfov_deg)


            vfov = 2 * np.arctan((height / width) * np.tan(hfov / 2))
            fx = (width  / 2.0) / np.tan(hfov / 2.0)
            fy = (height / 2.0) / np.tan(vfov / 2.0)
            cx = width  / 2.0
            cy = height / 2.0
            return [fx, fy, cx, cy]





