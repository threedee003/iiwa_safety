from isaacgym import gymapi
import numpy as np
import open3d.visualization
from configs.asset_config import AssetConfig
from configs.camera_configs import CameraConfigs, SDFConfigs
import open3d 
from workspace.iiwa_ws import iiwaScene
from scipy.spatial.transform import Rotation as R









class iiwaTestBed(iiwaScene):
      '''
      We integrate RGBD cameras for SDF generation in this class.
      
      '''
      def __init__(self):
            super().__init__(control_type="velocity")
            self.camera_cfg = CameraConfigs()
            self.sdf_cfg = SDFConfigs()
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






      def init_tsdf(self):
            # for isaac gym experiments both cameras have same intrinsic. Real world will have different.
            # NOTE: self.intr_left will be used in integrate_sdf()
            self.intr_left = open3d.camera.PinholeCameraIntrinsic(
                  int(self.camera_cfg.left_cam.resolution[0]),
                  int(self.camera_cfg.left_cam.resolution[1]),
                  self.intrinsic_params[0],
                  self.intrinsic_params[1],
                  self.intrinsic_params[2],
                  self.intrinsic_params[3]  
            )
            self.tsdf = open3d.pipelines.integration.ScalableTSDFVolume(
                  voxel_length = self.sdf_cfg.tsdf_params.voxel_len,
                  sdf_trunc = self.sdf_cfg.tsdf_params.sdf_trunc,
                  color_type = open3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            self.integrated_frames = 0


      # def integrate_tsdf(self):
      #       rgb_left, depth_left = self.get_rgbd(self.camera_handles[0])
      #       rgb_right, depth_right = self.get_rgbd(self.camera_handles[1])
      #       # print("rgb_left shape:", rgb_left.shape, "dtype:", rgb_left.dtype)
      #       # print(f"depth left shape: {depth_left.shape} dtype: {depth_left.dtype}")

      #       # depth_left_np = depth_left.astype(np.float32)
      #       # print("depth_left", depth_left_np.dtype, depth_left_np.shape)
      #       # print("min/max:", np.nanmin(depth_left_np), np.nanmax(depth_left_np))
      #       # print("NaNs:", np.isnan(depth_left_np).sum(), "inf:", np.isinf(depth_left_np).sum())


      #       rgb_left = open3d.geometry.Image(rgb_left)
      #       depth_left = open3d.geometry.Image(depth_left.astype(np.float32))
      #       rgbdL = open3d.geometry.RGBDImage.create_from_color_and_depth(
      #             rgb_left, depth_left, depth_scale = 1., depth_trunc = self.camera_cfg.depth_process.clip_dist, convert_rgb_to_intensity = False
      #       )


      #       rgb_right = open3d.geometry.Image(rgb_right)
      #       depth_right = open3d.geometry.Image(depth_right.astype(np.float32))
      #       rgbdR = open3d.geometry.RGBDImage.create_from_color_and_depth(
      #             rgb_right, depth_right, depth_scale = 1., depth_trunc = self.camera_cfg.depth_process.clip_dist, convert_rgb_to_intensity = False
      #       )

      #       # print("rgbdL.color:", type(rgbdL.color), "->", np.asarray(rgbdL.color).shape, np.asarray(rgbdL.color).dtype)
      #       # print("rgbdL.depth:", type(rgbdL.depth), "->", np.asarray(rgbdL.depth).shape, np.asarray(rgbdL.depth).dtype)


      #       # uncertain 
      #       left_extrinsic = self.camera_transforms[0]
      #       right_extrinsic = self.camera_transforms[1]
      #       # print(left_extrinsic)
      #       # print(right_extrinsic)

      #       # T = np.eye(4, dtype=np.float32)
      #       # T_dash =  right_extrinsic @ np.linalg.inv(left_extrinsic)
      #       view_matrix_left = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[0]))
      #       view_matrix_right = np.matrix(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[1]))

      #       print(f"View matrix of left cam : {view_matrix_left}")
      #       print(f"left calculated homo : {np.linalg.inv(left_extrinsic)}")
            


      #       # view_matrix_right[1, 3] = 2
      #       left_ext = np.linalg.inv(view_matrix_left)
      #       right_ext = np.linalg.inv(view_matrix_right)
            


      #       self.tsdf.integrate(rgbdL, self.intr_left, left_ext)
      #       self.tsdf.integrate(rgbdR, self.intr_left, right_ext)
      #       self.integrated_frames += 1




      # def integrate_tsdf(self, use_view_matrix=False, debug_show_pcd=True):
      #       """
      #       use_view_matrix: if True, use gym.get_camera_view_matrix; otherwise use stored gym.get_camera_transform (self.camera_transforms)
      #       debug_show_pcd: if True, create point clouds from each RGBD (with extrinsics) and visualize for alignment check
      #       """

      #       # --- 1) Get images
      #       rgb_left_np, depth_left_np  = self.get_rgbd(self.camera_handles[0])
      #       rgb_right_np, depth_right_np = self.get_rgbd(self.camera_handles[1])

      #       # Quick depth debug (print once)
      #       if self.integrated_frames == 0:
      #             print("Left depth range:", depth_left_np.min(), depth_left_np.max())
      #             print("Right depth range:", depth_right_np.min(), depth_right_np.max())

      #       # --- 2) Wrap into Open3D images
      #       rgb_left  = open3d.geometry.Image(rgb_left_np.astype(np.uint8))
      #       depth_left = open3d.geometry.Image(depth_left_np.astype(np.float32))

      #       rgb_right  = open3d.geometry.Image(rgb_right_np.astype(np.uint8))
      #       depth_right = open3d.geometry.Image(depth_right_np.astype(np.float32))

      #       rgbdL = open3d.geometry.RGBDImage.create_from_color_and_depth(
      #             rgb_left,
      #             depth_left,
      #             depth_scale=1.0,
      #             depth_trunc=self.camera_cfg.depth_process.clip_dist,
      #             convert_rgb_to_intensity=False
      #       )
      #       rgbdR = open3d.geometry.RGBDImage.create_from_color_and_depth(
      #             rgb_right,
      #             depth_right,
      #             depth_scale=1.0,
      #             depth_trunc=self.camera_cfg.depth_process.clip_dist,
      #             convert_rgb_to_intensity=False
      #       )

      #       # --- 3) Prepare GL -> CV camera-frame conversion (always required)
      #       T_GL_TO_CV = np.array([
      #             [1,  0,  0, 0],
      #             [0, -1,  0, 0],
      #             [0,  0, -1, 0],
      #             [0,  0,  0, 1],
      #       ], dtype=np.float32)

      #       # --- 4) Compute extrinsics for left and right (world -> Open3D-camera)
      #       if use_view_matrix:
      #             # gym returns a flat column-major view matrix (OpenGL). Convert to 4x4 and transpose:
      #             V_left_flat  = np.array(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[0]), dtype=np.float32)
      #             V_right_flat = np.array(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handles[1]), dtype=np.float32)

      #             V_left  = V_left_flat.reshape((4,4)).T   # now row-major 4x4 (world->camera in OpenGL coords)
      #             V_right = V_right_flat.reshape((4,4)).T

      #             left_extrinsic  = T_GL_TO_CV @ V_left
      #             right_extrinsic = T_GL_TO_CV @ V_right

      #       else:
      #             # use camera transforms you already stored (camera->world), invert them
      #             T_WC_left  = self.camera_transforms[0].astype(np.float32)   # camera->world
      #             T_WC_right = self.camera_transforms[1].astype(np.float32)

      #             T_CW_left  = np.linalg.inv(T_WC_left)   # world->camera (OpenGL camera coords)
      #             T_CW_right = np.linalg.inv(T_WC_right)

      #             left_extrinsic  = T_GL_TO_CV @ T_CW_left
      #             right_extrinsic = T_GL_TO_CV @ T_CW_right

      #       # --- 5) Optional quick diagnostic: make point clouds and visualise overlap
      #       if debug_show_pcd:
      #             pcdL = open3d.geometry.PointCloud.create_from_rgbd_image(rgbdL, self.intr_left, left_extrinsic)
      #             pcdR = open3d.geometry.PointCloud.create_from_rgbd_image(rgbdR, self.intr_left, right_extrinsic)

      #             # Downsample for speed
      #             pcdL = pcdL.voxel_down_sample(voxel_size=max(0.002, self.sdf_cfg.tsdf_params.voxel_len/2))
      #             pcdR = pcdR.voxel_down_sample(voxel_size=max(0.002, self.sdf_cfg.tsdf_params.voxel_len/2))

      #             print("PCD sizes L,R:", len(pcdL.points), len(pcdR.points))
      #             open3d.visualization.draw_geometries([pcdL.paint_uniform_color([1,0,0]), pcdR.paint_uniform_color([0,1,0])],
      #                                                 window_name="Left (red) vs Right (green) - check alignment")
      #             # return early if visualising
      #             return

      #       # --- 6) Integrate to TSDF (use same intrinsics if they are identical)
      #       self.tsdf.integrate(rgbdL, self.intr_left, left_extrinsic)
      #       self.tsdf.integrate(rgbdR, self.intr_left, right_extrinsic)

      #       self.integrated_frames += 1




















      def save_tsdf_mesh(self):
            mesh = self.tsdf.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            mesh_path = "meshes/mesh.ply"
            open3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Saved mesh at {mesh_path}")            

      def visualise_mesh(self):
            mesh = open3d.io.read_triangle_mesh("meshes/mesh.ply")
            mesh.compute_vertex_normals()
            open3d.visualization.draw_geometries([mesh], window_name = "TSDF Mesh")


