from isaacgym import gymapi
import numpy as np
from configs.asset_config import AssetConfig
import open3d
from workspace.iiwa_testbed_retired import iiwaTestBed



class ShelfEnv(iiwaTestBed):
      def __init__(self):
            super().__init__()
            self.cfg = AssetConfig()
            self.shelf_asset = []
            self.shelf_poses = []
            self.create_shelf()
            self.shelf_handles = []
            for i in range(0, len(self.shelf_asset)):
                  plank_handle = self.gym.create_actor(self.env, self.shelf_asset[i], self.shelf_poses[i], f"plank_{i+1}", 0, 0)
                  self.shelf_handles.append(plank_handle)


            self.gym.prepare_sim(self.sim)

      def create_shelf(self):
            plank_dim = self.cfg.shelf.shelf_plank_dim
            num_plank = self.cfg.shelf.num_planks
            plank_gap = self.cfg.shelf.plank_gap
            shelf_position = self.cfg.shelf.shelf_position
            asset_options = gymapi.AssetOptions()
            asset_options.thickness = 1
            asset_options.armature = 0.001
            asset_options.fix_base_link = True
            

            for i in range(num_plank):
                  shelf_pose = gymapi.Transform()
                  z = shelf_position[2] + i * plank_gap
                  shelf_pose.p = gymapi.Vec3(shelf_position[0],
                                             0.5 * plank_dim[1] + shelf_position[1], 
                                             z)

                  planks_ = self.gym.create_box(self.sim, plank_dim[0], plank_dim[1], plank_dim[2], asset_options)
                  self.shelf_asset.append(planks_)
                  self.shelf_poses.append(shelf_pose)
            return self.shelf_asset
      




