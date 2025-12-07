from isaacgym import gymapi
import numpy as np
from configs.asset_config import AssetConfig
from workspace.iiwa_ws import iiwaScene


class ShelfEnv(iiwaScene):
    def __init__(self):
        self.cfg = AssetConfig()
        
        
        




    def create_shelf(self):
        plank_dim = self.cfg.shelf.shelf_plank_dim
        num_plank = self.cfg.shelf.num_planks
        plank_gap = self.cfg.shelf.plank_gap
        shelf_position = self.cfg.shelf.shelf_loc
        
        self.shelf_asset = []
        plank_pose = gymapi.Transform()
        for i in range(num_plank):
            shelf_pose = gymapi.Transform()
            if i & 1 == 0:
                shelf_pose.p = gymapi.Vec3(shelf_position[0], 0.5 * plank_dim[1] + shelf_position[1], shelf_position[2] - i * plank_gap)
            else:
                shelf_pose.p = gymapi.Vec3(shelf_position[0], 0.5 * plank_dim[1] + shelf_position[1], shelf_position[2] + i * plank_gap)
            asset_options = gymapi.AssetOptions()
            asset_options.thickness = 1
            asset_options.armature = 0.001
            planks_ = self.gym.create_box(self.sim, plank_dim[0], plank_dim[1], plank_dim[2], asset_options)
            self.shelf_asset.append(planks_)
        return self.shelf_asset
    




