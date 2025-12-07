from isaacgym import gymapi
import numpy as np




class AssetConfig:
    class shelf:
        # width, length, thickness
        shelf_plank_dim = [0.4, 1.5, 0.04]
        # please keep the num planks upto 3 or 4
        # depending on the plank gap for reachability
        num_planks = 3
        plank_gap = 0.5
        shelf_position = [0.5, 0., 1.]
        shelf_orientation = [0., 0., 0., 1.]
