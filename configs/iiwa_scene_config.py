from isaacgym import gymapi
import numpy as np


class iiwaSceneCfg:
    class env:
        freq = 60.0                     # used as 1 / cfg.env.freq for dt
        compute_device_id = 0
        graphics_device_id = 0
        sim_type = gymapi.SIM_PHYSX
        substeps = 2

        # physx props
        solver_type = 1
        num_position_iterations = 10
        num_velocity_iterations = 0
        num_threads = 0
        use_gpu = True

        # plane params
        distance = 0.0
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

        # env layout
        spacing = 1.0   # used for env_lower/env_upper

    class robot:
        # path is relative to ROOT in iiwaScene
        # ROOT = "/home/bikram/Documents/isaacgym/assets"

        ROOT = "/home/tribikram/iiwa_gym/iiwa_rg2"
        iiwa_urdf = "iiwa_wsg.urdf"

        iiwa_pos = gymapi.Vec3(0.0, 0.0, 0.7)
        # these values denote base transformation in world frame.
        iiwa_pos_vector = np.array([0., 0., 0.7])
        iiwa_quat_vector = np.array([0. , 0., 0.706, 0.707])


    class gripper:
        # these are written into robot_props[7:]
        driveMode = gymapi.DOF_MODE_POS
        stiffness = 1100.0
        damping = 300.0
        friction = 0.0

    class position_control:
        # used when control_type == 'position'
        driveMode = gymapi.DOF_MODE_POS
        stiffness = 1100.0
        damping = 300.0

    class velocity_control:
        # used when control_type == 'velocity'
        driveMode = gymapi.DOF_MODE_VEL
        damping = 5.0
        friction = 0.0

    class task:
        # cube initial position in world frame
        # used by _create_cube(cube_pos=cfg.task.cube_pos)
        cube_pos = np.array([12.2, 5.0, 0.8], dtype=np.float32)


    class viewerCamera:
        cam_loc = gymapi.Vec3(2., 0., 3.)
        look_at = gymapi.Vec3(0., 0., 0.7)

