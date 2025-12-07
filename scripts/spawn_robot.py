from workspace.iiwa_ws import iiwaScene
from isaacgym import gymapi



def run():
      act = 7 * [0.]
      act[3] = 1.1
      scene = iiwaScene(spawn_table=True,  spawn_cube=False, control_type='position')
      while scene.viewer_running():
            scene.step()
            scene.apply_arm_action(action = act)


      scene.__del__()