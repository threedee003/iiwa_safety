from workspace.iiwa_ws import iiwaScene
from isaacgym import gymapi



def run():

      scene = iiwaScene(spawn_table=True,  spawn_cube=False, control_type='velocity')
      while scene.viewer_running():
            scene.step()



      scene.__del__()