import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.robots import Bimanual
import numpy as np
from robosuite.utils.input_utils import choose_environment,choose_robots

class render():
    
    def __init__(self,controller_name="JOINT_POSITION"):
        self.env="Lift"
        self.controller_name=controller_name
        self.controller_settings = {
        "OSC_POSE": [6, 6, 0.1],
        "OSC_POSITION": [3, 3, 0.1],
        "IK_POSE": [6, 6, 0.01],
        "JOINT_POSITION": [7, 7, 0.2],
        "JOINT_VELOCITY": [7, 7, -0.1],
        "JOINT_TORQUE": [7, 7, 0.25],
        }

        

        self.action_dim = self.controller_settings[controller_name][0]
        self.num_test_steps = self.controller_settings[controller_name][1]
        self.test_value = self.controller_settings[controller_name][2]


    """
    #commented code not used
    def action(self, steps_per_action = 75,steps_per_rest = 75):

        action_dim=self.controller_settings[self.controller_name][0]
        neutral = np.zeros(action_dim)

        count = 0
        num_test_steps = self.controller_settings[self.controller_name][1]
        # Loop through controller space
        while count < num_test_steps:
            action = neutral.copy()
            for i in range(steps_per_action):
                if self.controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
                    # Set this value to be the scaled axis angle vector
                    vec = np.zeros(3)
                    vec[count - 3] = test_value
                    action[3:6] = vec
                else:
                    action[count] = test_value
                total_action = np.tile(action, n)
                self.env.step(total_action)
                self.env.render()
            for i in range(steps_per_rest):
                total_action = np.tile(neutral, n)
                self.env.step(total_action)
                self.env.render()
            count += 1
    """    

    def createenv(self,env="Lift",robots=None,controller_name="JOINT_POSITION",render=True):
        controller_config = load_controller_config(
            default_controller=controller_name)
        if render:
            print("Warning: Rendering is enabled in the simulator. This will not record the camera observation data")
            
            self.env = suite.make(
                env,
                # Use single arm env
                robots=["Jaco"],
                gripper_types="default",                # use default grippers per robot arm
                controller_configs=controller_config,   # each arm is controlled using OSCjoint_dim
                # (two-arm envs only) arms face each other
                #env_configuration="single-arm-opposed",
                has_renderer=True,                      # on-screen rendering
                render_camera="frontview",              # visualize the "frontview" camera
                camera_names="frontview",
                has_offscreen_renderer=False,           # no off-screen rendering
                control_freq=20,                        # 20 hz control for applied actions
                horizon=200,                            # each episode terminates after 200 steps
                use_object_obs=False,                   # no observations needed
                use_camera_obs=False,                   # no observations needed
            )
        else:
            self.env = suite.make(
                env,
                # Use single arm env
                robots=["Jaco"],
                gripper_types="default",                # use default grippers per robot arm
                controller_configs=controller_config,   # each arm is controlled using OSCjoint_dim
                # (two-arm envs only) arms face each other
                #env_configuration="single-arm-opposed",
                has_renderer=False,                      # on-screen rendering
                render_camera="frontview",              # visualize the "frontview" camera
                camera_names="frontview",
                has_offscreen_renderer=True,           # no off-screen rendering
                control_freq=20,                        # 20 hz control for applied actions
                horizon=200,                            # each episode terminates after 200 steps
                use_object_obs=False,                   # no observations needed
                use_camera_obs=True,                   # no observations needed
            )


    def randomAction(self,frames=120, save_path="!",render=True,debug=False):
        self.createenv(render=render)
        low, high = self.env.action_spec
        self.currentobs=None
        self.observation=[]
        for i in range(60):
            action = np.random.uniform(low, high)
            obs, reward, done, _ = self.env.step(action)
            self.observation.append(obs)
            if render:
                self.env.render()
            if done:
                self.env.reset()
            if i==0 and debug:
                print(obs)
                print(obs.keys())
            self.currentobs=obs
        if save_path!="!":
            np.save(save_path,self.observation)