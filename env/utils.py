import numpy as np
import gymnasium as gym

class FlattenMultiDiscreteActions(gym.ActionWrapper):
    """
    A reinforcement learning environment for optimal sensor placement on a cantilever beam using OpenAI Gym.
    
    Spaces:
        - Action Space: Discrete (number_of_sensor * direction)
        - Observation Space: Box for the binary representation of sensor states.
        - Direction ACTION = { "LEFT" : 0,
                  "RIGHT" : 1,
                  'UP'    : 2,
                  'Down': 3}
        -action = {Sensor1 and Left : 0,
                   Sensor1 and Right: 1,
                   Sensor1 and Up : 2,
                   Sensor1 and Down : 3,
                   Sensor2 and Left : 4,
                   Sensor2 and Right: 5,
                   Sensor2 and Up : 6,
                   Sensor2 and Down : 7,
                   ...
                   SensorN and all combination of direction: N *4}
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.nvec = self.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self.action_space.nvec))

    def action(self, action):
        actions = []
        actions.append(action//self.nvec[-1])
        actions.append(action%self.nvec[-1])
        return np.array(actions)