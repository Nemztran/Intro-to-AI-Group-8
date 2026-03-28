import gymnasium_robotics
import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
from gymnasium.spaces import Box
# ObservationWrapper là công cụ để tiền xử lý dữ liệu quan sát của môi trường trước khi agent nhìn thấy nó
class RoboGymObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
    # obs chưá observation chứa tọa độ x, y và vận tốc theo trục x, y của quả bóng xanh
    # obs chứa desired_goal là tọa độ đích cần tới (quả bóng đỏ)
    # obs chứa achieved_goal là tọa độ hiện tại của quả bóng xanh
    # info chứa biến success xem nó đến đc đích ch

    # chuyển obs chứa 3 phần -> vector
    def observation(self, observation):
        obs_map = observation["observation"]
        obs_achieved_goal = observation["achieved_goal"]
        obs_desired_goal = observation["desired_goal"]
        return np.concatenate((obs_map, obs_achieved_goal, obs_desired_goal)).astype(np.float32)