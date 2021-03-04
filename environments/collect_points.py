import numpy as np
import cv2
from gym.envs.registration import register

#register(
#    id="CollectPoints-v0",
#    entry_point="environments.envs:CollectPointsEnv",
#    max_episode_steps=1000  # Arbitrarily chosen, can be overwritten in the CollectPointsEnv class
#)


class CollectPointsEnv:

    def __init__(self, env_seed):

        self.screen_width = 500
        self.screen_height = 500

        # Radius of agent
        self.agent_radius = 10
        self.point_radius = 8

        self.number_points_to_collect = 10

        # Agent coordinates
        self.agent_position_x = 200
        self.agent_position_y = 200

        self.seed = env_seed

        self.points = [(self.generate_random_number(self.screen_width), self.generate_random_number(self.screen_height)) for _ in
                       range(self.number_points_to_collect)]

        self.ob = list()
        self.ob.append(self.agent_position_x / self.screen_width)
        self.ob.append(self.agent_position_y / self.screen_height)
        for point in self.points:
            self.ob.append(point[0] / self.screen_width)
            self.ob.append(point[1] / self.screen_height)

        self.t = 0

    def reset(self):
        return np.asarray(self.ob)

    def step(self, action: np.ndarray):

        self.agent_position_x += int(action[0] * 10)
        self.agent_position_y += int(action[1] * 10)

        self.agent_position_x = min(self.agent_position_x, self.screen_height - self.agent_radius)
        self.agent_position_x = max(self.agent_position_x, self.agent_radius)
        self.agent_position_y = min(self.agent_position_y, self.screen_width - self.agent_radius)
        self.agent_position_y = max(self.agent_position_y, self.agent_radius)

        self.ob[0] = self.agent_position_x / self.screen_width
        self.ob[1] = self.agent_position_y / self.screen_height

        rew = 0.0

        points_new = []
        for point in self.points:
            if (point[0] - self.agent_position_x) ** 2 + (point[1] - self.agent_position_y) ** 2 < (
                    self.point_radius + self.agent_radius) ** 2:
                rew += 1.0
            else:
                points_new.append(point)

        self.points = points_new

        self.t += 1

        if self.t < 1000:
            done = False
        else:
            done = True

        info = []

        return np.asarray(self.ob), rew, done, info

    def render(self):

        color_red = (0, 0, 255)
        color_blue = (0, 255, 0)

        image = 255 * np.ones(shape=[self.screen_width, self.screen_height, 3], dtype=np.uint8)

        image = cv2.circle(image, (self.agent_position_x, self.agent_position_y), self.agent_radius, color_red, -1)

        for point in self.points:
            image = cv2.circle(image, point, self.point_radius, color_blue, -1)

        cv2.imshow("ProcGen Agent", cv2.resize(image, (self.screen_width, self.screen_height)))
        cv2.waitKey(1)

    # https://stackoverflow.com/questions/39714326/random-number-generation-using-python
    # Generates a random number between 0.0 and 1.0
    def generate_random_number(self, range: int):

        m = 2 ** 32
        a = 1664525
        c = 1013904223

        self.seed = (a * self.seed + c) % m
        return int((self.seed / (2 ** 32 - 1)) * range)
