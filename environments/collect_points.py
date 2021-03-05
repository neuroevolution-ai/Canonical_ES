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

        self.number_of_time_steps = 300

        self.screen_width = 800
        self.screen_height = 800

        # Radius of agent
        self.agent_radius = 10
        self.point_radius = 8

        self.number_points_to_collect = 5

        # Agent coordinates
        self.agent_position_x = 400
        self.agent_position_y = 400

        self.sensor_range = 100

        self.seed = env_seed

        self.points = [(self.generate_random_number(self.screen_width), self.generate_random_number(self.screen_height)) for _ in
                       range(self.number_points_to_collect)]

        #ob = np.array(
        #    [self.agent_position_x / self.screen_width, self.agent_position_y / self.screen_height, 0.0, 0.0])

        ob_list = list()
        ob_list.append(self.agent_position_x / self.screen_width)
        ob_list.append(self.agent_position_y / self.screen_height)
        for point in self.points:
            ob_list.append(point[0] / self.screen_width)
            ob_list.append(point[1] / self.screen_height)

        self.ob = np.asarray(ob_list)

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

        # Collect points in reach
        rew = 0.0
        points_new = []
        for point in self.points:
            if (point[0] - self.agent_position_x) ** 2 + (point[1] - self.agent_position_y) ** 2 < (
                    self.point_radius + self.agent_radius) ** 2:
                rew += 1.0
            else:
                points_new.append(point)

        self.points = points_new

        # Check points in sensor range
        #sensor_signal = 0.0
        #for point in self.points:
        #    if (point[0] - self.agent_position_x) ** 2 + (point[1] - self.agent_position_y) ** 2 < (
        #            self.point_radius + self.sensor_range) ** 2:
        #        sensor_signal += 0.5

        self.t += 1

        if self.t < self.number_of_time_steps:
            done = False
        else:
            done = True

        self.ob[0] = self.agent_position_x / self.screen_width
        self.ob[1] = self.agent_position_y / self.screen_height

        info = []

        return self.ob, rew, done, info

    def render(self):

        color_red = (0, 0, 255)
        color_blue = (0, 255, 0)
        color_yellow = (255, 255, 0)

        image = 255 * np.ones(shape=[self.screen_width, self.screen_height, 3], dtype=np.uint8)

        # Draw sensor range
        #image = cv2.circle(image, (self.agent_position_x, self.agent_position_y), self.sensor_range, color_yellow, -1)

        # Draw agent
        image = cv2.circle(image, (self.agent_position_x, self.agent_position_y), self.agent_radius, color_red, -1)

        # Draw points
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
