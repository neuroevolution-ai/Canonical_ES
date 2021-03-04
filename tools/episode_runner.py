import gym
import random
import numpy as np
from gym.spaces import flatdim

class EpisodeRunner:

    def __init__(self, env_name: str, brain_class, brain_configuration: dict):

        self.env_name = env_name
        #env = gym.make(env_name)
        self.input_size = 22  # flatdim(env.observation_space)
        self.output_size = 2 #  flatdim(env.action_space)

        self.brain_class = brain_class
        self.brain_configuration = brain_configuration

        self.brain_state = brain_class.generate_brain_state(number_inputs=self.input_size,
                                                            number_outputs=self.output_size,
                                                            configuration=brain_configuration)

    def get_individual_size(self):
        return self.brain_class.get_individual_size(self.brain_state)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def save_brain_state(self, path):
        self.brain_class.save_brain_state(path, self.brain_state)

    def get_free_parameter_usage(self):
        return self.brain_class.get_free_parameter_usage(self.brain_state)

    def eval_fitness(self, evaluation):

        # Extract parameters, this list of lists is necessary since pool.map only accepts a single argument
        # See here: http://python.omics.wiki/multiprocessing_map/multiprocessing_partial_function_multiple_arguments
        individual = evaluation[0]
        env_seed = evaluation[1]
        number_of_rounds = evaluation[2]

        brain = self.brain_class(individual=individual, configuration=self.brain_configuration,
                                 brain_state=self.brain_state)

        fitness_total = 0

        screen_width = 500
        screen_height = 500

        # Radius of agent
        agent_radius = 10
        point_radius = 8

        number_points = 10

        for i in range(number_of_rounds):

            brain.reset()

            # Agent coordinates
            agent_position_x = 200
            agent_position_y = 200

            random.seed(env_seed)

            points = [(random.randrange(screen_width), random.randrange(screen_height)) for _ in range(number_points)]

            ob = list()
            ob.append(agent_position_x/screen_width)
            ob.append(agent_position_y/screen_height)
            for point in points:
                ob.append(point[0]/screen_width)
                ob.append(point[1]/screen_height)

            fitness_current = 0

            for _ in range(1000):

                action = brain.step(np.asarray(ob))

                agent_position_x += int(action[0] * 10)
                agent_position_y += int(action[1] * 10)

                agent_position_x = min(agent_position_x, screen_height - agent_radius)
                agent_position_x = max(agent_position_x, agent_radius)
                agent_position_y = min(agent_position_y, screen_width - agent_radius)
                agent_position_y = max(agent_position_y, agent_radius)

                ob[0] = agent_position_x/screen_width
                ob[1] = agent_position_y/screen_height

                points_new = []
                for point in points:
                    if (point[0] - agent_position_x) ** 2 + (point[1] - agent_position_y) ** 2 < (
                            point_radius + agent_radius) ** 2:
                        fitness_current += 1.0
                    else:
                        points_new.append(point)

                points = points_new

            fitness_total += fitness_current

        return fitness_total / number_of_rounds
