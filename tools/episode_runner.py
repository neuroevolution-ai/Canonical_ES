import gym
import numpy as np
from gym.spaces import flatdim
from environments.collect_points import *


class EpisodeRunner:

    def __init__(self, env_name: str, brain_class, brain_configuration: dict):

        self.env_name = env_name
        env = CollectPointsEnv(0)
        self.input_size = env.get_number_inputs()  # flatdim(env.observation_space)
        self.output_size = env.get_number_outputs() # flatdim(env.action_space)

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

        for i in range(number_of_rounds):

            env = CollectPointsEnv(env_seed)
            ob = env.reset()
            brain.reset()

            fitness_current = 0
            done = False

            while not done:
                action = brain.step(ob)
                ob, rew, done, info = env.step(action)
                fitness_current += rew

            fitness_total += fitness_current

        return fitness_total / number_of_rounds
