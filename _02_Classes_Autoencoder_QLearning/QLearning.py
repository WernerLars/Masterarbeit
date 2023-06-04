import dis
import logging
import random
from math import ceil
from random import randint, choice
import numpy as np
import pandas as pd
from numpy.random import uniform


class Q_Learning(object):
    def __init__(self, parameter_logger, number_of_features, normalise=False, punishment_coefficient=0.6,
                 alpha=0.8, epsilon=0.01, gamma=0.97, episode_number=0,
                 episode_number_coefficient=1.4, random_features_number=20, planning_number=20,
                 maxRandomFeatures=60):
        self.parameter_logger = parameter_logger
        self.number_of_features = number_of_features
        self.parameter_logger.info("---Q Learning Parameters---")
        self.q_table = {
            "new_cluster": [0]
        }
        self.model = {
            "new_cluster": [[0, ""]]
        }
        self.randomFeatures = {}
        self.features = [[] for _ in range(self.number_of_features)]
        self.states = ["new_cluster"]
        self.actions = [0]
        self.clusters_number = 0
        self.spikes = []
        self.clusters = []
        self.normalise = normalise
        self.parameter_logger.info(f"Normalisation: {self.normalise}")
        self.punishment_coefficient = punishment_coefficient
        self.parameter_logger.info(f"Punishment Coefficient: {self.punishment_coefficient}")
        self.alpha = alpha
        self.parameter_logger.info(f"Alpha: {self.alpha}")
        self.epsilon = epsilon
        self.parameter_logger.info(f"Epsilon: {self.epsilon}")
        self.gamma = gamma
        self.parameter_logger.info(f"Gamma: {self.gamma}")
        self.episode_number = episode_number
        self.parameter_logger.info(f"Initial Episode Number: {self.episode_number}")
        self.episode_number_coefficient = episode_number_coefficient
        self.parameter_logger.info(f"Episode Number Coefficient: {self.episode_number_coefficient}")
        self.random_features_number = random_features_number
        self.parameter_logger.info(f"Number of Random Features: {self.random_features_number}")
        self.planning_number = planning_number
        self.parameter_logger.info(f"Planning Number: {self.planning_number}")
        self.maxRandomFeatures = maxRandomFeatures
        self.parameter_logger.info(f"Max Random Features: {self.maxRandomFeatures}")

    def reset_q_learning(self):
        """
            resets q-learning to originally state
            all prior information of parameters of q learning are lost
        """

        self.q_table = {
            "new_cluster": [0]
        }
        self.model = {
            "new_cluster": [[0, ""]]
        }
        self.randomFeatures = {}
        self.features = [[] for _ in range(self.number_of_features)]
        self.states = ["new_cluster"]
        self.actions = [0]
        self.clusters_number = 0
        self.spikes = []
        self.clusters = []

    def set_q_value(self, state, action, value):
        """
            Sets Value into Q-Table Dictionary in list position of action
        """

        self.q_table[state][action] = value

    def set_model_value(self, state, action, r, s_new):
        """
            Sets Reward and new state into Model Dictionary in list position of action
        """

        self.model[state][action] = [r, s_new]

    def reset_q_table(self):
        """
            Resets all Values of Lists in Q-Table
        """

        for state in self.q_table.keys():
            for action in range(0, len(self.q_table[state])):
                self.q_table[state][action] = 0

    def reset_model_table(self):
        """
            Resets all Rewards and new states of Lists in Model
        """

        for state in self.model.keys():
            for action in range(0, len(self.q_table[state])):
                self.model[state][action] = [0, ""]

    def reset_spikes_clusters(self):
        """
            Resets spike and cluster lists
        """

        self.spikes = []
        self.clusters = []

    def print_q_table(self):
        """
            prints q table as a table (dataframe)
        """

        for state in self.q_table:
            self.q_table[state] = np.round(self.q_table[state], 2)
        df = pd.DataFrame.from_dict(self.q_table, orient="index")
        self.parameter_logger.info(df)
        self.parameter_logger.info(df.to_latex())

    def print_model(self):
        """
            prints model as a table (dataframe)
        """

        for state in self.model:
            for action, _ in enumerate(self.model[state]):
                self.model[state][action][0] = np.round(self.model[state][action][0], 2)
        df = pd.DataFrame.from_dict(self.model, orient="index")
        self.parameter_logger.info(df)
        self.parameter_logger.info(df.to_latex())

    def new_cluster(self):
        """
            in q-table and model a new key and action is added
            for new key a list are added in q-table model
            for new action every list entry in q table and model ist expanded with zero or rather reward+new state
            parameters are adjusted
        """

        number_of_states = len(self.q_table.keys())
        new_key = f"c{str(number_of_states)}"

        for state in self.q_table.keys():
            self.q_table[state] = np.append(self.q_table[state], 0)
        self.q_table[new_key] = np.zeros(number_of_states + 1)

        for state in self.model.keys():
            self.model[state].append([0, ""])

        self.model[new_key] = [[0, ""]]
        for _ in range(0, number_of_states):
            self.model[new_key].append([0, ""])

        # new state and action are appended to list, randomFeatures gets new list
        self.states.append(new_key)
        self.actions.append(number_of_states)
        self.randomFeatures[new_key] = [[] for _ in range(self.number_of_features)]
        self.clusters_number += 1

        # episode number is changed after adding a new cluster
        self.episode_number = self.compute_episode_number()
        self.parameter_logger.info(f"New Episode Number: {self.episode_number}")

    def add_to_feature_set(self, spike):
        """
            every dimension of spike feature is added to a list in features
        """

        for i in range(0, self.number_of_features):
            self.features[i].append(spike[i])

    def normalise_features(self, spike):
        """
            every dimension of spike features are normalised with formula (feature/max-min)
            :return: normalised features
        """

        features_normalised = np.zeros(len(spike))
        for i in range(0, self.number_of_features):
            features_normalised[i] = (spike[i]) / (max(self.features[i]) - min(self.features[i]))
        return features_normalised

    def compute_episode_number(self):
        """
            computes new episode number for q-learning with formula
            :return: new episode number
        """

        return ceil(100 * (self.clusters_number / self.episode_number_coefficient))

    def epsilon_greedy_method(self, state):
        """
            performs epsilon greedy method in a state
            :return: random action or max action (dependent on random p and epsilon)
        """

        q_values = self.q_table[state]
        p = uniform(0, 1)
        if p < self.epsilon:
            return randint(0, max(self.actions))
        else:
            return np.argmax(q_values)

    def select_random_features(self, action, i):
        """
            computes random indexes for a feature dimension i in length of random_features_number
            adds random features to a list by using random indexes
        :return: list of random features depended on length of random_features_number
        """

        random_indexes = [randint(0, len(self.randomFeatures[f"c{action}"][i]) - 1)
                          for _ in range(0, self.random_features_number)]
        feature_selection = []
        for j in range(0, self.random_features_number):
            feature_selection.append(self.randomFeatures[f"c{action}"][i][random_indexes[j]])
        return feature_selection

    def compute_reward(self, action, features):
        """
            computes reward for action and features
            if action is not new_cluster (0) random features are selected in every dimension
                with select_random_features method and mean is computed on them
        :return: reward with formulas dependent of action
        """

        if action == 0:
            return -(self.punishment_coefficient * self.number_of_features) ** 2
        else:
            feature_sum = 0
            for i in range(0, self.number_of_features):
                features_selected = self.select_random_features(action, i)
                feature_sum += (features[i] - np.mean(features_selected)) ** 2
            return -feature_sum

    def add_spike(self, spike, s):
        """
            computes cluster based on max action of state s
            spike and cluster are added to corresponding lists
            add every dimension of spike features to randomFeatures based on max action
            if max action is 0, then new_cluster function is called
            if maxRandomFeatures is reached, the first element of dimension is deleted and new feature is appended
        :return: cluster label (used for templates in variant 5)
        """

        max_action = np.argmax(self.q_table[s])
        if max_action == 0:
            self.new_cluster()

            # because of new_cluster, clusters_number is incremented
            # corresponding cluster is number of clusters - 1, because of new_cluster label (not in dictionary)
            cluster = self.clusters_number - 1
            self.spikes.append(spike)
            self.clusters.append(cluster)
            for i in range(0, self.number_of_features):
                self.randomFeatures[f"c{cluster + 1}"][i] = np.append(self.randomFeatures[f"c{cluster + 1}"][i],
                                                                      spike[i])
        else:
            # corresponding cluster is max_action-1, because of new_cluster label (not in dictionary)
            cluster = max_action - 1
            self.spikes.append(spike)
            self.clusters.append(cluster)
            for i in range(0, self.number_of_features):

                # limiting number of Random Features by maxRandomFeatures
                if len(self.randomFeatures[f"c{cluster + 1}"][i]) >= self.maxRandomFeatures:
                    self.randomFeatures[f"c{cluster + 1}"][i] = np.delete(self.randomFeatures[f"c{cluster + 1}"][i], 0)

                self.randomFeatures[f"c{cluster + 1}"][i] = np.append(self.randomFeatures[f"c{cluster + 1}"][i],
                                                                      spike[i])
        return cluster

    def dyna_q_algorithm(self, spike):
        """
            performing dyna-q-learning on a spike
            normalises features if normalise boolean is true
            choice at start to set up a random state
            state_action_taken to remember visited states and actions for planning
        :return: cluster label (used for templates in variant 5)
        """

        # Normalises features of spike
        if self.normalise:
            self.add_to_feature_set(spike)
            features = self.normalise_features(spike)
        else:
            features = spike

        # for every new spike the q-table and model entries are reset, not the whole q-learning
        self.reset_q_table()
        self.reset_model_table()

        # initial set up of parameters
        s = choice(self.states)
        counter = 0
        state_action_taken = []

        while counter <= self.episode_number:
            a = self.epsilon_greedy_method(s)
            r = self.compute_reward(a, features)
            s_new = self.states[a]
            max_action = np.max(self.q_table[s_new])
            value = self.q_table[s][a] + self.alpha * (r + (self.gamma * max_action) - self.q_table[s][a])
            self.set_q_value(s, a, value)
            self.set_model_value(s, a, r, s_new)

            # Add state and action to state_action_taken if not already in it
            if [s, a] not in state_action_taken:
                state_action_taken.append([s, a])

            for i in range(0, self.planning_number):

                # choose random state action pair in state_action_taken
                state_action_pair = choice(state_action_taken)
                s_rand = state_action_pair[0]
                a_rand = state_action_pair[1]

                r_m, s_m = self.model[s_rand][a_rand]
                max_action = np.max(self.q_table[s_m])
                value = self.q_table[s_rand][a_rand] + self.alpha * (
                        r_m + (self.gamma * max_action) - self.q_table[s_rand][a_rand])
                self.set_q_value(s_rand, a_rand, value)

            s = s_new
            counter += 1

        cluster = self.add_spike(features, s)
        return cluster

    def print_byte_code(self):
        """
            dis for printing byte Code of a function
        """
        dis.dis(self.compute_reward)
        # dis.dis(self.dynaQAlgorithm)

# Code for printing out ByteCode of interested Functions in Q-Learning
# logger = logging.getLogger("Test Logger")
# logger.setLevel(logging.INFO)
# ql = Q_Learning(logger, 2)
# ql.print_byte_code()
