from math import ceil
from random import randint, choice
import numpy as np
from numpy.random import uniform


class Q_Learning(object):
    def __init__(self, parameter_logger, punishment_coefficient=0.27, alpha=0.8, epsilon=0.01, gamma=0.97, episode_number=0,
                 episode_number_coefficient=1.4, random_features_number=20, planning_number=20):
        self.parameter_logger = parameter_logger
        self.parameter_logger.info("---Q Learning Parameters---")
        self.q_table = {
            "new_cluster": [0]
        }
        self.model = {
            "new_cluster": [[0, ""]]
        }
        self.randomFeatures = {}
        self.features = [[], []]
        self.states = ["new_cluster"]
        self.actions = [0]
        self.clusters_number = 0
        self.spikes = []
        self.clusters = []
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

    def reset_q_learning(self):
        self.q_table = {
            "new_cluster": [0]
        }
        self.model = {
            "new_cluster": [[0, ""]]
        }
        self.randomFeatures = {}
        self.features = [[], []]
        self.states = ["new_cluster"]
        self.actions = [0]
        self.clusters_number = 0
        self.spikes = []
        self.clusters = []

    def set_q_value(self, state, action, value):
        self.q_table[state][action] = value

    def set_model_value(self, state, action, r, s_new):
        self.model[state][action] = [r, s_new]

    def reset_q_table(self):
        for state in self.q_table.keys():
            for action in range(0, len(self.q_table[state])):
                self.q_table[state][action] = 0

    def reset_model_table(self):
        for state in self.model.keys():
            for action in range(0, len(self.q_table[state])):
                self.model[state][action] = [0, ""]

    def reset_spikes_clusters(self):
        self.spikes = []
        self.clusters = []

    def new_cluster(self):
        state_length = len(self.q_table.keys())
        new_key = f"c{str(state_length)}"

        for state in self.q_table.keys():
            self.q_table[state] = np.append(self.q_table[state], 0)
        self.q_table[new_key] = np.zeros(state_length + 1)

        for state in self.model.keys():
            self.model[state].append([0, ""])

        self.model[new_key] = [[0, ""]]
        for _ in range(0, state_length):
            self.model[new_key].append([0, ""])

        self.states.append(new_key)
        self.actions.append(state_length)
        self.randomFeatures[new_key] = [[], []]
        self.clusters_number += 1
        self.episode_number = self.computeEpisodeNumber()
        self.parameter_logger.info(f"New Episode Number: {self.episode_number}")

    def addToFeatureSet(self, spike):
        self.features[0].append(spike[0])
        self.features[1].append(spike[1])

    def normaliseFeatures(self, spike):
        features_normalised = np.zeros(len(spike))
        for i in range(0, len(spike)):
            features_normalised[i] = (spike[i]) / (max(self.features[i]) - min(self.features[i]))
        return features_normalised

    def computeEpisodeNumber(self):
        return ceil(100 * (self.clusters_number / self.episode_number_coefficient))

    def epsilonGreedy(self, state):
        q_values = self.q_table[state]
        p = uniform(0, 1)
        if p < self.epsilon:
            return randint(0, max(self.actions))
        else:
            return np.argmax(q_values)

    def selectRandomFeatures(self, action, i):
        random_indexes = [randint(0, len(self.randomFeatures[f"c{action}"][i]) - 1)
                          for _ in range(0, self.random_features_number)]
        feature_selection = []
        for j in range(0, self.random_features_number):
            feature_selection.append(self.randomFeatures[f"c{action}"][i][random_indexes[j]])
        return feature_selection

    def computeReward(self, action, features_numbers, features_normalised):
        if action == 0:
            return -(self.punishment_coefficient * features_numbers) ** 2
        else:
            feature_sum = 0
            for i in range(0, features_numbers):
                features_selected = self.selectRandomFeatures(action, i)
                feature_sum += (features_normalised[i] - np.mean(features_selected)) ** 2
            return -feature_sum

    def addSpike(self, spike, s):
        max_action = np.argmax(self.q_table[s])
        if max_action == 0:
            self.new_cluster()
            cluster = self.clusters_number - 1
            self.spikes.append(spike)
            self.clusters.append(cluster)
            for i in range(0, len(spike)):
                self.randomFeatures[f"c{cluster + 1}"][i] = np.append(self.randomFeatures[f"c{cluster + 1}"][i],
                                                                      spike[i])
        else:
            cluster = max_action - 1
            self.spikes.append(spike)
            self.clusters.append(cluster)
            for i in range(0, len(spike)):
                self.randomFeatures[f"c{cluster + 1}"][i] = np.append(self.randomFeatures[f"c{cluster + 1}"][i],
                                                                      spike[i])
        return cluster

    def dynaQAlgorithm(self, spike):
        self.addToFeatureSet(spike)
        feature_normalised = self.normaliseFeatures(spike)
        self.reset_q_table()
        self.reset_model_table()
        s = choice(self.states)
        counter = 0
        state_action_taken = []

        while counter <= self.episode_number:
            a = self.epsilonGreedy(s)
            r = self.computeReward(a, len(feature_normalised), feature_normalised)
            s_new = self.states[a]
            max_action = np.max(self.q_table[s_new])
            value = self.q_table[s][a] + self.alpha * (r + (self.gamma * max_action) - self.q_table[s][a])
            self.set_q_value(s, a, value)
            self.set_model_value(s, a, r, s_new)

            if [s, a] not in state_action_taken:
                state_action_taken.append([s, a])

            for i in range(0, self.planning_number):
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

        cluster = self.addSpike(feature_normalised, s)
        return cluster
