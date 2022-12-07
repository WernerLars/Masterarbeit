from math import ceil
from random import randint
import numpy as np
from numpy import mean


class Q_Learning(object):
    def __init__(self):
        self.q_table = {
            "new_cluster": np.array([0])
        }
        self.model = {
            "new_cluster": [[0, ""]]
        }
        self.actions = [0]
        self.spikes = []
        self.clusters = []
        self.randomFeatures = [[1, 2, 3]]
        self.punishment_coefficient = 0.05
        self.alpha = 0.8
        self.gamma = 0.97
        self.planning_number = 20
        self.clusters_number = 0
        self.episode_number = self.computeEpisodeNumber(self.clusters_number)

    def set_q_value(self, state, action, value):
        self.q_table[state][action] = value

    def set_model_value(self, state, action, r, s):
        self.model[state][action] = [r, s]

    def reset_q_table(self):
        for state in self.q_table.keys():
            for action in range(0, len(self.q_table[state])):
                self.q_table[state][action] = 0

    def reset_model_table(self):
        for state in self.model.keys():
            for action in range(0, len(self.q_table[state])):
                self.model[state][action] = [0, ""]

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

        self.actions.append(state_length)
        self.clusters_number += 1
        self.computeEpisodeNumber(self.clusters_number)

    def normaliseFeatures(self, spike):
        features_normalised = np.zeros(len(spike))
        for i in range(0, len(spike)):
            features_normalised[i] = (spike[i]) / (max(spike) - min(spike))
        return features_normalised

    def computeEpisodeNumber(self, clusters_number):
        return ceil(100 * (clusters_number / 1.4))

    def selectRandomFeatures(self, action):
        random = [randint(0, len(self.randomFeatures[action-1])-1) for _ in range(0, 20)]
        feature_selection = []
        for i in range(0, 20):
            feature_selection.append(self.randomFeatures[action-1][random[i]])
        return feature_selection

    def computeReward(self, action, features_number, features_normalised):
        if action == 0:
            return -(self.punishment_coefficient*features_number)**2
        else:
            feature_sum = 0
            features_selected = self.selectRandomFeatures(action)
            for i in range(0, features_number):
                feature_sum += (features_normalised[i] - mean(features_selected))**2
            return -feature_sum

ql = Q_Learning()
print(ql.q_table)
print(ql.model)

ql.set_q_value("new_cluster", 0, 5)
print(ql.q_table)

ql.new_cluster()
print(ql.q_table)
print(ql.model)
ql.new_cluster()
print(ql.q_table)
print(ql.model)
ql.new_cluster()
print(ql.q_table)
print(ql.model)

s=[1,2,3,64,128]
s_no = ql.normaliseFeatures(s)

print(s)
print(s_no)




