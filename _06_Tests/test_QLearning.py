from unittest import TestCase

from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning


class TestQ_Learning(TestCase):

    def setUp(self):
        self.ql = Q_Learning()
        print(self.ql.q_table)
        print(self.ql.model)

    def test_set_q_value(self):
        self.ql.set_q_value("new_cluster", 0, 5)
        print(self.ql.q_table)

    def test_set_model_value(self):
        self.ql.set_model_value("new_cluster", 0, 3, "new_cluster")
        print(self.ql.model)

    def test_reset_q_table(self):
        self.ql.reset_q_table()
        print(self.ql.q_table)

    def test_reset_model_table(self):
        self.ql.reset_model_table()
        print(self.ql.model)

    def test_new_cluster(self):
        self.ql.new_cluster()
        print(self.ql.q_table)
        print(self.ql.model)
        print("Q_Table_Keys: ", self.ql.q_table.keys())
        print("Q_Table_Values: ", self.ql.q_table.values())
        print("Model_Keys: ", self.ql.model.keys())
        print("Model_Values: ", self.ql.model.values())
        print("Action List: ", self.ql.actions)

    def test_normalise_features(self):
        s = [1, 2, 3]
        s_normalised = self.ql.normaliseFeatures(s)
        print(s)
        print(s_normalised)

    def test_compute_episode_number(self):
        episode_number = self.ql.computeEpisodeNumber(self.ql.clusters_number)
        print(episode_number)

    def test_select_random_features(self):
        feature_selected = self.ql.selectRandomFeatures(1)
        print(feature_selected)

    def test_compute_reward(self):
        s = [1, 2, 3]
        s_normalised = self.ql.normaliseFeatures(s)

        reward = self.ql.computeReward(0, len(s_normalised), s_normalised)
        print("New Cluster Reward: ", reward)

        reward = self.ql.computeReward(1, len(s_normalised), s_normalised)
        print("Cluster 1 Reward: ", reward)
