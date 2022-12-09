from unittest import TestCase

from _02_Classes_Autoencoder_QLearning.QLearning import Q_Learning


class TestQ_Learning(TestCase):

    def setUp(self):
        self.ql = Q_Learning()

    def test_set_q_value(self):
        print(f"Set_Q_Value_Before: {self.ql.q_table}")
        self.ql.set_q_value("new_cluster", 0, 5)
        self.assertEqual(self.ql.q_table["new_cluster"][0], 5)
        print(f"Set_Q_Value_After: {self.ql.q_table}")

    def test_set_model_value(self):
        print(f"Set_Model_Values_Before: {self.ql.model}")
        self.ql.set_model_value("new_cluster", 0, 3, "new_cluster")
        self.assertEqual(self.ql.model["new_cluster"][0], [3, "new_cluster"])
        print(f"Set_Model_Values_After: {self.ql.model}")

    def test_reset_q_table(self):
        self.ql.set_q_value("new_cluster", 0, 5)
        print(f"Reset_Q_Table_Before: {self.ql.q_table}")
        self.ql.reset_q_table()
        self.assertEqual(self.ql.q_table["new_cluster"][0], 0)
        print(f"Reset_Q_Table_After: {self.ql.q_table}")

    def test_reset_model_table(self):
        self.ql.set_model_value("new_cluster", 0, 3, "new_cluster")
        print(f"Reset_Model_Before: {self.ql.model}")
        self.ql.reset_model_table()
        self.assertEqual(self.ql.model["new_cluster"][0], [0, ""])
        print(f"Reset_Model_After: {self.ql.model}")

    def test_new_cluster(self):
        self.assertEqual(self.ql.clusters_number, 0)
        self.assertEqual(len(self.ql.states), 1)
        self.assertEqual(len(self.ql.q_table.keys()), 1)
        self.assertEqual(len(self.ql.q_table["new_cluster"]), 1)
        self.assertEqual(len(self.ql.model.keys()), 1)
        self.assertEqual(len(self.ql.model["new_cluster"]), 1)

        print(f"Clusters_Number_Before: {self.ql.clusters_number}")
        print(f"Episode_Number_Before: {self.ql.episode_number}")
        print(f"States_List_Before: {self.ql.states}")
        print(f"Action_List_Before: {self.ql.actions}")
        print(f"Q_Table_Before: {self.ql.q_table}")
        print(f"Q_Table_Keys_Before: {self.ql.q_table.keys()}")
        print(f"Q_Table_Values_Before: {self.ql.q_table.values()}")
        print(f"Model_Before: {self.ql.model}")
        print(f"Model_Keys_Before: {self.ql.model.keys()}")
        print(f"Model_Values_Before: {self.ql.model.values()}")

        self.ql.new_cluster()

        self.assertEqual(self.ql.clusters_number, 1)
        self.assertEqual(self.ql.computeEpisodeNumber(), self.ql.episode_number)
        self.assertEqual(len(self.ql.states), 2)
        self.assertEqual(len(self.ql.q_table.keys()), 2)
        self.assertEqual(len(self.ql.q_table["new_cluster"]), 2)
        self.assertEqual(len(self.ql.q_table["c1"]), 2)
        self.assertEqual(len(self.ql.model["new_cluster"]), 2)
        self.assertEqual(len(self.ql.model["c1"]), 2)

        print(f"Clusters_Number_After: {self.ql.clusters_number}")
        print(f"Episode_Number_After: {self.ql.episode_number}")
        print(f"States_List_After: {self.ql.states}")
        print(f"Action_List_After: {self.ql.actions}")
        print(f"Q_Table_After: {self.ql.q_table}")
        print(f"Q_Table_Keys_After: {self.ql.q_table.keys()}")
        print(f"Q_Table_Values_After: {self.ql.q_table.values()}")
        print(f"Model_After: {self.ql.model}")
        print(f"Model_Keys_After: {self.ql.model.keys()}")
        print(f"Model_Values_After: {self.ql.model.values()}")

    def test_normalise_features(self):
        s = [1, 2, 3]
        s_normalised = self.ql.normaliseFeatures(s)
        print(s)
        print(s_normalised)

    def test_compute_episode_number(self):
        print(f"Old_Episode_Number: {self.ql.episode_number}")
        self.ql.new_cluster()
        self.assertEqual(self.ql.computeEpisodeNumber(), self.ql.episode_number)
        print(f"New_Episode_Number: {self.ql.episode_number}")

    def test_select_random_features(self):
        feature_selected = self.ql.selectRandomFeatures(1)
        print(feature_selected)

    def test_compute_reward(self):
        s = [1, 2, 3]
        s_normalised = self.ql.normaliseFeatures(s)

        reward = self.ql.computeReward(0, len(s_normalised), s_normalised)
        print("New Cluster Reward: ", reward)

        self.ql.randomFeatures.append(s_normalised)
        reward = self.ql.computeReward(1, len(s_normalised), s_normalised)
        print("Cluster 1 Reward: ", reward)
