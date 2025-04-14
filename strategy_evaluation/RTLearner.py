import numpy as np
from scipy import stats

class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.tree = None
        self.verbose = verbose
        np.random.seed(903648350)
        # np.random.seed(99999)

    def author(self):
        return "lliao32"

    def study_group(self):
        return "lliao32"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # construct data:
        # data = np.concatenate((data_x, data_y[:, None]), axis=1)
        data = np.concatenate((data_x, data_y), axis=1)
        self.tree = self.build_tree(data)
        if self.verbose: print(self.tree)

    def build_tree(self, data):
        # base case
        # Tree structure: [feature, split_val, left_child, right_child]
        if data.shape[0] <= self.leaf_size:
            # return np.array([["Leaf", np.mean(data[:, -1]), np.nan, np.nan]])
            return np.array([["Leaf", stats.mode(data[:, -1])[0][0], np.nan, np.nan]])
        if np.unique(data[:, -1]).shape[0] == 1:
            return np.array([["Leaf", np.unique(data[:,-1])[0], np.nan, np.nan]])

        best_i, best_val = self.best_split(data)
        if best_i == -1:
            # return np.array([["Leaf", np.mean(data[:, -1]), np.nan, np.nan]])
            return np.array([["Leaf", stats.mode(data[:, -1])[0][0], np.nan, np.nan]])
        # split data into left and right
        left_mask = data[:, best_i] <= best_val
        right_mask = ~left_mask

        # make a leaf, if split doesn't separate data,
        if np.all(left_mask) or np.all(~left_mask):
            return np.array([["Leaf", np.mean(data[:, -1]), np.nan, np.nan]])

        left_tree = self.build_tree(data[left_mask])
        right_tree = self.build_tree(data[right_mask])
        root = np.array([[best_i, best_val, 1, left_tree.shape[0] + 1]])
        return np.concatenate((root, left_tree, right_tree))

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.zeros(points.shape[0])
        for i, p in enumerate(points):
            node_idx = 0

            while self.tree[node_idx][0] != "Leaf":
                node = self.tree[node_idx]
                feature = int(float(node[0]))
                split_val = float(node[1])

                if p[feature] <= split_val:
                    node_idx += int(float(node[2]))  # Left path
                else:
                    node_idx += int(float(node[3]))  # Right path

            predictions[i] = float(self.tree[node_idx][1])
        return predictions

    def best_split(self, data):
        # random choice a feature and use its median as split value
        if data.shape[1] <= 1: return -1, 0
        best_i = np.random.randint(0, data.shape[1] - 1)
        best_val = np.median(data[:, best_i])
        return best_i, best_val

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")