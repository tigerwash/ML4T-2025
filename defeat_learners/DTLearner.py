""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import warnings  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
class DTLearner(object):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		 	 	 			  		 			     			  	 
    your own correct DTLearner from Project 3.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		 	 	 			  		 			     			  	 
    :type leaf_size: int  		  	   		 	 	 			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	 	 			  		 			     			  	 
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	 	 			  		 			     			  	 
    :type verbose: bool  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    def __init__(self, leaf_size=1, verbose=False):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Constructor method  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        self.leaf_size = leaf_size
        self.tree = None
        self.verbose = verbose
  		  	   		 	 	 			  		 			     			  	 
    def author(self):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
        :rtype: str  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        return "lliao32"
  		  	   		 	 	 			  		 			     			  	 
    def add_evidence(self, data_x, data_y):  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
        Add training data to learner  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        :param data_x: A set of feature values used to train the learner  		  	   		 	 	 			  		 			     			  	 
        :type data_x: numpy.ndarray  		  	   		 	 	 			  		 			     			  	 
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	 	 			  		 			     			  	 
        :type data_y: numpy.ndarray  		  	   		 	 	 			  		 			     			  	 
        """  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
        data = np.concatenate((data_x, data_y[:, None]), axis=1)
        self.tree = self.build_tree(data)
        if self.verbose: print(self.tree)
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
    def build_tree(self, data):

        #  base case
        # Tree structure: [feature, split_val, left_child, right_child]
        if data.shape[0] <= self.leaf_size:
            return np.array([["Leaf", np.mean(data[:, -1]), np.nan, np.nan]])
        if np.unique(data[:, -1]).shape[0] == 1:
            return np.array([["Leaf", np.unique(data[:,-1])[0], np.nan, np.nan]])

        best_i, best_val = self.best_split(data)
        if best_i == -1:
            return np.array([["Leaf", np.mean(data[:, -1]), np.nan, np.nan]])
        #split data into left and right
        left_mask = data[:, best_i] <= best_val
        right_mask = ~left_mask

        # make a leaf, if split doesn't separate data,
        if np.all(left_mask) or np.all(~left_mask):
            return np.array([["Leaf", np.mean(data[:, -1]), np.nan, np.nan]])

        left_tree = self.build_tree(data[left_mask])
        right_tree = self.build_tree(data[right_mask])
        root = np.array([[best_i, best_val, 1, left_tree.shape[0] + 1]])
        return np.concatenate((root, left_tree, right_tree))

    def best_split(self, data):
        best_i = -1
        best_val = 0
        max_corr = -1
        # data_x = data[:, :-1]
        for i in range(data.shape[1] - 1):
            # if len(np.unique(data[:, i])) > 1:
            corr = np.corrcoef(data[:, i], data[:, -1])[0, 1]
            if not np.isnan(corr) and abs(corr) > max_corr:
                max_corr = abs(corr)
                best_i = i
                best_val = np.median(data[:, i])
        return best_i, best_val

if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")  		  	   		 	 	 			  		 			     			  	 
