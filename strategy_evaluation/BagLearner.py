import numpy as np
from scipy import stats
class BagLearner(object):
    def __init__(self, learner, kwargs={}, bags = 20, boost = False, verbose=False):
        """
        Constructor method
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        self.learners = []
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

        np.random.seed(903648350)
        # np.random.seed(99999)
    def author(self):
        return "lliao32"
    def study_group(self):
        return "lliao32"
    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        """
        for learner in self.learners:
            index = np.random.choice(data_x.shape[0], data_x.shape[0])
            learner.add_evidence(data_x[index], data_y[index])
        if self.verbose:
            print("Current Learners: \n ", self.learners)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.zeros((points.shape[0], self.bags))
        for i in range(len(self.learners)):
            predictions[:, i] = self.learners[i].query(points)

        # For classification, use mode instead of mean
        result = np.zeros(points.shape[0])

        for i in range(points.shape[0]):
            # Find the most common value (mode) for each row
            values, counts = np.unique(predictions[i, :], return_counts=True)
            result[i] = values[np.argmax(counts)]

        # return np.mean(predictions, axis=1)
        return result


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")