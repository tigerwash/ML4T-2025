import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.learners = []
        self.bags = 20
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in range(self.bags )]
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        predictions = np.zeros((points.shape[0], self.bags))
        for i in range(len(self.learners)):
            predictions[:, i] = self.learners[i].query(points)
        return np.mean(predictions, axis=1)
    def author(self): return "lliao32"