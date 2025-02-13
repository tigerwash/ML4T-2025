""""""

import pandas as pd
from matplotlib import pyplot as plt

"""  		  	   		 	 	 			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import math  		  	   		 	 	 			  		 			     			  	 
import sys  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import LinRegLearner as lrl
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import time


def author(self):
    return "lliao32"
def experiment_1(train_x, train_y, test_x, test_y):
    num_leafs = 50
    in_rsmes_list = []
    out_rsmes_list = []

    for size in range(1, num_leafs+1):
        learner = dtl.DTLearner(leaf_size=size, verbose = False)
        learner.add_evidence(train_x, train_y)

        in_sample_y = learner.query(train_x)
        rsmes_in_sample = math.sqrt(((train_y - in_sample_y)**2).sum()/train_y.shape[0])
        in_rsmes_list.append(rsmes_in_sample)

        out_sample_y = learner.query(test_x)
        rsmes_out_sample = math.sqrt(((test_y - out_sample_y)**2).sum()/test_y.shape[0])
        out_rsmes_list.append(rsmes_out_sample)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_leafs + 1), in_rsmes_list, label='In Sample')
    plt.plot(range(1, num_leafs + 1), out_rsmes_list, label='Out Sample')
    plt.title("Influence of Leaf Size on Overfitting Analysis", fontsize=12)
    plt.xlabel("Leaf Size", fontsize=10)
    plt.ylabel("RMSE", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.insert(np.arange(5, num_leafs + 1, step=5), 0, 1))
    # plt.show()
    plt.savefig("experiment_1.png", dpi=300, bbox_inches='tight')
    plt.clf()

def experiment_2(train_x, train_y, test_x, test_y):
    num_leafs = 50
    in_rsmes_list = []
    out_rsmes_list = []

    for size in range(1, num_leafs+1):
        learner = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": size}, bags=20, boost=False, verbose=False)
        learner.add_evidence(train_x, train_y)

        in_sample_y = learner.query(train_x)
        rsmes_in_sample = math.sqrt(((train_y - in_sample_y)**2).sum()/train_y.shape[0])
        in_rsmes_list.append(rsmes_in_sample)

        out_sample_y = learner.query(test_x)
        rsmes_out_sample = math.sqrt(((test_y - out_sample_y)**2).sum()/test_y.shape[0])
        out_rsmes_list.append(rsmes_out_sample)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_leafs + 1), in_rsmes_list, label='In Sample')
    plt.plot(range(1, num_leafs + 1), out_rsmes_list, label='Out Sample')
    plt.title("Influence of Bagging on Overfitting Analysis", fontsize=12)
    plt.xlabel("Leaf Size", fontsize=10)
    plt.ylabel("RMSE", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.insert(np.arange(5, num_leafs + 1, step=5), 0, 1))
    # plt.show()
    plt.savefig("experiment_2.png", dpi=300, bbox_inches='tight')
    plt.clf()

# calculate the median absolute error (MAE) of the predictions
def experiment_3_1(train_x, train_y, test_x, test_y):
    num_leafs = 50
    dt_medae_list = []
    dt_ev_list = []
    rt_medae_list = []
    rt_ev_list = []

    for size in range(1,num_leafs+1):
        dt_learner = dtl.DTLearner(leaf_size=size, verbose=False)
        dt_learner.add_evidence(train_x, train_y)

        rt_learner = rtl.RTLearner(leaf_size=size, verbose=False)
        rt_learner.add_evidence(train_x, train_y)

        dt_pred = dt_learner.query(test_x)
        dt_medae = np.median(np.abs(test_y - dt_pred))
        dt_medae_list.append(dt_medae)
        dt_ev_list.append(np.std(test_y - dt_pred))

        rt_pred = rt_learner.query(test_x)
        rt_medae = np.median(np.abs(test_y - rt_pred))
        rt_medae_list.append(rt_medae)
        rt_ev_list.append(np.std(test_y - rt_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_leafs + 1), dt_medae_list, label='DTLearner')
    plt.plot(range(1, num_leafs + 1), rt_medae_list, label='RTLearner')
    plt.title("Influence of Leaf Size on Median Absolute Error", fontsize=12)
    plt.xlabel("Leaf Size", fontsize=10)
    plt.ylabel("Median Absolute Error", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.insert(np.arange(5, num_leafs + 1, step=5), 0, 1))
    # plt.show()
    plt.savefig("experiment_3_1.png", dpi=300, bbox_inches='tight')
    plt.clf()

#  anaylize the training time of the two learners
def experiment_3_2(train_x, train_y):
    data_size = train_x.shape[0]
    dt_time_list = []
    rt_time_list = []

    for size in range(1, data_size+1):
        dt_learner = dtl.DTLearner(leaf_size=10, verbose=False)
        rt_learner = rtl.RTLearner(leaf_size=10, verbose=False)

        train_x_trimed = train_x[:size]
        train_y_trimed = train_y[:size]
        dt_start = time.time()
        dt_learner.add_evidence(train_x_trimed, train_y_trimed)
        dt_end = time.time()
        dt_time_list.append(dt_end - dt_start)

        rt_start = time.time()
        rt_learner.add_evidence(train_x_trimed, train_y_trimed)
        rt_end = time.time()
        rt_time_list.append(rt_end - rt_start)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, data_size + 1), dt_time_list, label='DTLearner')
    plt.plot(range(1, data_size + 1), rt_time_list, label='RTLearner')
    plt.title("Influence of Data Size on Training Time", fontsize=12)
    plt.xlabel("Data Size", fontsize=10)
    plt.ylabel("Training Time", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.insert(np.arange(100, data_size + 1, step=100), 0, 1))
    # plt.show()
    plt.savefig("experiment_3_2.png", dpi=300, bbox_inches='tight')
    plt.clf()



if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 2:  		  	   		 	 	 			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 			  		 			     			  	 
        sys.exit(1)  		  	   		 	 	 			  		 			     			  	 
    inf = open(sys.argv[1])  		  	   		 	 	 			  		 			     			  	 
    # data = np.array(
    #     [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    # )
    data = np.array([list(map(float,s.strip().split(',')[1:])) for s in inf.readlines()[1:]])
  		  	   		 	 	 			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		 	 	 			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		 	 	 			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # separate out training and testing data  		  	   		 	 	 			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		 	 	 			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		 	 	 			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		 	 	 			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		 	 	 			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # create a learner and train it  		  	   		 	 	 			  		 			     			  	 
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner = dtl.DTLearner(leaf_size=1, verbose=True) # create DTLearner
    learner = rtl.RTLearner(leaf_size=1, verbose=True) # create DTLearner



    learner.add_evidence(train_x, train_y)  # train it  		  	   		 	 	 			  		 			     			  	 
    print(learner.author())  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # evaluate in sample  		  	   		 	 	 			  		 			     			  	 
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	 	 			  		 			     			  	 
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print("In sample results")  		  	   		 	 	 			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		 	 	 			  		 			     			  	 
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	 	 			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # evaluate out of sample  		  	   		 	 	 			  		 			     			  	 
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	 	 			  		 			     			  	 
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print("Out of sample results")  		  	   		 	 	 			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		 	 	 			  		 			     			  	 
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	 	 			  		 			     			  	 
    print(f"corr: {c[0,1]}")

    experiment_1(train_x, train_y, test_x, test_y)
    experiment_2(train_x, train_y, test_x, test_y)
    experiment_3_1(train_x, train_y, test_x, test_y)
    experiment_3_2(train_x, train_y)

