""""""
import matplotlib.pyplot as plt

"""Assess a betting strategy.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
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
  		  	   		 	 	 			  		 			     			  	 
Student Name: Longtai Liao 		  	   		 	 	 			  		 			     			  	 
GT User ID: lliao32  		  	   		 	 	 			  		 			     			  	 
GT ID: 903648350	  	   		 	 	 			  		 			     			  	 
"""

import numpy as np


def author():
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: str  		  	   		 	 	 			  		 			     			  	 
    """
    return "lliao32"  # replace tb34 with your Georgia Tech username.

def study_group():
    return "lliao32"
def gtid():
    """  		  	   		 	 	 			  		 			     			  	 
    :return: The GT ID of the student  		  	   		 	 	 			  		 			     			  	 
    :rtype: int  		  	   		 	 	 			  		 			     			  	 
    """
    return 903648350  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		 	 	 			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		 	 	 			  		 			     			  	 
    :type win_prob: float  		  	   		 	 	 			  		 			     			  	 
    :return: The result of the spin.  		  	   		 	 	 			  		 			     			  	 
    :rtype: bool  		  	   		 	 	 			  		 			     			  	 
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def martingale(win_prob):
    spin = 0
    winnings = 0
    winning_map = np.zeros(1001)
    winning_map.fill(80)
    # code is based on Professor Balch’s actual betting strategy: https://gatech.instructure.com/courses/430880/assignments/1952092
    while winnings < 80:
        bet = 1
        won = False;
        while not won:
            if spin > 1000: return winning_map
            winning_map[spin] = winnings
            won = get_spin_result(win_prob)
            if won:
                winnings += bet
            else:
                winnings -= bet
                bet *= 2
            spin += 1

    return winning_map


def martingale_realistic(win_prob):
    spin = 0
    winnings = 0
    winning_map = np.zeros(1001)
    winning_map.fill(80)
    while winnings < 80:
        bet = 1
        won = False;
        while not won:
            if spin > 1000: return winning_map
            winning_map[spin] = winnings
            won = get_spin_result(win_prob)
            if won:
                winnings += bet
            else:
                winnings -= bet
                bet = min(winnings + 256, bet * 2)
                if winnings == -256:
                    winning_map[spin:] = -256
                    return winning_map

            spin += 1

    return winning_map


def build_canvas():
    plt.axis([0, 300, -256, 100])
    plt.xlabel("Spin numbers")
    plt.ylabel("Winnings")

# Experiment 1, simulator 10 episodes and track the winnings
def experiment_1_figure_1(win_prob):
    build_canvas()
    i = 0
    while i < 10:
        cur_episode_record = martingale(win_prob)
        plt.plot(cur_episode_record)
        i += 1
    plt.legend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.title("E1F1: Winning over 10 Episodes")
    # plt.show()
    plt.savefig("f1.png")
    plt.clf()


def experiment_1_figure_2(win_prob):
    build_canvas()
    i = 0
    val_list = []
    while i < 1000:
        cur_episode_record = martingale(win_prob)
        # create matrix for 1000 episode and its values
        val_list.append(cur_episode_record)
        i += 1
    np_list = np.array(val_list)
    mean = np.mean(np_list, axis=0)
    stand_dev = np.std(np_list, axis=0)
    mean_mins_std = mean - stand_dev
    mean_add_std = mean + stand_dev

    plt.title("E1F2: Mean winnings of 1000 Episodes")
    plt.plot(mean_add_std, 'r--', alpha=0.5, label="Mean + standard_deviation")
    plt.plot(mean_mins_std, 'g--', alpha=0.5, label="Mean - standard_deviation")
    plt.plot(mean, 'b-', label="Mean")
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig("f2.png")
    plt.clf()


def experiment_1_figure_3(win_prob):
    build_canvas()
    i = 0
    val_list = []
    while i < 1000:
        cur_episode_record = martingale(win_prob)
        # create matrix for 1000 episode and its values
        val_list.append(cur_episode_record)
        i += 1
    np_list = np.array(val_list)
    median = np.median(np_list, axis=0)
    stand_dev = np.std(np_list, axis=0)
    median_mins_std = median - stand_dev
    median_add_std = median + stand_dev

    plt.title("E1F3: Median winnings of 1000 Episodes")
    plt.plot(median_add_std, 'r--', alpha=0.5, label="median + standard_deviation")
    plt.plot(median_mins_std, 'g--', alpha=0.5, label="median - standard_deviation")
    plt.plot(median, 'b-', label="median")
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig("f3.png")
    plt.clf()


def experiment_2_figure_4(win_prob):
    build_canvas()
    i = 0
    val_list = []
    while i < 1000:
        cur_episode_record = martingale_realistic(win_prob)
        # create matrix for 1000 episode and its values
        val_list.append(cur_episode_record)
        i += 1
    np_list = np.array(val_list)
    mean = np.mean(np_list, axis=0)
    stand_dev = np.std(np_list, axis=0)
    mean_mins_std = mean - stand_dev
    mean_add_std = mean + stand_dev

    plt.title("E2F4: Mean winnings of 1000 Episodes with limited bankroll")
    plt.plot(mean_add_std, 'r--', alpha=0.5, label="Mean + standard_deviation")
    plt.plot(mean_mins_std, 'g--', alpha=0.5, label="Mean - standard_deviation")
    plt.plot(mean, 'b-', label="Mean")

    # Add annotation for spin 300 value
    final_value = mean[300]
    plt.plot(300, final_value, 'ro')
    plt.annotate(f'{final_value:.1f}',
                 xy=(300, final_value),
                 xytext=(10, 10),
                 textcoords='offset points')

    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig("f4.png")
    plt.clf()


def experiment_2_figure_5(win_prob):
    build_canvas()
    i = 0
    val_list = []
    while i < 1000:
        cur_episode_record = martingale_realistic(win_prob)
        # create matrix for 1000 episode and its values
        val_list.append(cur_episode_record)
        i += 1
    np_list = np.array(val_list)
    median = np.median(np_list, axis=0)
    stand_dev = np.std(np_list, axis=0)
    median_mins_std = median - stand_dev
    median_add_std = median + stand_dev

    plt.title("E2F5: Median winnings of 1000 Episodes with limited bankroll")
    plt.plot(median_add_std, 'r--', alpha=0.5, label="median + standard_deviation")
    plt.plot(median_mins_std, 'g--', alpha=0.5, label="median - standard_deviation")
    plt.plot(median, 'b-', label="median")
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig("f5.png")
    plt.clf()


def test_code():
    """  		  	   		 	 	 			  		 			     			  	 
    Method to test your code  		  	   		 	 	 			  		 			     			  	 
    """
    win_prob = 18 / 38  # the actual prob of a win in real world
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 			  		 			     			  	 
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	 	 			  		 			     			  	 
    # add your code here to implement the experiments
    experiment_1_figure_1(win_prob)
    experiment_1_figure_2(win_prob)
    experiment_1_figure_3(win_prob)
    experiment_2_figure_4(win_prob)
    experiment_2_figure_5(win_prob)


if __name__ == "__main__":
    test_code()
