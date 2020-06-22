import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import stdout
from tick import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.datasets import load_boston
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import time
from tick.preprocessing.features_binarizer import FeaturesBinarizer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from operator import itemgetter
import pandas as pd
import seaborn as sns

from dev.library import compute_score, get_groups
from dev.fitted_Q_learning import Q_value, binarsity_reg

# hyper-params setting
nb_actions = 2   #nb of actions in the environnement (2 for CarPole-v0)
gamma = 0.98 # discount factor
K = 100   # nb of episodes
N = 4000      # len of each episode for the first idea (equal to the number of observations in each regressions
              #  with binarsity penalty)
epsilon = 0.9 # exploration rate (for the first idea of algorithm)
alpha = 0.5 # learning rate (for the first idea of algorithm)

N_2 = 50000  # len of each episode for the second idea (equal to jump_length TIMES the number of
             #  observations in each regressions with binarsity penalty)
grid_C = list(np.round(np.logspace(-1, 2, 20),3)) # the weigth associated to the binarsity penalty will be chosen
                                                  #  by cross-validation among this list
jump_length = 35 # for building the training sample in the second idea of algorithm
nb_cells_limit = 1000 # max size of the discretization computed in the second idea of algorithm


t0 = time.clock()
env_name = "CartPole-v0"
env = gym.make(env_name)

Q_k_params = []
for k in range(K):  # run K episodes

    print("\nk=%s\n" % k)
    print("exploration phase :")
    # Exploration_phase
    Q_k_minus_1_params = Q_k_params.copy()

    # in the algorithm we need to update simulteanously all the new discrete
    # approximate action-values function
    #  so we need to keep a copy the parameters that define these

    episode = []
    # where we are  going to store the transition tuples observed during the exploration phase

    # Exploration phase
    print("epsilon :", epsilon)
    for i in range(N):
        stdout.write("\r   i: %s/%s" % (i, N))
        stdout.flush()

        if i > 0:
            if episode[i - 1]["end_i"] == True:
                s_i = env.reset()
            else:
                s_i = episode[i - 1]["s_i_plus_1"]
        else:
            s_i = env.reset()

        if np.random.uniform() < epsilon:
            a_i = env.action_space.sample()
        else:
            key_a_i = np.array([Q_value(Q_k_minus_1_params, act, s_i) for act in range(nb_actions)])
            key_a_i = np.argwhere(key_a_i == max(key_a_i)).flatten().tolist()
            a_i = np.random.choice(key_a_i)

        s_i_plus_1, r_i, end_i, _ = env.step(a_i)

        if i < 200:
            env.render()

        episode.append({"s_i": s_i,
                        "a_i": a_i,
                        "r_i": r_i,
                        "end_i": end_i,
                        "s_i_plus_1": s_i_plus_1})

    # Discretization phase
    print(" ")
    print(" ")
    print("discretization phase :")
    Q_k_params = [0] * nb_actions

    for l in range(nb_actions):

        # transition tuples for which the action chosen is the action l
        data_l = [episode[i] for i in range(N) if episode[i]["a_i"] == l]

        # creation of the learning sample {(s_i, q_l(s_i) ; a_i=l} which will be used to compute the updated
        # discrete action-value function for action l
        learning_sample = []
        for i in range(len(data_l)):
            # if the state s_i is terminal then there is update of value at state s_i with the value at state s_(i+1) which is reseted
            tmp = 0 if data_l[i]["end_i"] else gamma * max([Q_value(Q_k_minus_1_params,
                                                                     l_,
                                                                     data_l[i]["s_i_plus_1"])
                                                            for l_ in range(nb_actions)])

            learning_sample.append(
                [data_l[i]["s_i"],
                 (1 - alpha) * Q_value(Q_k_minus_1_params, l, data_l[i]["s_i"]) + alpha * (data_l[i]["r_i"] + tmp)]
            )

        # binarsity regression applied on the learning sample :
        X = [list(learning_sample[i][0]) for i in range(len(learning_sample))]
        # print(X[0])

        X_data = pd.DataFrame(np.array(X), columns=[str(j) + ":continuous" for j in range(len(X[0]))], dtype='float64')
        y = np.array([learning_sample[i][1] for i in range(len(learning_sample))])

        # new discretized action-value function approximator for action l
        print(" ")
        print("k =", k, "action :", l)
        Q_k_params[l] = binarsity_reg(X_data, y, grid_C=np.logspace(-1, 3, 15))

env.close()

t1 = time.clock()
print("total time elapsed :", t1 - t0)

# the final (discrete) policy learned by the algorithm :
hat_pi = lambda s: np.argmax([Q_value(Q_k_params, l, s) for l in range(nb_actions)])

# testing phase
env.reset()
obs = env.reset()

Q_k_minus_1_params = Q_k_params.copy()

total_reward = []
temp = 0
end = False
for i in range(100):
    while end != True:

        # if we want to select the action chosen by the policy computed with the second idea
        key_a = np.array([Q_value(Q_k_minus_1_params, act, obs) for act in range(nb_actions)])
        key_a = np.argwhere(key_a == max(key_a)).flatten().tolist()
        a = np.random.choice(key_a)
        obs, reward, end, _ = env.step(a)
        temp += reward
        env.render()

    obs = env.reset()
    total_reward.append(temp)
    temp = 0
    end = False
env.close()

# visualize testing result
n, bins, patches = plt.hist(x=total_reward, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Reward')
plt.show()