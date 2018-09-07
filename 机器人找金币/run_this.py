# -*- coding: utf-8 -*-
'''
Author: winddy
'''
import numpy as np
from grid_mdp import GridEnv
from DQN_modified import DeepQNetwork

env = GridEnv()
RL = DeepQNetwork(len(env.getAction()), len(env.getStates()),
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.1,
                  replace_target_iter=200,
                  memory_size=2000)

episodes = 2000
step = 0
for i in range(episodes):

    state = env.reset()
    while True:
        env.render()

        feature = [0] * len(env.getStates())
        feature[state - 1] = 1
        feature = np.hstack(feature)
        action = RL.choose_action(feature)

        state_, reward, done = env.step(action)

        feature_ = [0] * len(env.getStates())
        feature_[state_ - 1] = 1
        feature_ = np.hstack(feature_)

        RL.store_transition(feature, action, reward, feature_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        state = state_

        if done:
            break
        step += 1

