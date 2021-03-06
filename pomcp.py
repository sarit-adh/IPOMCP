import random

import numpy as np

from node import *


class POMCP:
    def __init__(self, environment, gamma=1, epsilon=0.01, horizon=3):
        self.environment = environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon = horizon
        self.history_to_node = {}

    def search(self, h, initial_belief,i=10000):
        N = i
        #self.history_to_node = {} #should we resuse the built tree in previous time step , after action and observation, root of tree is reset
        while i>0:
            #print("iteration:" + str(N - i + 1))
            if len(h)==0:
                sampled_state = np.random.choice(list(initial_belief.keys()), 1, list(initial_belief.values()))[0]

            else:
                sampled_state = self.sample_state_from_history(h)

            #print("running simulate with sampled_state: " + str(sampled_state))
            self.simulate(sampled_state,h,0)
            i-=1

            #debugging START

            for action in range(0, len(self.environment.actions)):
                mean_value = self.history_to_node[h + str(action)].get_mean_value()
                #print("Mean value of taking action "+ str(action) +"  : " + str(mean_value))
            #debugging END
        root_node = self.history_to_node[h]
        return root_node, np.argmax([self.history_to_node[h+str(action)].get_mean_value() for action in range(0, len(self.environment.actions))])

    def simulate(self,s,h,depth):
        if self.gamma ** depth < self.epsilon or depth==self.horizon :
            return 0

        # if depth == self.horizon-1: #last time to act, observation doesn't matter, no need to create node
        #     return self.rollout(s, h, depth)

        if h not in self.history_to_node:
            # next 3 lines are not in the algorithm in the paper
            #print("creating node with history: "+ str(h))
            cur_node = Node(h, None)
            self.history_to_node[h] = cur_node
            cur_node.particle_set.append(s)

            for action in range(0, len(self.environment.actions)):
                #print("creating node with history: " + str(h+str(action)))
                new_node = Node(h+str(action), cur_node)
                self.history_to_node[h+str(action)] = new_node
            if not isinstance(s, (int, np.integer)) : s = s.state
            rollout_value = self.rollout(s,h,depth)
            #print("Rollout Value for state "+ str(s)+" and history "+ str(h)+" is "+str(rollout_value))
            return rollout_value

        #print("Retrieving node with history: "+ str(h))
        cur_node = self.history_to_node[h]

        #print([self.history_to_node[h + str(action)].ucb_score() for action in range(0, len(self.environment.actions))])
        max_action = np.argmax([self.history_to_node[h+str(action)].ucb_score() for action in range(0, len(self.environment.actions))])
        #print("Action "+ str(max_action)+ " selected according to ucb score")

        if not isinstance(s, (int, np.integer)): s = s.state

        next_state, observation, reward = self.environment.step(s, max_action)
        #print("sampled next_state "+ str(next_state)+" observation "+ str(observation)+" reward "+str(reward))

        max_action_node = self.history_to_node[h+str(max_action)]


        R = reward + self.gamma * self.simulate(next_state, h+str(max_action)+str(observation), depth+1)
        cur_node.particle_set.append(s)
        cur_node.times_visited+=1
        max_action_node.times_visited+=1
        max_action_node.value_sum += R #different from the paper
        return R

    def rollout(self, s, h, depth):
        #print("Executing Rollout with history: "+ str(h))
        if self.gamma ** depth < self.epsilon or depth==self.horizon :
            return 0

        sampled_action = self.environment.sample_action()
        #sampled_action = 0
        if not isinstance(s, (int, np.integer)): s = s.state
        next_state, observation, reward = self.environment.step(s, sampled_action)
        #print(s, sampled_action, next_state, observation, reward)
        return reward + self.gamma* self.rollout(next_state, h+str(sampled_action)+str(observation), depth+1)

    def sample_state_from_history(self, history):
        #print(self.history_to_node)
        node = self.history_to_node[history]
        #print("Belief: " + str([node.particle_set.count(state) / len(node.particle_set) for state in range(0,len(self.environment.states))]))
        return random.sample(node.particle_set,1)[0]











