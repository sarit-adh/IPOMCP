from interactive_state import *
from node import *
import numpy as np
import random


class IPOMCP:
    def __init__(self, environment, gamma=1, epsilon=0.01, horizon=3):
        self.environment = environment
        self.gamma = gamma
        self.epsilon = epsilon
        self.horizon = horizon
        self.history_to_node = {}

    def search(self, h, initial_belief,i=10000):
        while i>0:
            #print("iteration:" + str(N - i + 1))
            if len(h)==0:
                sampled_interactive_state = np.random.choice(list(initial_belief.keys()), 1, list(initial_belief.values()))[0]

            else:
                sampled_interactive_state = self.sample_interactive_state_from_history(h)

            #print("running simulate with sampled_interactive_state: " + str(sampled_interactive_state))
            self.simulate(sampled_interactive_state,h,0)
            i-=1

            #debugging START

            for action in range(0, len(self.environment.actions)):
                mean_value = self.history_to_node[h + str(action)].get_mean_value()
                #print("Mean value of taking action "+ str(action) +"  : " + str(mean_value))
            #debugging END
        root_node = self.history_to_node[h]
        return root_node, np.argmax([self.history_to_node[h+str(action)].get_mean_value() for action in range(0, len(self.environment.actions))])

    def simulate(self,is_,h,depth):
        if self.gamma ** depth < self.epsilon or depth==self.horizon :
            return 0

        # if depth == self.horizon-1: #last time to act, observation doesn't matter, no need to create node
        #     return self.rollout(s, h, depth)

        if h not in self.history_to_node:
            # next 3 lines are not in the algorithm in the paper
            #print("creating node with history: "+ str(h))
            cur_node = Node(h, None)
            self.history_to_node[h] = cur_node
            cur_node.particle_set.append(is_) #should each particle represt is as a whole, or store s and model particles separately??

            for action in range(0, len(self.environment.actions)):
                #print("creating node with history: " + str(h+str(action)))
                new_node = Node(h+str(action), cur_node)
                self.history_to_node[h+str(action)] = new_node
            rollout_value = self.rollout(is_,h,depth)
            #print("Rollout Value for state "+ str(is_.state)+" and history "+ str(h)+" is "+str(rollout_value))
            return rollout_value

        #print("Retrieving node with history: "+ str(h))
        cur_node = self.history_to_node[h]

        #print([self.history_to_node[h + str(action)].ucb_score() for action in range(0, len(self.environment.actions))])
        max_action = np.argmax([self.history_to_node[h+str(action)].ucb_score() for action in range(0, len(self.environment.actions))])
        #print("Action "+ str(max_action)+ " selected according to ucb score")

        root_node, action_other = is_.model.solver.search(is_.model.history, is_.model.belief, 100)
        #print("Action other: "+ str(action_other))
        is_.model.best_action = action_other

        next_state, observation, reward = self.environment.step(is_.state, max_action, action_other)
        #print("sampled next_state "+ str(next_state)+" observation "+ str(observation)+" reward "+str(reward))

        #TODO create next interactive state

        next_state, observation, reward = is_.model.solver.environment.step(is_.state, action_other) # level 0, TODO for level>0

        new_history = is_.model.history + str(action_other) + str(observation)
        new_model = Model(is_.model.belief, new_history, is_.model.frame, is_.model.solver)

        next_is_ = InteractiveState(next_state, new_model) #which next state to use, one from this agent or other agent? Does it matter?

        #END Create next interactive state

        max_action_node = self.history_to_node[h+str(max_action)]


        R = reward + self.gamma * self.simulate(next_is_, h+str(max_action)+str(observation), depth+1)
        cur_node.particle_set.append(is_)
        cur_node.times_visited+=1
        max_action_node.times_visited+=1
        max_action_node.value_sum += R #different from the paper
        return R

    def rollout(self, is_, h, depth):
        #print("Executing Rollout with history: "+ str(h))
        if self.gamma ** depth < self.epsilon or depth==self.horizon :
            return 0

        sampled_action = self.environment.sample_action()
        #print(is_)
        sampled_action_other = is_.model.solver.environment.sample_action()
        #sampled_action = 0
        next_state, observation, reward = self.environment.step(is_.state, sampled_action, sampled_action_other)
        #print(is_.state, sampled_action, next_state, observation, reward)
        next_is_ = InteractiveState(next_state,is_.model) #TODO change to new model
        return reward + self.gamma* self.rollout(next_is_, h+str(sampled_action)+str(observation), depth+1)

    def sample_interactive_state_from_history(self, history):
        node = self.history_to_node[history]
        #print("Belief: " + str([node.particle_set.count(state) / len(node.particle_set) for state in range(0,len(self.environment.states))]))
        return random.sample(node.particle_set,1)[0]
