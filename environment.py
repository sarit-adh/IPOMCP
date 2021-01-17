import random
import numpy as np

class Environment(object):
	def __init__(self, states, observations, actions, observation_table, reward_table, transition_table):
		self.states = states
		self.observations = observations
		self.actions = actions
		self.reward_table = reward_table
		self.observation_table = observation_table
		self.transition_table = transition_table
	
	def get_state_name(self,index):
		return self.states[index]

	def get_observation_name(self,index):
		return self.observations[index]

	def get_action_name(self,index):
		return self.actions[index]

	def sample_action(self):
		return random.randrange(0,len(self.actions))

	def sample_state_from_state_distribution(self):
		return random.randrange(0,len(self.states))

	def step(self, state, action):

		sampled_next_state = self.sample_next_state(state, action)
		sampled_observation = self.sample_observation(sampled_next_state, action)
		reward = self.get_reward(state, action)
		return sampled_next_state, sampled_observation, reward



		
	def sample_next_state(self,cur_state, action):

		if type(self.transition_table[cur_state][action][0]) is dict:
			next_state_distribution = self.transition_table[cur_state][action][0]
		else:
			next_state_distribution = self.transition_table[cur_state][action]
		rand_val = random.random()
		cum_probability = 0
		for next_state, prob in next_state_distribution.items():
			cum_probability += prob
			if rand_val <= cum_probability:
				return next_state
		return None
		
	def sample_observation(self,cur_state,prev_action):
		if type(self.observation_table[cur_state][prev_action][0]) is dict:
			observation_distribution = self.observation_table[cur_state][prev_action][0]
		else:
			observation_distribution = self.observation_table[cur_state][prev_action]
		rand_val = random.random()
		cum_probability = 0
		for observation, prob in observation_distribution.items():
			cum_probability += prob
			if rand_val <= cum_probability:
				return observation
		return None
		
	def get_reward(self, state, action):
		if type(self.reward_table[state][action]) is dict:
			return self.reward_table[state][action][0]
		else:
			return self.reward_table[state][action]



	def belief_update(self, prev_belief, action, observation):
		# print(prev_belief, action, observation)
		if type(prev_belief) is dict:
			new_belief = {}

			for is_ in prev_belief:
				sum_prob_transition = 0
				for is_prev in prev_belief:
					sum_prob_transition += self.get_transition_likelihood(is_prev.state, action, is_.state) * \
										   prev_belief[is_prev]
				likelihood = self.get_observation_likelihood(is_.state, observation, action) * sum_prob_transition
				new_belief[is_] = likelihood
			sum_prob = float(sum(new_belief.values()))
			# print(new_belief)
			normalized_belief = dict({k: (new_belief[k] / sum_prob) for k in new_belief})
			return normalized_belief
		else:
			return None


class MultiAgentEnvironment(object):
	def __init__(self, states, observations, actions, observation_table, reward_table, transition_table):
		self.states = states
		self.observations = observations
		self.actions = actions
		self.reward_table = reward_table
		self.observation_table = observation_table
		self.transition_table = transition_table
	
	def get_state_name(self,index):
		return self.states[index]

	def get_observation_name(self,index):
		return self.observations[index]

	def get_action_name(self,index):
		return self.actions[index]

	def sample_action(self):
		return random.randrange(0,len(self.actions))
	
	def sample_state_from_state_distribution(self):
		return random.randrange(0,len(self.states))
		
	def sample_next_state(self,cur_state, action, action_other):
		next_state_distribution = self.transition_table[cur_state][action][action_other]
		rand_val = random.random()
		cum_probability = 0
		for next_state, prob in next_state_distribution.items():
			cum_probability += prob
			if rand_val <= cum_probability:
				return next_state
		return None
		
	def sample_observation(self,cur_state,prev_action, prev_action_other):
		observation_distribution = self.observation_table[cur_state][prev_action][prev_action_other]
		rand_val = random.random()
		cum_probability = 0
		for observation, prob in observation_distribution.items():
			cum_probability += prob
			if rand_val <= cum_probability:
				return observation
		return None
		
	def get_reward(self, state, action, action_other):
		return self.reward_table[state][action][action_other]

	def step(self, state, action, action_other):
		sampled_next_state = self.sample_next_state(state, action, action_other)
		sampled_observation = self.sample_observation(sampled_next_state, action, action_other)
		reward = self.get_reward(state, action, action_other)
		return sampled_next_state, sampled_observation, reward



#Redundant code TigerPOMDP and TigerEnvironment => REFACTOR	; #environment shouldn't have notion of noisy or beta	
class TigerEnvironment(Environment):
	def __init__(self,states, observations, actions, sensor_accuracy, reward_listen, reward_gold, reward_tiger, noisy, beta):
		reward_table = {}
		for state in range(0,len(states)):
			reward_table[state] = {}
			for action in range(0,len(actions)):
				if action == 0:
					reward_table[state][action] = reward_listen
				elif action-1 == state:
					reward_table[state][action] = reward_tiger
				else:
					reward_table[state][action] = reward_gold
		
		observation_table = {}
		for state in range(0,len(states)):
			observation_table[state] = {}
			for action in range(0,len(actions)):
				observation_table[state][action] = {}
				for observation_ind, observation in enumerate(observations):
					if beta is None: #No communication
						if action==0:
							if observation_ind==state:
								observation_table[state][action][observation_ind] = sensor_accuracy
							else:
								observation_table[state][action][observation_ind] = 1 - sensor_accuracy
						else:
							observation_table[state][action][observation_ind] = 1.0/ len(observations)	
					
					else: #There is communication
						if action==0:
							#if message is nil , belief update should be same as update due to observation only
							if observation[1] == 'NIL':
								if state==observation[0]:
									observation_table[state][action][observation_ind] = sensor_accuracy/((len(observations))/2.0)
								else:
									observation_table[state][action][observation_ind] = (1 - sensor_accuracy)/((len(observations))/2.0)
							else:
								if state == 0:
									if observation[0] == 0:

										observation_table[state][action][observation_ind] = (beta * sensor_accuracy + (1 - beta) * observation[1])/((len(observations))/2.0)
									else:	
										observation_table[state][action][observation_ind] = (beta * (1 - sensor_accuracy) + (1 - beta) * observation[1])/((len(observations))/2.0)
								else:
									if observation[0] == 1:
										observation_table[state][action][observation_ind] = (beta * sensor_accuracy + (1 - beta)* (1 - observation[1]))/((len(observations))/2.0)
									else:
										observation_table[state][action][observation_ind] = (beta * (1 - sensor_accuracy) + (1 - beta)* (1 - observation[1]))/((len(observations))/2.0)
						else:
							observation_table[state][action][observation_ind] = 1.0/ len(observations)
							
		#print(observation_table)
		transition_table = {}
		for state in range(0,len(states)):
			transition_table[state] = {}
			for action in range(0,len(actions)):
				transition_table[state][action] = {}
				for next_state in range(0,len(states)):
					if action==0:
						if noisy:
							if state==next_state:
								transition_table[state][action][next_state] = 0.9
							else:
								transition_table[state][action][next_state] = 0.1
						else:
							if state==next_state:
								transition_table[state][action][next_state] = 1.0
							else:
								transition_table[state][action][next_state] = 0.0
					else:
						transition_table[state][action][next_state] = 0.5
		
		super(TigerEnvironment, self).__init__(states, observations, actions, observation_table, reward_table, transition_table)
		
		
#0 L, 1 OL, 2 OR, reward independent of other's action		
class MultiAgentTigerEnvironment(MultiAgentEnvironment):
	def __init__(self,states, observations, actions, sensor_accuracy, reward_listen, reward_gold, reward_tiger):
		reward_table = {}
		for state in range(0,len(states)):
			reward_table[state] = {}
			for action in range(0,len(actions)):
				reward_table[state][action]={}
				for action_other in range(0,len(actions)):
					if action == 0:
						reward_table[state][action][action_other] = reward_listen
					elif action-1 == state: #0=> TL, 1=> TR #1=>OL 2=> OR 
						reward_table[state][action][action_other] = reward_tiger
					else:
						reward_table[state][action][action_other] = reward_gold
		
		observation_table = {}
		for state in range(0,len(states)):
			observation_table[state] = {}
			for action in range(0,len(actions)):
				observation_table[state][action] = {}
				for action_other in range(0,len(actions)):
					observation_table[state][action][action_other]={}
					for observation in range(0, len(observations)):
						if action==0:
							if observation==state:
								observation_table[state][action][action_other][observation] = sensor_accuracy
							else:
								observation_table[state][action][action_other][observation] = 1 - sensor_accuracy
						else:
							observation_table[state][action][action_other][observation] = 1.0/ len(observations)
							
		transition_table = {}
		for state in range(0,len(states)):
			transition_table[state] = {}
			for action in range(0,len(actions)):
				transition_table[state][action] = {}
				for action_other in range(0,len(actions)):
					transition_table[state][action][action_other]={}
					for next_state in range(0,len(states)):
						if action==0 and action_other==0:
							if state==next_state:
								transition_table[state][action][action_other][next_state] = 1.0
							else:
								transition_table[state][action][action_other][next_state] = 0.0
						else:
							transition_table[state][action][action_other][next_state] = 0.5
							
		super(MultiAgentTigerEnvironment, self).__init__(states, observations, actions, observation_table, reward_table, transition_table)
						
	