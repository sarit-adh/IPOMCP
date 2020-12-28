from utils import *
import sys
from environment import *
from pomcp import POMCP
from ipomcp import IPOMCP

filename = "Tiger.L1_no_creek.txt"
physical_states, frames_all_level = read_ipomdp_file(filename, comm=False)
top_level = max(list(map(int,frames_all_level.keys())))
cur_level = 0
model_list_prev_level = None

while cur_level <= top_level:

    is_list_cur_level = []

    # create interactive states for current level
    for physical_state in range(0, len(physical_states)):
        if model_list_prev_level is None:
            is_list_cur_level.append(InteractiveState(physical_state, None))
        else:
            for model_prev_level in model_list_prev_level:
                is_list_cur_level.append(InteractiveState(physical_state, model_prev_level))

    # interactive_states_dict[cur_level] = is_list_cur_level

    # print(is_list_cur_level)
    initial_beliefs = [[1.0 / len(is_list_cur_level)] * len(is_list_cur_level)]  # uncomment for uniform belief

    if cur_level < top_level:  # we don't need models for highest level ipomdp
        frames_cur_level = frames_all_level[str(cur_level)]
        model_list_cur_level = []
        for frame_cur_level in frames_cur_level:
            frame_cur_level.level = cur_level
        for initial_belief in initial_beliefs:

            belief_dict = {is_: belief_e for belief_e, is_ in zip(initial_belief,
                                                                  is_list_cur_level)}  # check might be source of bug when intial belief is <TL, TR>
            for frame_cur_level in frames_cur_level:
                #cur_model = Model.create_or_get_model(belief_dict, frame_cur_level)

                if cur_level==0:
                    environment = Environment(physical_states, frame_cur_level.observations,
                                                        frame_cur_level.actions, frame_cur_level.observation_table,
                                                        frame_cur_level.reward_table, frame_cur_level.transition_table)
                    solver = POMCP(environment)
                else:
                    environment = MultiAgentEnvironment(physical_states,frame_cur_level.observations,frame_cur_level.actions,frame_cur_level.observation_table,frame_cur_level.reward_table,frame_cur_level.transition_table)
                    solver = IPOMCP(environment)

                cur_model = Model(belief_dict, "", frame_cur_level, solver)
                model_list_cur_level.append(cur_model)
        if cur_level == 0:
            model_list_0_level = model_list_cur_level
        model_list_prev_level = model_list_cur_level

    cur_level += 1


top_level_frame = frames_all_level[str(top_level)][0]

initial_belief_values = [1.0/len(is_list_cur_level)]* len(is_list_cur_level)
initial_belief = dict(zip(is_list_cur_level, initial_belief_values))
history = ""
iterations = 10000

environment = MultiAgentEnvironment(physical_states, top_level_frame.observations, top_level_frame.actions, top_level_frame.observation_table, top_level_frame.reward_table, top_level_frame.transition_table)
ipomcp = IPOMCP(environment)
ind = np.random.choice(len(initial_belief.keys()), 1, p=list(initial_belief.values()))[0]
sampled_interactive_state = list(initial_belief.keys())[ind]

# root_node, best_action = ipomcp.search(history, initial_belief,iterations)
# print("current best action: " + str(best_action))
#
#
# ind = np.random.choice(len(initial_belief.keys()), 1, p=list(initial_belief.values()))[0]
# sampled_interactive_state = list(initial_belief.keys())[ind]
# print("current state and opponent:" + str(sampled_interactive_state))
# state = sampled_interactive_state.state
# opponent_action= sampled_interactive_state.model.best_action
# print("opponent action" + str(opponent_action))
# next_state, observation, reward = ipomcp.environment.step(state, best_action, opponent_action)
# print("next state: " + str(next_state), "observation: " + str(observation), "reward: " + str(reward))
#
# next_state, opponent_observation, reward = sampled_interactive_state.model.solver.environment.step(sampled_interactive_state.state, opponent_action)
#
# opponent_history = sampled_interactive_state.model.history + str(opponent_action) + str(opponent_observation)
# sampled_interactive_state = ipomcp.sample_interactive_state_from_history(opponent_history)
#
# history = history + str(best_action)+str(observation)
#
# root_node, best_action = ipomcp.search(history, initial_belief,iterations)
# print(best_action)
#
# print(sampled_interactive_state.model.best_action)

steps =0
while steps<3:
    print("Step: " + str(steps+1))
    root_node, best_action = ipomcp.search(history, initial_belief, iterations)
    print("current best action: " + str(best_action))
    print("current state and opponent:" + str(sampled_interactive_state))
    state = sampled_interactive_state.state
    opponent_action = sampled_interactive_state.model.best_action
    print("opponent action" + str(opponent_action))
    next_state, observation, reward = ipomcp.environment.step(state, best_action, opponent_action)
    print("next state: " + str(next_state), "observation: " + str(observation), "reward: " + str(reward))
    next_state, opponent_observation, reward = sampled_interactive_state.model.solver.environment.step(
        sampled_interactive_state.state, opponent_action)
    opponent_history = sampled_interactive_state.model.history + str(opponent_action) + str(opponent_observation)
    history = history + str(best_action) + str(observation)
    sampled_interactive_state = ipomcp.sample_interactive_state_from_history(history) #should it be history or opponent history

    steps+=1
