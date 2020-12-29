import numpy as np
from interactive_state import InteractiveState, Model, Frame

def run_simulation(pomcp, initial_belief, steps):
    history = ''
    time_step = 0
    total_reward = 0
    while time_step<steps:
        print("time-step:" + str(time_step))

        root_node, best_action = pomcp.search(history, initial_belief,i=10000)


        if time_step==0:
            state = np.random.choice(list(initial_belief.keys()), 1, list(initial_belief.values()))[0]
        else:
            # state = pomcp.sample_state_from_history(history) #TODO check this step, is exact belief update required
            state = next_state #current state is set to next state from previous time step



        next_state, observation, reward = pomcp.environment.step(state, best_action)

        history = history + str(best_action)+str(observation)

        print("state, action, next_state, observation, reward")
        print(state, best_action, next_state, observation, reward)

        node = pomcp.history_to_node[history]
        print("Belief: " + str(
            [node.particle_set.count(state) / len(node.particle_set) for state in range(0, len(pomcp.environment.states))]))
        time_step+=1
        total_reward+= reward

    return total_reward


# Function to parse IPOMDP file into IPOMDP object, frames of lower level are also parsed, interactive states not generated

def read_ipomdp_file(filename, comm=False):
    with open(filename) as file:
        lines = file.readlines()
        # print(lines)
        states = None
        actions = None
        # other_actions = None
        observations = None
        num_frames = None
        cur_nesting_level = None
        observation_function = {}
        transition_function = {}
        reward_function = {}
        highest_level_IPOMDP = None
        cur_frame_id = None
        frames_all_level = {}  # dict {level : list of frames}
        for i in range(0, len(lines)):
            if lines[i].startswith("STATES"):
                num_states = int(lines[i].split(':')[-1])
                states = list(map(lambda x: x.strip().strip(","), lines[i - 1].split(" ")[-num_states:]))
            if lines[i].startswith("ACTIONS"):
                num_actions = int(lines[i].split(':')[-1])
                actions = list(map(lambda x: x.strip(), lines[i - 1].split(" ")[-num_actions:]))
            if lines[i].startswith("OBSERVATIONS"):
                num_observations = int(lines[i].split(':')[-1])
                observations = list(map(lambda x: x.strip(), lines[i - 1].split(" ")[-num_observations:]))
            if lines[i].startswith("OTHERS_ACTIONS"):
                pass
            # num_others_actions = int(lines[i].split(':')[-1])
            # other_actions = list(map(lambda x: x.strip(), lines[i-1].split(" ")[-num_others_actions:]))
            # TODO LATER currently assume that the other agent has the same set of actions as current agent
            if lines[i].startswith("FRAMES"):
                num_frames = lines[i].split(':')[-1].strip()
            if lines[i].startswith("LEVELS"):
                cur_nesting_level = lines[i].split(':')[-1].strip()
            if lines[i].startswith("FRAME ID"):
                cur_frame_id = lines[i].split(':')[-1].strip()
            if lines[i].startswith("O:"):
                print("\n")
                action_i, action_j = list(lines[i].split(":")[-1].strip().split())
                if action_i != '*':  action_i = int(action_i)
                if action_j != '*':  action_j = int(action_j)
                for k in range(0, num_states):
                    if k not in observation_function:
                        observation_function[k] = {}
                    if action_i not in observation_function[k]:
                        observation_function[k][action_i] = {}
                    if action_j not in observation_function[k][action_i]:
                        if action_j == '*':
                            for action_j_ in range(0, num_actions):
                                observation_function[k][action_i][action_j_] = {}
                        else:
                            observation_function[k][action_i][action_j] = {}
                    observations_for_cur_state = map(lambda x: float(x.strip()), lines[i + k + 1].split())
                    for cur_observation_ind, cur_observation_prob in enumerate(observations_for_cur_state):
                        if action_j == '*':
                            for action_j_ in range(0, num_actions):
                                observation_function[k][action_i][action_j_][cur_observation_ind] = cur_observation_prob
                        else:
                            observation_function[k][action_i][action_j][cur_observation_ind] = cur_observation_prob

                print(observation_function)

            if lines[i].startswith("T:"):
                action_i, action_j = list(lines[i].split(":")[-1].strip().split())
                if action_i != '*':  action_i = int(action_i)
                if action_j != '*':  action_j = int(action_j)
                for k in range(0, num_states):
                    if k not in transition_function:
                        transition_function[k] = {}
                    if action_i not in transition_function[k]:
                        transition_function[k][action_i] = {}
                    if action_j not in transition_function[k][action_i]:
                        if action_j == '*':
                            for action_j_ in range(0, num_actions):
                                transition_function[k][action_i][action_j_] = {}
                        else:
                            transition_function[k][action_i][action_j] = {}
                    transitions_from_cur_state = map(lambda x: float(x.strip()), lines[i + k + 1].split())
                    for next_state, transition_prob in enumerate(transitions_from_cur_state):
                        if action_j == '*':
                            for action_j_ in range(0, num_actions):
                                transition_function[k][action_i][action_j_][next_state] = transition_prob
                        else:
                            transition_function[k][action_i][action_j][next_state] = transition_prob

            # print(transition_function)

            if lines[i].startswith("R:"):
                action_i, action_j = list(lines[i].split(":")[-1].strip().split())
                if action_i != '*':  action_i = int(action_i)
                if action_j != '*':  action_j = int(action_j)
                reward_list = lines[i + 1].split()
                for k in range(0, num_states):
                    if k not in reward_function:
                        reward_function[k] = {}
                    if action_i not in reward_function[k]:
                        reward_function[k][action_i] = {}

                    print(reward_list)
                    reward_function[k][action_i][action_j] = float(reward_list[k])

            # print(reward_function)

            if states and actions and observations and num_frames and cur_nesting_level and len(
                    reward_function) == num_states and len(reward_function[0]) == num_actions and len(
                    reward_function[0][
                        num_actions - 1]) == num_actions:  # when everything required for IPOMDP/POMDP definition has been obtained
                print(states, actions, observations, num_frames, cur_nesting_level)
                print("\nTransition Function")
                print(transition_function)
                print("\nObservation Function")
                print(observation_function)
                print("\nReward Function")
                print(reward_function)



                frame = Frame(cur_frame_id, actions, observations, transition_function, observation_function,
                              reward_function, "i", None, int(cur_nesting_level) )
                if cur_nesting_level in frames_all_level:
                    frames_all_level[cur_nesting_level].append(frame)
                else:
                    frames_all_level[cur_nesting_level] = [frame]

                '''

                if highest_level_IPOMDP is None:
                    if comm:
                        print("creating highest level cipomdp....")
                        ipomdp_cur = cipomdp.CIPOMDP(states, actions, observations, transition_function,
                                                     observation_function, reward_function, int(cur_nesting_level))
                    else:
                        print("creating highest level ipomcp....")
                    # ipomdp_cur = ipomdp.IPOMDP(states, actions, observations, transition_function, observation_function, reward_function, int(cur_nesting_level))
                    highest_level_IPOMDP = ipomdp_cur
                else:
                    frame = Frame(cur_frame_id, actions, observations, transition_function, observation_function,
                                  reward_function, "i")
                    if cur_nesting_level in frames_all_level:
                        frames_all_level[cur_nesting_level].append(frame)
                    else:
                        frames_all_level[cur_nesting_level] = [frame]

                # TODO generate interactive states based on frame
                
                '''

                #states = None
                actions = None
                other_actions = None
                observations = None
                num_frames = None
                cur_nesting_level = None
                transition_function = {}
                observation_function = {}
                reward_function = {}

                print("\n\n")

    return states, frames_all_level
