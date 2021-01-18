from IPOMCP_solver.pomcp import POMCP
from Problems.tiger.tiger_environment import TigerEnvironment
from utils import run_simulation

states = ['TL','TR']
actions = ['L','OL','OR']
observations = ['GL','GR']
sensor_accuracy = 0.85
reward_listen = -1
reward_gold = 10
reward_tiger = -100
noisy = True
beta = None


gamma = 0.9
epsilon = 0.01
horizon =3

environment = TigerEnvironment(states, observations, actions, sensor_accuracy, reward_listen, reward_gold, reward_tiger, noisy, beta)
pomcp = POMCP(environment, gamma, epsilon, horizon)

history = ''
initial_belief = {0 : 0.5 , 1 : 0.5}

#testing first best action
#root_node, best_action=pomcp.search(history, initial_belief, i = 10000)
#print("Root Node: "+ str(root_node))
#print("Best Action: "+ str(best_action))
#sys.exit()

#finite horizon, averaged across multiple episodes
'''
num_episodes = 100
rewards = []
#run for all time-steps
steps = horizon
steps = 3
for i in range(0, num_episodes):
    print("Episode: "+ str(i))
    cur_reward = run_simulation(pomcp, initial_belief, steps)
    rewards.append(cur_reward)

print("Average Reward:" + str(np.mean(rewards)))
print("Standard deviation: " + str(np.std(rewards)))
'''

#infinite time horizon

horizon = 1000 #to simulate infinite horizon, would terminate much earlier whenever gamma**depth < epsilon
steps =10
pomcp = POMCP(environment, 0.95, epsilon, horizon)
cur_reward = run_simulation(pomcp, initial_belief, steps)
print("cur_reward: "+ str(cur_reward))




