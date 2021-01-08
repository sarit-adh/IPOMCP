from Environment.environment import IPOMDPEnvironment
from Problems.rock_sample.rs_objects import *


class RockSampleEnvironment(IPOMDPEnvironment):

    def __init__(self, n, k, half_efficiency_dist=20):
        self._n = n
        self._k = k
        self._rock_locs = self._set_rocks_locations()
        self._rock = {}
        self.init_state = self._set_initial_state()
        self.half_efficiency_dist = half_efficiency_dist
        super().__init__(None, None, None)

    def __str__(self):
        return f'Rock Sample problem with grid size {self._n} by {self._n}' \
               f' {self} rocks'
        
    def print_map(self):
        initial_map = np.zeros([self._n, self._n])
        for i in range(self._rock_locs.shape[1]):
            loc = self._rock_locs[:, i]
            initial_map[loc[0], loc[1]] = 1 if self.init_state.rock_types[i] == 'good' else -1
        return initial_map

    def _set_rocks_locations(self) -> np.array:
        """
        This method randomly sets the k rocks
        :return:
        """
        x_locs = np.random.randint(0, self._n, self._k)
        y_locs = np.random.randint(0, self._n, self._k)
        locations = np.array([x_locs, y_locs])
        return locations

    def _set_initial_state(self, state=None) -> State:
        """Returns init_state and rock locations for an instance of RockSample(n,k)"""
        rover_position = (0, 0)
        for i in range(self._rock_locs.shape[1]):
            loc = self._rock_locs[:, i]
            # Each rock has a location
            self._rock[i] = loc
        rock_types = np.random.choice([Rock.GOOD, Rock.BAD], self._k)
        # Ground truth state
        init_state = State(rover_position, rock_types, False)
        return init_state

    def _in_exit_area(self, position):
        return position[0] == self._n

    def _move_or_exit(self, position, action):
        expected = (position[0] + action.motion[0],
                    position[1] + action.motion[1])
        if self._in_exit_area(expected):
            return expected, True
        else:
            return (max(0, min(position[0] + action.motion[0], self._n - 1)),
                    max(0, min(position[1] + action.motion[1], self._n - 1))), False

    def transition_function(self, state, actions, **kwargs):
        next_position = tuple(state.position)
        rock_types = tuple(state.rock_types)
        next_rock_types = rock_types
        next_terminal = state.is_terminal
        if state.is_terminal:
            next_terminal = True
        else:
            if isinstance(actions, MoveAction):
                next_position, exiting = self._move_or_exit(state.position, actions)
                if exiting:
                    next_terminal = True
            elif isinstance(actions, SampleAction):
                if next_position in self._rock_locs:
                    rock_id = self._rock_locs[next_position]
                    _rocktypes = list(rock_types)
                    _rocktypes[rock_id] = Rock.BAD
                    next_rock_types = tuple(_rocktypes)
        return State(next_position, next_rock_types, next_terminal)

    def observation_function(self, state, actions, next_state, **kwargs):
        rocks = {self._rock_locs[pos]: pos for pos in self._rock_locs}
        if not next_state.is_terminal and isinstance(actions, CheckAction):
            # compute efficiency
            rock_pos = rocks[actions.rock_id]
            dist = euclidean_dist(rock_pos, next_state.position)
            eta = (1 + pow(2, -dist / self.half_efficiency_dist)) * 0.5
            keep = eta > 0.5
            true_rock_type = next_state.rocktypes[actions.rock_id]
            if not keep:
                observed_rock_type = Rock.invert(true_rock_type)
                return Observation(observed_rock_type)
            else:
                return Observation(true_rock_type)
        else:
            # Terminated or not a check action. So no observation.
            return Observation(None)
        #return self._probs[next_state][action][observation]

    def reward_function(self, state, actions, **kwargs):
        # deterministic
        if state.is_terminal:
            return 0  # terminated. No reward
        if isinstance(actions, SampleAction):
            # need to check the rocktype in `state` because it has turned bad in `next_state`
            if state.position in self._rock_locs:
                if state.rocktypes[self._rock_locs[state.position]] == Rock.GOOD:
                    return 10
                else:
                    # No rock or bad rock
                    return -10
            else:
                return 0  # problem didn't specify penalty for sampling empty space.

    def step(self, state, actions, **kwargs) -> object:
        """

        :rtype: object
        """
        next_state = self.transition_function(state, actions)
        observation = self.observation_function(state, actions, next_state)
        reward = self.reward_function(state, actions)
        return next_state, observation, reward

    def pomdp_step(self, actions):
        state = self.current_state
        next_state = self.transition_function(state, actions)
        observation = self.observation_function(state, actions, next_state)
        reward = self.reward_function(state, actions)
        return observation, reward


def test_rocksample_env():
    rs_env = RockSampleEnvironment(8, 5)
    print(rs_env.print_map())


if __name__ == '__main__':
    test_rocksample_env()
