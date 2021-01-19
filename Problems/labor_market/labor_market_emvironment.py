from Environment.environment import Environment


class LaborMarketEnvironment(Environment):

    def __init__(self, number_of_trails, input_file):
        super().__init__(number_of_trails, input_file)

    def simulate_environment(self, agent_types_list: list, starting_agent):
        """
        This method simulate the interaction between the agents
        :param starting_agent:
        :param agent_types_list:
        :return:
        """
        pass

    def create_market(self):
        pass

