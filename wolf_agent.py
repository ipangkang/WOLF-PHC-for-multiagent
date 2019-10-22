import numpy as np

class WoLFAgent():
    """
        Policy hill-climbing algorithm(PHC)
        http://www.cs.cmu.edu/~mmv/papers/01ijcai-mike.pdf
    """
    def __init__(self, alpha=0.1, delta=0.0001, actions=None, high_delta=0.004, low_delta=0.002):
        self.alpha = alpha
        self.actions = actions  
        self.last_action_id = None
        self.q_values = self._init_q_values()
        self.pi = [(1.0/len(actions)) for idx in range(len(actions))]
        self.pi_average = [(1.0/len(actions)) for idx in range(len(actions))]
        self.high_delta = high_delta
        self.row_delta = low_delta 

        self.pi_history = [self.pi[0]]
        self.reward_history = []
        self.conter = 0

    def _init_q_values(self):
        q_values = {}
        q_values = np.repeat(0.0, len(self.actions))
        return q_values

    def act(self, q_values=None):
        # sum = 0.
        # for i in range(len(self.pi)):
        #     sum += self.pi[i]
        # self.pi = self.pi / sum
        action_id = np.random.choice(np.arange(len(self.pi)), p=self.pi)
        self.last_action_id = action_id
        action = self.actions[action_id]
        return action

    def observe(self, reward):
        self.reward_history.append(reward)
        self.q_values[self.last_action_id] = ((1.0 - self.alpha) * self.q_values[self.last_action_id]) + (self.alpha * reward)
        self._update_pi_average()
        self._update_pi()

    def _update_pi_average(self):
       self.conter += 1
       for aidx, _ in enumerate(self.pi):
           self.pi_average[aidx] = self.pi_average[aidx] + (1/self.conter)*(self.pi[aidx]-self.pi_average[aidx])
           if self.pi_average[aidx] > 1: self.pi_average[aidx] = 1
           if self.pi_average[aidx] < 0: self.pi_average[aidx] = 0

    def _update_pi(self):
       delta = self.decide_delta()
       max_action_id = np.argmax(self.q_values)
       for aidx, _ in enumerate(self.pi):
           if aidx == max_action_id:
               update_amount = delta
           else:
               update_amount = ((-delta)/(len(self.actions)-1))
           self.pi[aidx] = self.pi[aidx] + update_amount
           if self.pi[aidx] > 1: self.pi[aidx] = 1
           if self.pi[aidx] < 0: self.pi[aidx] = 0
       self.pi_history.append(self.pi[0])

    def decide_delta(self):
        """
            comfirm win or lose 
        """
        expected_value = 0
        expected_value_average = 0
        for aidx, _ in enumerate(self.pi):
            expected_value += self.pi[aidx]*self.q_values[aidx]
            expected_value_average += self.pi_average[aidx]*self.q_values[aidx]

        if expected_value > expected_value_average: # win
            return self.row_delta
        else:   # lose
            return self.high_delta
