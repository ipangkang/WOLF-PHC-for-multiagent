import numpy as np

class MatrixGame():

    def __init__(self):
        self.reward_matrix = self._create_reward_table()

    def step(self, actions):
        reward = self.cal_reward(actions)

        return None, reward

    def _create_reward_table(self):
        reward_matrix = [
                            [[1, -1], [-1, 1]],
                            [[-1, 1], [1, -1]]
                        ]

        return reward_matrix

    # 输出应该是一个与actions数组长度
    def cal_reward(self, actions):

        M = []
        D = []
        L = []
        reward = []
        # k1 , k2
        k1 = 2
        k2 = 3
        user_num = len(actions)
        # # D2D cell 用户的信道增益 (假设的是所有的用户的D2D信道增益，
        # # 在其中可能有一些用户并没有选择D2D方式)
        # Gd = np.ones(user_num)
        #
        # # Macrocell 用户的信道增益(假设的是所有用户的macrocell 信道增益，
        # # 在其中可能有一些用户并没有选择macrocell方式)
        # Gm = np.ones(user_num)
        #
        # # small cell 用户的信道增益(假设的是所有用户的small cell 信道增益，
        # # 在其中可能有一些用户并没有选择small cell方式)
        # Gl = np.ones(user_num)
        Gd = np.random.uniform(0, 2, [user_num, k1])
        Gm = np.random.uniform(0, 1, user_num)
        Gl = np.random.uniform(4, 6, [user_num, k2])

        Bm = 1  # macrocell 的 bandwidth
        Bd = 1  # D2D cluster bandwidth
        Bl = 1  # small cell bandwidth

        # Pm = 1  # macrowave 的 power
        # Pl = np.ones(user_num)   #D2D 的 power
        # Pd = np.ones(user_num)   #small cell 的 power
        Pm = 1
        Pd = np.random.uniform(2, 4, k1)
        Pl = np.random.uniform(2, 4, k2)

        sigma = 1  # macrowave 的 AWGN Power
        for i in range(len(actions)):
            if actions[i] == 0:
                M.append(i)
            elif 0 < actions[i] <= k1:
                D.append(i)
            else:
                L.append(i)

        for i in range(len(actions)):
            r = 0
            if actions[i] == 0:
                r = Bm * np.log2(np.min(Gm[M]) * Pm / sigma + 1)
            elif 0 < actions[i] <= k1:
                sum_d = 0
                # D.append(index)
                for j in range(k1):
                    sum_d += Gd[i, j]
                r = Bd * np.log2(1 + Gd[i, actions[i] - 1] / (sum_d - Gd[i, actions[i] - 1] + sigma))
            else:
                sum_l = 0
                # D.append(index)
                for j in range(k1):
                    sum_l += Gl[i, j]
                r = Bl * np.log2(1 + Gl[i, actions[i] - 1 - k1] / (sum_l - Gl[i, actions[i] - 1 - k1] + sigma))
            reward.append(r)
        return reward
