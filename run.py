import numpy as np
import matplotlib.pyplot as plt
from wolf_agent import WoLFAgent 
from matrix_game import MatrixGame
import pandas as pd

if __name__ == '__main__':
    nb_episode = 1000

    actions = np.arange(3)
    agent1 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002) 
    agent2 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002)
    agent3 = WoLFAgent(alpha=0.1, actions=actions, high_delta=0.0004, low_delta=0.0002)

    game = MatrixGame()
    for episode in range(nb_episode):
        actions = []
        action1 = agent1.act()
        action2 = agent2.act()
        action3 = agent3.act()
        actions.append(action1)
        actions.append(action2)
        actions.append(action3)
        _, reward = game.step(actions)

        agent1.observe(reward=reward[0])
        agent2.observe(reward=reward[1])
        agent3.observe(reward=reward[2])

    print(agent1.q_values)
    print(agent2.q_values)
    print(agent3.q_values)
    # plt.plot(np.arange(len(agent1.pi_history)),agent1.pi_history, label="agent1's pi(0)")
    # plt.plot(np.arange(len(agent2.pi_history)),agent2.pi_history, label="agent2's pi(0)")
    # plt.plot(np.arange(len(agent3.pi_history)),agent3.pi_history, label="agent3's pi(0)")
    # plt.ylim(0, 1)
    # plt.xlabel("episode")
    # plt.ylabel("pi(0)")
    # plt.legend()
    # plt.savefig("result.png")
    # plt.show()
