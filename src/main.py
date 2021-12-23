import gym
import agent
import transition
import numpy as np
import matplotlib.pyplot as plt
import os
import time

env = gym.make('LunarLander-v2')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_status(epi, eps, ep_l, time):
    """
    Prints the current status of the training simulation.

    :param epi: Current episode number
    :param eps: Current agent epsilon
    :param ep_l: Amount of taken steps of previous episode
    :param time: Duration of previous episode in seconds
    """
    print('\r', f"Episode nr: \033[93m{epi:>4}\033[35m", end=", ")
    print(f"Epsilon: \033[93m{round(eps, 4):0<6}\033[35m", end=", ")
    print(f"Previous episode length: \033[93m{ep_l:>3}\033[35m", end=", ")
    print(f"Previous episode duration: \033[93m{round(time, 5):0<7}s\033[35m", end="")


def main(n_epi: int, steps: int, n_train: int, n_update: int, n_decay: int, decay_r: float, load_m: bool):
    """
    The main function where the simulation is run. All code is run and called from here

    :param steps: maximum amount of steps an episode is allowed to take
    :param n_epi: amount of episodes
    :param n_train: after n times train policy network
    :param n_update: after n steps update target network
    :param n_decay: after n episodes
    :param decay_r: the rate of how much epsilon decreases
    :param load_m: load a trained model or not
    """
    tot_rewards = np.array([])
    count = 1  # counter across all episodes
    ep_length, duration = 0, 0
    step = 0

    if load_m:
        agent.policy_network.load_network("policy_network")
        agent.target_network.load_network("target_network")

    for i_episode in range(n_epi):
        if i_episode % 5 == 0:
            print_status(i_episode+1, agent.epsilon, ep_length, duration)

        start = time.time()  # keeps track of duration
        current_state = env.reset()
        tot_reward = 0  # total reward of 1 episode

        # Epsilon decay
        if i_episode % n_decay == 0:
            agent.epsilon *= decay_r
            if agent.epsilon < 0.1:
                agent.epsilon = 0.1

        for step in range(steps):
            # Train policy network
            if count % n_train == 0:
                agent.train_efficient()

            # Update target network
            if count % n_update == 0 and i_episode > 0:
                agent.update_t_network()

            # Select_action using policy network in agent.py
            action = agent.choose_action(np.array([current_state]))
            next_state, reward, done, info = env.step(action)  # next_state is observation

            # Add reward to total reward
            tot_reward += reward

            # Save transition in memory of agent
            t = transition.Transition(current_state, action, reward, next_state, done)
            agent.memory.record(t)

            current_state = next_state
            count += 1
            if done:
                break

        tot_rewards = np.append(tot_rewards, tot_reward)
        ep_length = step + 1
        duration = time.time() - start

    # Save networks to file
    agent.policy_network.save_network("policy_network")
    agent.target_network.save_network("target_network")
    env.close()

    # plot total rewards over all episodes
    plt.plot(list(tot_rewards))
    plt.ylabel("Total reward over time")
    plt.xlabel("Episode")
    plt.show()

# bron: https://medium.datadriveninvestor.com/training-the-lunar-lander-agent-with-deep-q-learning-and-its-variants-2f7ba63e822c


if __name__ == "__main__":
    # All parameters
    load_network = False
    epsilon, e_decay_rate, discount, memory_size = [0.9, 0.998, 0.95, 9000]  # Q learning parameters
    epochs, batch_size, learning_rate, tau = [3, 128, 0.001, 1]  # Network parameters
    episode_amount, max_steps, train_speed, t_update_speed = [500, 300, 200, 300]  # Simulation parameters

    print(f"\033[93m[PARAMETERS]\033[35m\n"
          f"Start epsilon: \033[93m{epsilon}\033[35m, Epsilon decay rate: \033[93m{e_decay_rate}\033[35m, Gamma: \033[93m{discount}\033[35m, Memory size: \033[93m{memory_size}\033[35m.\n"
          f"Epochs: \033[93m{epochs}\033[35m, Batch size: \033[93m{batch_size}\033[35m, Learning rate: \033[93m{learning_rate}\033[35m, Tau: \033[93m{tau}\033[35m.\n"
          f"Amount of episodes: \033[93m{episode_amount}\033[35m\n")
    agent = agent.Agent(discount=discount, epsilon=epsilon, tau=tau, batch_size=batch_size,
                        epochs=epochs, memory_size=memory_size, learning_rate=learning_rate)
    main(n_epi=episode_amount, steps=max_steps, n_train=train_speed, n_update=t_update_speed, n_decay=1,
         decay_r=e_decay_rate, load_m=load_network)
