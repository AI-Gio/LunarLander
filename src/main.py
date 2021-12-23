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


def print_status(epi, eps, ep_l, avg_time, rem_time):
    """
        Prints the current status of the training simulation.
    :param epi: Current episode number
    :param eps: Current agent epsilon
    :param ep_l: Amount of taken steps of previous episode
    :param avg_time: Average time of an episode
    :param rem_time: Remaining time of the simulation
    """
    print('\r', f"Episode nr: \033[93m{epi:>4}\033[35m", end=", ")
    print(f"Epsilon: \033[93m{round(eps, 4):0<6}\033[35m", end=", ")
    print(f"Previous episode length: \033[93m{ep_l:>3}\033[35m", end=", ")
    print(f"Average episode duration: \033[93m{round(avg_time, 4):0<6}s\033[35m", end=", ")
    print(f"Remaining time: \033[93m{round(rem_time, 1):0<3}s\033[35m", end="")


def main(n_epi: int, steps: int, n_train: int, n_update: int, load_nn: bool):
    """
    The main function where the simulation is run. All code is run and called from here

    :param steps: maximum amount of steps an episode is allowed to take
    :param n_epi: amount of episodes
    :param n_train: after n times train policy network
    :param n_update: after n steps update target network
    :param load_nn: load a trained model or not
    """
    tot_rewards = np.array([])
    count = 1  # counter across all episodes
    ep_length, duration = 0, 0
    step = 0
    average_duration = 0

    if load_nn:
        agent.policy_network.load_network("policy_network")
        agent.target_network.load_network("target_network")

    for i_episode in range(n_epi):
        remaining_time = (n_epi-i_episode)*average_duration
        if i_episode % 5 == 0:  # print status every 5 episodes
            print_status(i_episode+1, agent.epsilon, ep_length, average_duration, remaining_time)

        start = time.time()  # keeps track of duration
        current_state = env.reset()
        tot_reward = 0  # total reward of 1 episode

        # Epsilon decay
        if i_episode < 50:
            agent.epsilon = 0.5
        elif i_episode < 100:
            agent.epsilon = 0.3
        elif i_episode < 300:
            agent.epsilon = 0.2
        elif i_episode < 500:
            agent.epsilon = 0.1
        elif i_episode < 1000:
            agent.epsilon = 0.5
        else:
            agent.epsilon = 0.1

        for step in range(steps):
            # Train policy network
            if count % n_train == 0:
                agent.train_efficient()

            # Update target network
            if count % n_update == 0:
                agent.update_t_network()

            # Select_action using policy network in agent.py
            action = agent.choose_action(np.array([current_state]))
            next_state, reward, done, info = env.step(action)

            # Add reward to total reward
            tot_reward += reward

            # Save transition in memory of agent
            t = transition.Transition(current_state, action, reward, next_state, done)
            agent.memory.record(t)

            current_state = list(next_state)
            count += 1
            if done:
                break

        tot_rewards = np.append(tot_rewards, tot_reward)

        ep_length = step + 1
        duration = time.time() - start
        average_duration = ((average_duration * i_episode) + duration) / (i_episode + 1)

    # Save networks to file
    agent.policy_network.save_network("policy_network")
    agent.target_network.save_network("target_network")
    env.close()

    # plot total rewards over all episodes
    plt.plot(tot_rewards)
    plt.ylabel("Total reward over time")
    plt.xlabel("Episode")
    plt.show()


if __name__ == "__main__":
    # Source of parameter choices:
    # https://medium.datadriveninvestor.com/training-the-lunar-lander-agent-with-deep-q-learning-and-its-variants-2f7ba63e822c

    # All parameters
    load_network = False
    epsilon, discount, memory_size = [0.5, 0.9, 10000]  # Q learning parameters
    epochs, batch_size, learning_rate, tau = [3, 64, 0.001, 0.7]  # Network parameters
    episode_amount, max_steps, train_speed, t_update_speed = [10000, 1000, 10, 20]  # Simulation parameters

    print(f"\033[93m[PARAMETERS]\033[35m\n"
          f"Start epsilon: \033[93m{epsilon}\033[35m, Gamma: \033[93m{discount}\033[35m, Memory size: \033[93m{memory_size}\033[35m.\n"
          f"Epochs: \033[93m{epochs}\033[35m, Batch size: \033[93m{batch_size}\033[35m, Learning rate: \033[93m{learning_rate}\033[35m, Tau: \033[93m{tau}\033[35m.\n"
          f"Amount of episodes: \033[93m{episode_amount}\033[35m\n")

    agent = agent.Agent(discount=discount, epsilon=epsilon, tau=tau, batch_size=batch_size,
                        epochs=epochs, memory_size=memory_size, learning_rate=learning_rate)
    main(n_epi=episode_amount, steps=max_steps, n_train=train_speed, n_update=t_update_speed, load_nn=load_network)
