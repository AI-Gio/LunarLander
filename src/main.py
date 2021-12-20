import gym
import agent, transition  # , memory, function_approx
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('LunarLander-v2')
np.random.seed(7)


def main(n_epi: int, steps: int, n_train: int, n_update: int, n_decay: int, decay_r: float, load_m: bool):
    """
    :param steps: maximum amount of steps an episode is allowed to take
    :param n_epi: amount of episodes
    :param n_train: after n times train policy network
    :param n_update: after n steps update target network
    :param n_decay: after n episodes
    :param decay_r: the rate of how much epsilon decreases
    :return:
    """
    rewards = np.array([])
    count = 1  # counter across all episodes

    if load_m:
        agent.policy_network.load_network("policy_network")
        agent.target_network.load_network("target_network")

    for i_episode in range(n_epi):
        current_state = env.reset()
        print(f"n episodes: {i_episode}")

        # Safe network after 25 episodes
        if i_episode % 25 == 0 and i_episode > 1:
            print("Saved networks")
            agent.policy_network.save_network("policy_network")
            agent.target_network.save_network("target_network")

        # Epsilon decay
        if i_episode % n_decay == 0:
            agent.epsilon *= decay_r
            print(f"agent epsilon: {agent.epsilon}")

        for step in range(steps):
            env.render()

            # Train policy network
            if count % n_train == 0:
                agent.train()

            # Update target network
            if count % n_update == 0 and i_episode > 1:
                print(count)
                agent.update_t_network()  # TODO: kan nog iets minder vak dan elke train

            if count % 2000 == 0:
                print(f"rewards freq ^ 50: {len([i for i in rewards if i > 50])}")

            # Select_action using policy network in agent.py
            action = agent.choose_action(current_state)  # env.action_space.sample()
            next_state, reward, done, info = env.step(action)  # next_state is observation

            rewards = np.append(rewards, reward)

            # Save transition in memory of agent
            t = transition.Transition(current_state, action, reward, next_state, done)
            agent.memory.record(t)

            current_state = next_state
            count += 1

            if done:
                # agent.train()
                print("Episode finished after {} timesteps".format(step + 1))
                break

    agent.policy_network.save_network("policy_network")
    agent.target_network.save_network("target_network")
    env.close()

    plt.plot(rewards)
    plt.ylabel("reward over time")
    plt.show()

# bron: https://medium.datadriveninvestor.com/training-the-lunar-lander-agent-with-deep-q-learning-and-its-variants-2f7ba63e822c

epsilon, discount = [0.5, 0.9]  # begin epsilon: 0.5
epochs, batch_size, learning_rate, tau = [1, 64, 0.001, 0.7]
memory_size = 10000

agent = agent.Agent(discount=discount, epsilon=epsilon, tau=tau, batch_size=batch_size,
                    epochs=epochs, memory_size=memory_size, learning_rate=learning_rate)
main(n_epi=800, steps=300, n_train=75, n_update=20, n_decay=1, decay_r=0.995, load_m=False)
