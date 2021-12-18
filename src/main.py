import gym
import agent, transition #, memory, function_approx
env = gym.make('LunarLander-v2')

agent = agent.Agent(0.9)

# TODO: ↓ maak hier een functie van

def main(n_epi: int, steps: int, n_train: int, n_update: int):
    """
    :param steps: maximum amount of steps an episode is allowed to take
    :param n_epi: amount of episodes
    :param n_train: after n times train policy network
    :param n_update: after n steps update target network
    :return:
    """
    count=0
    for i_episode in range(n_epi):
        current_state = env.reset()
        for step in range(steps):
            print(count)
            env.render()
            # print(f"curr_state = {current_state}")

            # Train policy network
            if count % n_train == 0:
                agent.train()

            # Update target network
            if count % n_update == 0:
                agent.update_t_network()  # TODO: kan nog iets minder vak dan elke train

            # Select_action using policy network in agent.py
            action = agent.choose_action(current_state, 0.1) #env.action_space.sample()
            next_state, reward, done, info = env.step(action) # next_state is observation

            # Save transition in memory of agent
            t = transition.Transition(current_state, action, reward, next_state, done)
            agent.memory.record(t)

            current_state = next_state
            count += 1

            if done:
                print("Episode finished after {} timesteps".format(step+1))
                break
    agent.policy_network.save_network("policy_network")
    agent.target_network.save_network("target_network")
    env.close()


main(20, 60, 200)