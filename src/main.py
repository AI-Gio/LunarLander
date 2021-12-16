import gym
import agent, transition #, memory, function_approx
env = gym.make('LunarLander-v2')

agent = agent.Agent(0.9)

# TODO: ↓ maak hier een functie van


def main(n_epi: int, batch_size, loop_timer:int):
    """

    :param n_epi: amount of episodes
    :param batch_size: how much transitions to be trained over
    :param loop_timer: after n times train
    :return:
    """
    count=0
    for i_episode in range(n_epi):
        current_state = env.reset()
        for step in range(1000):
            env.render()
            print(current_state)

            # Train policy network
            if count % loop_timer:
                agent.train(batch_size, 20)
                agent.update_t_network(0.1) # TODO: kan nog iets minder vak dan elke train

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
    env.close()


main(20, 60, 200)